# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU golden tests: every rewrite the TTNN port relies on, checked against the upstream reference.

The mirror in ``golden_dataflow.py`` uses the rearranged weights (``tt/weights.py``) and
host inputs (``tt/preprocessing.py``) exactly as the device does, in float32.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.timestral.tests import golden_dataflow as flow
from models.experimental.timestral.tests.golden_helpers import SEED, load_checkpoint, log_golden, series, tiny_model
from models.experimental.timestral.tt.preprocessing import (
    TimestralHostInputs,
    _block_diagonal,
    forecast_token_layout,
    postprocess_head,
    prepare_inputs,
    uniform_group_size,
)
from models.experimental.timestral.tt.weights import AttentionWeights, MlpWeights, TimestralWeights, rope_pair_order

from timestral.layers import TwoStreamMask  # noqa: E402
from timestral.masks import (  # noqa: E402
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_variate_mask,
)

R, L, H = 6, 5, 3
S = L + H
TOL = dict(rtol=1e-5, atol=1e-5)


def _host_inputs(model, x, *, group_size=1, time_mask=None, variate_mask=None) -> TimestralHostInputs:
    positions = build_forecast_positions(L, H, torch.zeros(x.shape[0], dtype=torch.long))
    cos, sin = model.transformer.rope(x, positions)
    return TimestralHostInputs(
        tokens=torch.zeros(x.shape[0], S, 2 * model.config.patch_size),
        num_context_patches=L,
        horizon_patches=H,
        group_size=group_size,
        cos=cos,
        sin=sin,
        time_mask=time_mask,
        variate_mask=variate_mask,
        anchor_loc=torch.zeros(x.shape[0], 1),
        anchor_scale=torch.ones(x.shape[0], 1),
    )


def _x(model, rows=R) -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(rows, S, model.config.d_model)


def test_qkv_row_permutation_is_exact():
    model = tiny_model()
    attn, norm = model.transformer.layers[0].attns[0], model.transformer.layers[0].attn_norms[0]
    x = _x(model)
    with torch.no_grad():
        ref_q, ref_k, ref_v = attn._qkv(norm(x))  # (B, S, H, Dh)
    w = AttentionWeights.from_reference(attn, norm)
    q, k, v = flow.split_heads(torch.nn.functional.linear(flow.rms_unit(x, w.eps), w.wqkv), w.num_heads)
    for ref, got in ((rope_pair_order(ref_q), q), (rope_pair_order(ref_k), k), (ref_v, v)):
        torch.testing.assert_close(got, ref.transpose(1, 2), **TOL)


def test_norm_gain_folding():
    model = tiny_model()
    layer = model.transformer.layers[0]
    x = _x(model)
    w = MlpWeights.from_reference(layer.mlp, layer.mlp_norm)
    with torch.no_grad():
        expected = layer.mlp(layer.mlp_norm(x))
    torch.testing.assert_close(flow.mlp(w, x), expected, **TOL)


def test_time_attention_single_causal_pass_matches_two_stream():
    """With every key valid, the context-only time mask is exactly ``tril``: one causal pass, no mask."""
    model = tiny_model()
    attn, norm = model.transformer.layers[0].attns[0], model.transformer.layers[0].attn_norms[0]
    x = _x(model)
    key_valid = torch.ones(R, L, dtype=torch.bool)
    time_mask = build_forecast_time_mask(L, H, key_valid, x.device)
    torch.testing.assert_close(time_mask, torch.tril(torch.ones(S, S, dtype=torch.bool)).expand(R, 1, S, S))
    prepared = _host_inputs(model, x)
    with torch.no_grad():
        expected = attn(norm(x), TwoStreamMask(mask=time_mask, enc_len=L), prepared.cos, prepared.sin)
    torch.testing.assert_close(
        flow.attention(AttentionWeights.from_reference(attn, norm), x, prepared), expected, **TOL
    )


def test_time_attention_masked_single_pass_matches_two_stream():
    """Leading unobserved patches: one pass with the full mask equals the reference's two passes."""
    model = tiny_model()
    attn, norm = model.transformer.layers[0].attns[0], model.transformer.layers[0].attn_norms[0]
    x = _x(model)
    key_valid = torch.ones(R, L, dtype=torch.bool)
    key_valid[0, :2] = False
    key_valid[3, :1] = False
    time_mask = build_forecast_time_mask(L, H, key_valid, x.device)
    prepared = _host_inputs(model, x, time_mask=time_mask)
    with torch.no_grad():
        expected = attn(norm(x), TwoStreamMask(mask=time_mask, enc_len=L), prepared.cos, prepared.sin)
    torch.testing.assert_close(
        flow.attention(AttentionWeights.from_reference(attn, norm), x, prepared), expected, **TOL
    )


def test_variate_attention_unique_groups_is_value_path():
    """One row per series: the only visible key is the row itself, so attention returns V."""
    model = tiny_model()
    layer = model.transformer.layers[1]
    attn, norm = layer.attns[1], layer.attn_norms[1]
    x = _x(model)
    mask = build_two_stream_variate_mask(torch.arange(R), torch.ones(R, L, dtype=torch.bool), H)
    with torch.no_grad():
        expected = attn(norm(x), mask, None, None)
    got = flow.attention(AttentionWeights.from_reference(attn, norm), x, _host_inputs(model, x))
    torch.testing.assert_close(got, expected, **TOL)


@pytest.mark.parametrize("missing", [False, True], ids=["all_valid", "leading_missing"])
def test_variate_attention_blocks_match_full_mask(missing):
    """Series as contiguous blocks of V rows: blocked attention equals the reference's (N, 1, R, R) mask."""
    model = tiny_model()
    layer = model.transformer.layers[1]
    attn, norm = layer.attns[1], layer.attn_norms[1]
    x = _x(model)
    group_ids = torch.arange(2).repeat_interleave(3)
    key_valid = torch.ones(R, L, dtype=torch.bool)
    if missing:
        key_valid[1, :2] = False
    full = build_two_stream_variate_mask(group_ids, key_valid, H)
    with torch.no_grad():
        expected = attn(norm(x), full, None, None)
    variate_mask = _block_diagonal(full, 3) if missing else None
    prepared = _host_inputs(model, x, group_size=uniform_group_size(group_ids), variate_mask=variate_mask)
    got = flow.attention(AttentionWeights.from_reference(attn, norm), x, prepared)
    torch.testing.assert_close(got, expected, **TOL)


def test_forecast_token_layout_matches_concat():
    model = tiny_model()
    rep = model.representation
    torch.manual_seed(SEED)
    patches = torch.randn(R, L, 2 * model.config.patch_size)
    with torch.no_grad():
        expected = torch.cat([rep.patch_proj(patches), rep.forecast_tokens(R, H)], dim=1)
        tokens = torch.cat([patches, torch.zeros(R, H, patches.shape[-1])], dim=1)
        keep, add = forecast_token_layout(rep.forecast_query, L, H)
        got = rep.patch_proj(tokens) * keep + add
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_uniform_group_size_rejects_ragged_packing(expect_error):
    assert uniform_group_size(torch.arange(4)) == 1
    assert uniform_group_size(torch.tensor([0, 0, 1, 1])) == 2
    with expect_error(NotImplementedError, "contiguous, equal-size block"):
        uniform_group_size(torch.tensor([0, 0, 1]))
    with expect_error(NotImplementedError, "contiguous, equal-size block"):
        uniform_group_size(torch.tensor([0, 1, 0, 1]))


def _assert_dataflow_matches(model, context, observed, group_ids, horizon, *, name, tol=TOL):
    with torch.no_grad():
        expected = model.predict_block(context, observed, group_ids, horizon)
    weights = TimestralWeights.from_reference(model)
    prepared = prepare_inputs(model.config, weights.inv_freq, context, observed, group_ids, horizon)
    with torch.no_grad():
        got = postprocess_head(model.config, flow.head_forward(weights, prepared), prepared)
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, **tol)
    log_golden(f"dataflow_{name}/median", got[..., model.config.median_index])


def test_dataflow_matches_predict_block_univariate():
    model = tiny_model()
    context = series(4, 1, 8 * model.config.patch_size)[:, 0]
    _assert_dataflow_matches(model, context, torch.ones_like(context, dtype=torch.bool), torch.arange(4), 3, name="uni")


def test_dataflow_matches_predict_block_missing_values():
    """NaN gaps, a late-starting row (positions shift) and left padding (masks uploaded)."""
    model = tiny_model()
    ps = model.config.patch_size
    context = series(4, 1, 8 * ps)[:, 0]
    observed = torch.ones_like(context, dtype=torch.bool)
    observed[0, 40:75] = False
    observed[2, : 2 * ps + 5] = False
    context = torch.where(observed, context, torch.zeros_like(context))
    _assert_dataflow_matches(model, context, observed, torch.arange(4), 2, name="missing")


def test_dataflow_matches_predict_block_multivariate():
    model = tiny_model()
    ps = model.config.patch_size
    context = series(3, 4, 6 * ps).reshape(12, -1)
    observed = torch.ones_like(context, dtype=torch.bool)
    observed[5, :ps] = False  # one variate starts a patch late
    context = torch.where(observed, context, torch.zeros_like(context))
    group_ids = torch.arange(3).repeat_interleave(4)
    _assert_dataflow_matches(model, context, observed, group_ids, 4, name="multi")


@pytest.mark.parametrize("name", ["Timestral-5M", "Timestral-209M"])
def test_dataflow_matches_predict_block_pretrained(name):
    model = load_checkpoint(name)
    ps = model.config.patch_size
    context = series(2, 2, 16 * ps).reshape(4, -1)
    observed = torch.ones_like(context, dtype=torch.bool)
    _assert_dataflow_matches(
        model, context, observed, torch.arange(2).repeat_interleave(2), 2, name=name, tol=dict(rtol=1e-4, atol=1e-3)
    )

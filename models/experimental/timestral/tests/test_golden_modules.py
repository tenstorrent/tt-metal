# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stubs: one test per Timestral module, oracle = the upstream reference (CPU, no device).

Each test checks the module's shape and invariants and logs its output stats.
"""

from __future__ import annotations

import torch

from models.experimental.timestral.tests.golden_helpers import SEED, log_golden, series, tiny_config, tiny_model

from timestral.layers import FusedSwiGLU, RMSNorm, ResidualBlock, SinCosRoPE, TwoStreamMask  # noqa: E402
from timestral.masks import (  # noqa: E402
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_variate_mask,
)
from timestral.model import monotone_quantiles  # noqa: E402
from timestral.scaler import causal_patch_loc_scale  # noqa: E402

R, L, H = 4, 6, 2  # rows, context patches, horizon patches
S = L + H


def _x(d: int) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(R, S, d)


def _time_inputs(model, x):
    time_mask = build_forecast_time_mask(L, H, torch.ones(R, L, dtype=torch.bool), x.device)
    positions = build_forecast_positions(L, H, torch.zeros(R, dtype=torch.long))
    cos, sin = model.transformer.rope(x, positions)
    return TwoStreamMask(mask=time_mask, enc_len=L), cos, sin


def test_rms_norm_golden():
    norm = RMSNorm(32, eps=1e-6)
    x = _x(32)
    y = norm(x)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(R, S), rtol=1e-4, atol=1e-4)
    log_golden("rms_norm/out", y)


def test_patch_embedding_residual_block_golden():
    torch.manual_seed(SEED)
    block = ResidualBlock(64, 4 * 32, 32, use_bias=True).eval()
    x = torch.randn(R, L, 64)
    y = block(x)
    assert y.shape == (R, L, 32)
    log_golden("patch_embedding/out", y)


def test_swiglu_golden():
    torch.manual_seed(SEED)
    mlp = FusedSwiGLU(32, 64, use_bias=False).eval()
    y = mlp(_x(32))
    assert y.shape == (R, S, 32)
    log_golden("swiglu/out", y)


def test_partial_rope_golden():
    rope = SinCosRoPE(dim=64, base=10000.0, rotary_dim=32)
    positions = torch.arange(S).unsqueeze(0).expand(R, -1)
    cos, sin = rope(torch.zeros(R, S, 64), positions)
    assert cos.shape == sin.shape == (R, S, 64)
    # Only the first rotary_dim / 2 frequencies of each half rotate; the rest are the identity.
    for half in (slice(16, 32), slice(48, 64)):
        torch.testing.assert_close(cos[..., half], torch.ones(R, S, 16))
        torch.testing.assert_close(sin[..., half], torch.zeros(R, S, 16))
    log_golden("rope/sin", sin)


def test_time_attention_golden():
    model = tiny_model()
    attn, norm = model.transformer.layers[0].attns[0], model.transformer.layers[0].attn_norms[0]
    assert attn.is_time
    x = _x(model.config.d_model)
    mask, cos, sin = _time_inputs(model, x)
    with torch.no_grad():
        y = attn(norm(x), mask, cos, sin)
    assert y.shape == x.shape
    log_golden("time_attention/out", y)


def test_variate_attention_golden():
    model = tiny_model()
    layer = model.transformer.layers[1]
    attn, norm = layer.attns[1], layer.attn_norms[1]
    assert not attn.is_time
    x = _x(model.config.d_model)
    group_ids = torch.tensor([0, 0, 1, 1])
    mask = build_two_stream_variate_mask(group_ids, torch.ones(R, L, dtype=torch.bool), H)
    with torch.no_grad():
        y = attn(norm(x), mask, None, None)
    assert y.shape == x.shape
    log_golden("variate_attention/out", y)


def test_transformer_blocks_golden():
    model = tiny_model()
    x = _x(model.config.d_model)
    mask, cos, sin = _time_inputs(model, x)
    variate_mask = build_two_stream_variate_mask(torch.arange(R), torch.ones(R, L, dtype=torch.bool), H)
    for kind, layer in zip(model.config.block_schedule, model.transformer.layers):
        with torch.no_grad():
            y = layer(x, mask, variate_mask, cos, sin)
        assert y.shape == x.shape
        log_golden(f"block_{kind}/out", y)


def test_quantile_head_is_monotone():
    model = tiny_model()
    torch.manual_seed(SEED)
    with torch.no_grad():
        preds = model.head(torch.randn(R, H, model.config.d_model))
    assert preds.shape == (R, H * model.config.patch_size, model.config.num_quantiles)
    assert (preds.diff(dim=-1) >= 0).all()
    raw = torch.randn(R, 3, model.config.num_quantiles)
    q = monotone_quantiles(raw, model.config.quantile_knots, model.config.quantile_width_factor)
    torch.testing.assert_close(q[..., model.config.median_index], raw[..., model.config.median_index])
    log_golden("quantile_head/out", preds)


def test_causal_patch_scaler_matches_naive():
    """Each patch uses the mean and (Bessel) std of every observed value up to its last step."""
    ps = 8
    x = series(1, 2, 4 * ps)[0]
    observed = torch.ones_like(x, dtype=torch.bool)
    observed[0, 3:7] = False
    loc, scale = causal_patch_loc_scale(x.unsqueeze(1), observed.unsqueeze(1), ps, 1e-6, False)
    for r in range(2):
        for p in range(4):
            seen = x[r, : (p + 1) * ps][observed[r, : (p + 1) * ps]].double()
            torch.testing.assert_close(loc[r, 0, p * ps].double(), seen.mean(), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(scale[r, 0, p * ps].double(), seen.std(), rtol=1e-6, atol=1e-6)


def test_predict_block_golden():
    model = tiny_model()
    cfg = tiny_config()
    context = series(R, 1, 8 * cfg.patch_size)[:, 0]
    with torch.no_grad():
        preds = model.predict_block(context, torch.ones_like(context, dtype=torch.bool), torch.arange(R), H)
    assert preds.shape == (R, H * cfg.patch_size, cfg.num_quantiles)
    assert torch.isfinite(preds).all() and (preds.diff(dim=-1) >= 0).all()
    log_golden("predict_block/median", preds[..., cfg.median_index])

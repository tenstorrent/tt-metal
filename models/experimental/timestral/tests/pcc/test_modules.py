# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests: TTNN Timestral modules and full decode blocks against the upstream reference (single chip)."""

from __future__ import annotations

import pytest
import torch

from models.experimental.timestral.tests.golden_helpers import (
    SEED,
    load_checkpoint,
    log_golden,
    series,
    tiny_model,
)
from models.experimental.timestral.tt.preprocessing import TimestralHostInputs

from timestral.forecaster import TimestralForecaster  # noqa: E402
from timestral.layers import TwoStreamMask  # noqa: E402
from timestral.masks import (  # noqa: E402
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_variate_mask,
)

R, L, H = 4, 5, 3
S = L + H


def _check_pcc(name: str, expected: torch.Tensor, got: torch.Tensor, pcc: float) -> float:
    from tests.ttnn.utils_for_testing import comp_pcc

    assert got.shape == expected.shape, f"{name}: shape {tuple(got.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(got).all(), f"{name}: non-finite output"
    passing, value = comp_pcc(expected.float(), got.float(), pcc)
    print(f"\n[PCC] {name}: {value}")
    assert passing, f"{name}: PCC {value} < {pcc}"
    return value


def _tt_model(mesh_device, model, precision=None):
    from models.experimental.timestral.tt.model import TtTimestral

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip model")
    return TtTimestral.from_reference(mesh_device, model, precision)


def _host_inputs(model, rows, *, ll=L, h=H, group_size=1, key_valid=None, first_obs=None) -> TimestralHostInputs:
    """Host inputs for module tests on arbitrary hidden states (the tokens are unused)."""
    from models.experimental.timestral.tt.preprocessing import _block_diagonal, rope_tables

    first_obs = torch.zeros(rows, dtype=torch.long) if first_obs is None else first_obs
    positions = build_forecast_positions(ll, h, first_obs)
    if bool((first_obs == first_obs[0]).all()):
        positions = positions[:1]
    cos, sin = rope_tables(positions, model.transformer.rope.inv_freq)
    time_mask = variate_mask = None
    if key_valid is not None:
        time_mask = build_forecast_time_mask(ll, h, key_valid, key_valid.device)
        if group_size > 1:
            group_ids = torch.arange(rows // group_size).repeat_interleave(group_size)
            variate_mask = _block_diagonal(build_two_stream_variate_mask(group_ids, key_valid, h), group_size)
    return TimestralHostInputs(
        tokens=torch.zeros(rows, ll + h, 2 * model.config.patch_size),
        num_context_patches=ll,
        horizon_patches=h,
        group_size=group_size,
        cos=cos,
        sin=sin,
        time_mask=time_mask,
        variate_mask=variate_mask,
        anchor_loc=torch.zeros(rows, 1),
        anchor_scale=torch.ones(rows, 1),
    )


def _run_on_device(tt, module, x: torch.Tensor, prepared: TimestralHostInputs, *, decoder=False) -> torch.Tensor:
    """Host ``(R, S, D)`` -> module on the padded, flat device layout -> host ``(R, S, D)``,
    or ``(R, h, D)`` when ``decoder`` cuts the module to the decoder slots."""
    import ttnn

    from models.experimental.timestral.tt.common import upload

    device_inputs = tt.upload_inputs(prepared)
    att = device_inputs.attention
    rows, seq, d = x.shape
    padded = torch.zeros(rows, att.padded_seq, d)
    padded[:, :seq] = x
    x_tt = upload(tt.device, padded.reshape(1, 1, -1, d))
    if decoder:
        out = module(x_tt, att, decoder=device_inputs.decoder)
        host = ttnn.to_torch(out).float().reshape(rows, prepared.horizon_patches, d)
    else:
        out = module(x_tt, att)
        host = ttnn.to_torch(out).float().reshape(rows, att.padded_seq, d)[:, :seq]
    ttnn.deallocate(x_tt)
    ttnn.deallocate(out)
    tt.deallocate_inputs(device_inputs)
    return host


def _run_split(tt, layer, x: torch.Tensor, prepared: TimestralHostInputs, *, last: bool) -> torch.Tensor:
    """Host ``(R, S, D)`` -> one block on the split context / decoder streams -> host ``(R, S, D)``,
    or ``(R, h, D)`` with ``last``."""
    import ttnn

    from models.experimental.timestral.tt.common import upload

    device_inputs = tt.upload_inputs(prepared)
    streams = device_inputs.split
    assert streams is not None, "shape does not take the split streams"
    rows, _, d = x.shape
    ll = prepared.num_context_patches
    context = torch.zeros(rows, streams.context.padded_seq, d)
    context[:, :ll] = x[:, :ll]
    x_c = upload(tt.device, context.reshape(1, 1, -1, d))
    x_d = upload(tt.device, x[:, ll:].reshape(1, 1, -1, d))
    out = tt.run_layer(layer, (x_c, x_d), device_inputs, last=last)
    y_c, y_d = (None, out) if last else out
    host = ttnn.to_torch(y_d).float().reshape(rows, prepared.horizon_patches, d)
    ttnn.deallocate(y_d)
    if y_c is not None:
        host_c = ttnn.to_torch(y_c).float().reshape(rows, streams.context.padded_seq, d)[:, :ll]
        ttnn.deallocate(y_c)
        host = torch.cat([host_c, host], dim=1)
    tt.deallocate_inputs(device_inputs)
    return host


def _x(model, rows=R, seq=S) -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(rows, seq, model.config.d_model)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_patch_embedding_pcc(mesh_device):
    import ttnn

    from models.experimental.timestral.tt.common import upload

    model = tiny_model()
    tt = _tt_model(mesh_device, model)
    torch.manual_seed(SEED)
    patches = torch.randn(R, 32, 2 * model.config.patch_size)
    with torch.no_grad():
        expected = model.representation.patch_proj(patches)
    out = tt._patch_proj(upload(mesh_device, patches.reshape(1, 1, -1, patches.shape[-1])))
    got = ttnn.to_torch(out).float().reshape(expected.shape)
    _check_pcc("patch_embedding", expected, got, 0.999)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_mlp_pcc(mesh_device):
    model = tiny_model()
    tt = _tt_model(mesh_device, model)
    layer = model.transformer.layers[0]
    x = _x(model)
    with torch.no_grad():
        expected = layer.mlp(layer.mlp_norm(x))
    got = _run_on_device(tt, lambda t, _att: tt._layers[0].mlp(t), x, _host_inputs(model, R))
    _check_pcc("mlp", expected, got, 0.999)


@pytest.mark.parametrize("positions", ["shared", "per_row", "masked"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_time_attention_pcc(mesh_device, positions):
    """Shared positions use ttnn's rotary op; per-row positions and masks take the general path."""
    model = tiny_model()
    tt = _tt_model(mesh_device, model)
    attn, norm = model.transformer.layers[0].attns[0], model.transformer.layers[0].attn_norms[0]
    x = _x(model)
    key_valid = None
    first_obs = torch.zeros(R, dtype=torch.long)
    if positions == "per_row":
        first_obs = torch.tensor([0, 1, 0, 2])
    if positions == "masked":
        key_valid = torch.ones(R, L, dtype=torch.bool)
        key_valid[1, :2] = False
    prepared = _host_inputs(model, R, key_valid=key_valid, first_obs=first_obs)
    cos, sin = prepared.cos.expand(R, -1, -1), prepared.sin.expand(R, -1, -1)
    time_mask = build_forecast_time_mask(
        L, H, torch.ones(R, L, dtype=torch.bool) if key_valid is None else key_valid, x.device
    )
    with torch.no_grad():
        expected = attn(norm(x), TwoStreamMask(mask=time_mask, enc_len=L), cos, sin)
    got = _run_on_device(tt, tt._layers[0].attns[0], x, prepared)
    _check_pcc(f"time_attention/{positions}", expected, got, 0.999)


@pytest.mark.parametrize(
    "group_size,masked", [(1, False), (3, False), (3, True)], ids=["unique", "blocks", "blocks_masked"]
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_variate_attention_pcc(mesh_device, group_size, masked):
    model = tiny_model()
    tt = _tt_model(mesh_device, model)
    layer = model.transformer.layers[1]
    attn, norm = layer.attns[1], layer.attn_norms[1]
    rows = 2 * group_size
    x = _x(model, rows)
    key_valid = torch.ones(rows, L, dtype=torch.bool)
    if masked:
        key_valid[1, :2] = False
    group_ids = torch.arange(rows // group_size).repeat_interleave(group_size)
    full = build_two_stream_variate_mask(group_ids, key_valid, H)
    with torch.no_grad():
        expected = attn(norm(x), full, None, None)
    prepared = _host_inputs(model, rows, group_size=group_size, key_valid=key_valid if masked else None)
    got = _run_on_device(tt, tt._layers[1].attns[1], x, prepared)
    _check_pcc(f"variate_attention/{group_size}{'_masked' if masked else ''}", expected, got, 0.999)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_transformer_blocks_pcc(mesh_device):
    model = tiny_model()
    tt = _tt_model(mesh_device, model)
    x = _x(model)
    time_mask = build_forecast_time_mask(L, H, torch.ones(R, L, dtype=torch.bool), x.device)
    prepared = _host_inputs(model, R)
    cos, sin = prepared.cos.expand(R, -1, -1), prepared.sin.expand(R, -1, -1)
    variate_mask = build_two_stream_variate_mask(torch.arange(R), torch.ones(R, L, dtype=torch.bool), H)
    for i, (kind, layer) in enumerate(zip(model.config.block_schedule, model.transformer.layers)):
        with torch.no_grad():
            expected = layer(x, TwoStreamMask(mask=time_mask, enc_len=L), variate_mask, cos, sin)
        got = _run_on_device(tt, tt._layers[i], x, prepared)
        _check_pcc(f"block_{kind}", expected, got, 0.999)


@pytest.mark.parametrize("group_size", [1, 2], ids=["univariate", "multivariate"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_last_block_decoder_slots_pcc(mesh_device, group_size):
    """The last block, cut to the decoder slots after its time attention, matches the full block there."""
    model = tiny_model()
    tt = _tt_model(mesh_device, model)
    x = _x(model)
    key_valid = torch.ones(R, L, dtype=torch.bool)
    key_valid[1, :2] = False
    group_ids = torch.arange(R // group_size).repeat_interleave(group_size)
    time_mask = build_forecast_time_mask(L, H, key_valid, x.device)
    prepared = _host_inputs(model, R, group_size=group_size, key_valid=key_valid)
    cos, sin = prepared.cos.expand(R, -1, -1), prepared.sin.expand(R, -1, -1)
    variate_mask = build_two_stream_variate_mask(group_ids, key_valid, H)
    layer = model.transformer.layers[-1]
    with torch.no_grad():
        expected = layer(x, TwoStreamMask(mask=time_mask, enc_len=L), variate_mask, cos, sin)[:, L:]
    got = _run_on_device(tt, tt._layers[-1], x, prepared, decoder=True)
    _check_pcc(f"last_block/{group_size}", expected, got, 0.999)


# 16 rows of 32 context patches and 2 decoder tokens: 34 tokens as 32 + 2 split streams instead of 64.
SPLIT_ROWS, SPLIT_L, SPLIT_H = 16, 32, 2


@pytest.mark.parametrize("masked", [False, True], ids=["causal", "masked"])
@pytest.mark.parametrize("group_size", [1, 4], ids=["univariate", "multivariate"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_split_stream_blocks_pcc(mesh_device, group_size, masked):
    """Every block on the split context / decoder streams, and the last block cut to the decoder stream."""
    model = tiny_model()
    tt = _tt_model(mesh_device, model)
    rows, ll, h = SPLIT_ROWS, SPLIT_L, SPLIT_H
    x = _x(model, rows, ll + h)
    key_valid = torch.ones(rows, ll, dtype=torch.bool)
    if masked:
        key_valid[1, :5] = False
        key_valid[9, 3:7] = False
    group_ids = torch.arange(rows // group_size).repeat_interleave(group_size)
    time_mask = build_forecast_time_mask(ll, h, key_valid, x.device)
    prepared = _host_inputs(model, rows, ll=ll, h=h, group_size=group_size, key_valid=key_valid if masked else None)
    cos, sin = prepared.cos.expand(rows, -1, -1), prepared.sin.expand(rows, -1, -1)
    variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
    name = f"split/{group_size}{'_masked' if masked else ''}"
    for i, (kind, layer) in enumerate(zip(model.config.block_schedule, model.transformer.layers)):
        with torch.no_grad():
            expected = layer(x, TwoStreamMask(mask=time_mask, enc_len=ll), variate_mask, cos, sin)
        _check_pcc(f"{name}/block_{kind}", expected, _run_split(tt, tt._layers[i], x, prepared, last=False), 0.999)
    got = _run_split(tt, tt._layers[-1], x, prepared, last=True)
    _check_pcc(f"{name}/last_block", expected[:, ll:], got, 0.999)


def _normalized(q: torch.Tensor, prepared: TimestralHostInputs) -> torch.Tensor:
    """Raw-space quantiles back to the model's asinh space, where bf16 error is not amplified by sinh."""
    return torch.asinh((q - prepared.anchor_loc[..., None]) / prepared.anchor_scale[..., None])


def _check_predict_block(tt, model, context, observed, group_ids, horizon, *, name, pcc_norm, pcc_raw):
    with torch.no_grad():
        expected = model.predict_block(context, observed, group_ids, horizon)
    got = tt.predict_block(context, observed, group_ids, horizon)
    assert (got.diff(dim=-1) >= 0).all(), "quantiles must be sorted"
    prepared = tt.prepare_inputs(context, observed, group_ids, horizon)
    _check_pcc(f"{name}/normalized", _normalized(expected, prepared), _normalized(got, prepared), pcc_norm)
    _check_pcc(f"{name}/raw", expected, got, pcc_raw)
    log_golden(f"tt_{name}/median", got[..., model.config.median_index])


@pytest.mark.parametrize(
    "case", ["univariate", "missing", "multivariate", "split_univariate", "split_missing", "split_multivariate"]
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_predict_block_tiny_pcc(mesh_device, case):
    """``split_*``: shapes that take the split context / decoder streams."""
    from models.experimental.timestral.tt.model import split_rows_per_block

    model = tiny_model()
    tt = _tt_model(mesh_device, model)
    ps = model.config.patch_size
    split = case.startswith("split_")
    horizon = SPLIT_H if split else 3
    if split:
        series_count, variates = (4, 4) if case == "split_multivariate" else (SPLIT_ROWS, 1)
        context = series(series_count, variates, SPLIT_L * ps).reshape(SPLIT_ROWS, -1)
        group_ids = torch.arange(series_count).repeat_interleave(variates)
    elif case == "multivariate":
        context = series(3, 4, 6 * ps).reshape(12, -1)
        group_ids = torch.arange(3).repeat_interleave(4)
    else:
        context = series(4, 1, 8 * ps)[:, 0]
        group_ids = torch.arange(4)
    observed = torch.ones_like(context, dtype=torch.bool)
    if case.endswith("missing"):
        observed[0, 40:75] = False
        observed[2, : 2 * ps + 5] = False
        context = torch.where(observed, context, torch.zeros_like(context))
    rows, ll = context.shape[0], context.shape[1] // ps
    assert (split_rows_per_block(rows, ll, horizon) is not None) == split
    _check_predict_block(
        tt, model, context, observed, group_ids, horizon, name=f"tiny_{case}", pcc_norm=0.99, pcc_raw=0.99
    )


@pytest.mark.parametrize("name", ["Timestral-5M", "Timestral-209M"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_predict_block_pretrained_pcc(mesh_device, name):
    """Real weights: 16 univariate series of 2048 steps, plus 2 series of 3 variates, horizon 64 and 1024.
    Horizon 64 (2 patches) takes the split context / decoder streams, the others the single stream."""
    model = load_checkpoint(name)
    tt = _tt_model(mesh_device, model)
    context = series(16, 1, 2048)[:, 0]
    observed = torch.ones_like(context, dtype=torch.bool)
    _check_predict_block(
        tt, model, context, observed, torch.arange(16), 2, name=f"{name}_uni_h2", pcc_norm=0.99, pcc_raw=0.98
    )
    _check_predict_block(
        tt, model, context, observed, torch.arange(16), 32, name=f"{name}_uni_h32", pcc_norm=0.99, pcc_raw=0.98
    )
    multi = series(2, 3, 1024, seed=1).reshape(6, -1)
    _check_predict_block(
        tt,
        model,
        multi,
        torch.ones_like(multi, dtype=torch.bool),
        torch.arange(2).repeat_interleave(3),
        4,
        name=f"{name}_multi",
        pcc_norm=0.99,
        pcc_raw=0.98,
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_forecaster_pretrained_pcc(mesh_device):
    """End to end: upstream forecaster (symmetric averaging, 2 decode blocks, refine frame) on TT vs CPU."""
    from models.experimental.timestral.tt.model import TtTimestralForecaster

    model = load_checkpoint("Timestral-209M")
    tt = _tt_model(mesh_device, model)
    context = series(4, 1, 1500, seed=2)
    context[1, 0, 200:260] = float("nan")
    expected = TimestralForecaster(model).forecast(context, prediction_length=1100)
    got = TtTimestralForecaster(tt).forecast(context, prediction_length=1100)
    assert got.quantiles.shape == expected.quantiles.shape == (4, 1, 1100, 21)
    _check_pcc("forecaster_209m/median", expected.median, got.median, 0.98)
    _check_pcc("forecaster_209m/quantiles", expected.quantiles, got.quantiles, 0.98)
    log_golden("tt_forecaster_209m/median", got.median)

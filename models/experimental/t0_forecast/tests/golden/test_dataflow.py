# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the host pipeline and the dataflow re-implementation against the pinned reference (fp32).

Runs in the reference venv (it imports the upstream ``t0`` package), from the worktree root:

    env -u VIRTUAL_ENV HF_HUB_OFFLINE=1 PYTHONPATH=$PWD /home/andy/bringup/venvs/t0-gamma/bin/python -m pytest \
        -p no:cacheprovider -c /dev/null models/experimental/t0_forecast/tests/golden/test_dataflow.py

Tolerances: the host scaler at rtol 1e-5 per element; network tensors at 1e-5 relative (Frobenius norm of the
difference over the norm of the reference), on real (not all-PAD) tokens.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.experimental.t0_forecast.common import GOLDEN_DIR, REVISION
from models.experimental.t0_forecast.tests.golden.dataflow import Dataflow
from models.experimental.t0_forecast.tt import host
from models.experimental.t0_forecast.tt.weights import T0Weights

t0 = pytest.importorskip("t0")
REL = 1e-5


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.double(), ref.double()
    return float((got - ref).norm() / ref.norm().clamp(min=1e-30))


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(8)
    return t0.T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma", revision=REVISION).eval()


@pytest.fixture(scope="module")
def weights():
    return T0Weights.load()


@pytest.fixture(scope="module")
def flow(weights):
    return Dataflow(weights, readout_cut=False)


def golden(name: str):
    path = GOLDEN_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} missing; run make_goldens.py")
    return torch.load(path, weights_only=False)


def reference_window(model, context, horizon):
    """The reference's single-pass window for array input (rollout buffer, sliced)."""
    from t0.rollout import prepare_rollout_buffer

    series = t0.TimeSeries.from_array(torch.as_tensor(context, dtype=torch.float32))
    ctx = series.seq_len
    buffer = prepare_rollout_buffer(series, horizon, ctx, model.patch_size)
    width = host.round_up(ctx, 32) + host.round_up(horizon, 32)
    return buffer.time_slice(0, width)


def test_weights_match_state_dict(model, weights):
    sd = model.state_dict()
    assert torch.equal(
        weights.layers[5].attention.wqkv, sd["transformer.layers.5.attention_block.attention.wQKV.weight"]
    )
    assert torch.equal(weights.decoder.residual_w, sd["decoder.residual_layer.weight"])
    assert [layer.kind for layer in weights.layers] == [
        layer.attention_type.value for layer in model.transformer.layers
    ]
    assert weights.config.quantile_levels == tuple(model.config.quantile_levels)


@pytest.mark.parametrize("case", ["single", "e2e_trimmed", "e2e_raw"])
def test_scaler_and_buffer(model, weights, case):
    """Rollout buffer, causal scaler (loc, scale, asinh) against the reference at rtol 1e-5."""
    cfg = weights.config
    if case == "single":
        g = golden("single.pt")
        context, horizon = g["context"], g["horizon"]
    else:
        g = golden("e2e.pt")
        context, horizon = g["context"], g["horizon"]
    ref_window = reference_window(model, context, horizon)
    series, _, n, ctx = host.series_from_array(context)
    if case == "e2e_trimmed":
        from t0.forecaster import _trim_leading_missing

        ref_window = None
        ref_series = _trim_leading_missing(t0.TimeSeries.from_array(torch.as_tensor(context)), ctx)
        from t0.rollout import prepare_rollout_buffer

        ref_window = prepare_rollout_buffer(ref_series, horizon, ctx, 32).time_slice(0, ctx + horizon)
        series = host.trim_leading(series, ctx)
    ours = host.prepare_rollout_buffer(series, horizon, ctx, 32)
    assert torch.equal(ours.mask, ref_window.mask.to(torch.int64))
    assert torch.equal(ours.group_ids, ref_window.group_ids)
    assert torch.equal(ours.variate_type, ref_window.variate_type)
    assert torch.equal(ours.values, ref_window.variates)
    scaled, loc, scale = host.scale_input(ours, cfg)
    ref_scaled, ref_ls = model.scaler.scale_input(ref_window)
    torch.testing.assert_close(loc, ref_ls.loc, rtol=1e-5, atol=0)
    torch.testing.assert_close(scale, ref_ls.scale, rtol=1e-5, atol=0)
    torch.testing.assert_close(scaled, ref_scaled.variates, rtol=1e-5, atol=1e-7)
    if case == "single":
        torch.testing.assert_close(scaled, g["scaled"], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(loc, g["loc"], rtol=1e-5, atol=0)


def test_masks_match_reference(model):
    """The dense mask port and its structured form (PAD patches from trimming) against ``t0.mask``."""
    from t0.forecaster import _trim_leading_missing
    from t0.mask import build_attention_masks
    from t0.rollout import prepare_rollout_buffer

    g = golden("e2e.pt")
    series = _trim_leading_missing(t0.TimeSeries.from_array(g["context"]), 2048)
    window = prepare_rollout_buffer(series, 64, 2048, 32).time_slice(0, 2112)
    p = lambda t: t.reshape(t.shape[0], -1, 32).to(torch.int64)
    ref_time, ref_group = build_attention_masks(p(window.group_ids), p(window.variate_type), p(window.mask))
    time, group = host.build_attention_masks(p(window.group_ids), p(window.variate_type), p(window.mask))
    assert torch.equal(time, ref_time[:, 0]) and torch.equal(group, ref_group[:, 0])
    time_mask, group_mask, real, gs = host.structured_masks(p(window.group_ids), p(window.variate_type), p(window.mask))
    assert gs == 1 and group_mask is None and time_mask is not None and not bool(real.all())
    assert torch.equal(time_mask[real], time[real])


@pytest.mark.parametrize("seq", [66, 288])
def test_rope_tables(model, weights, seq):
    """xPos-scaled interleaved RoPE tables against ``rotate_queries_and_keys`` (centre floor((S-1)/2))."""
    g = torch.Generator().manual_seed(seq)
    q = torch.randn(2, 8, seq, 128, generator=g)
    k = torch.randn(2, 8, seq, 128, generator=g)
    ref_q, ref_k = model.transformer.rotary_emb.rotate_queries_and_keys(q, k, seq_dim=-2)
    cos_q, sin_q, cos_k, sin_k = host.rope_tables(seq, weights.config)
    rot = host.pair_rotation(128)
    assert rel_err(q * cos_q + (q @ rot) * sin_q, ref_q) < REL
    assert rel_err(k * cos_k + (k @ rot) * sin_k, ref_k) < REL


@pytest.mark.parametrize("name", ["single.pt", "single_long.pt"])
def test_dataflow_full_trace(weights, flow, name):
    """Every capture point of the single-series workload (S = 66 and 288) at 1e-5 relative."""
    g = golden(name)
    series, _, n, ctx = host.series_from_array(g["context"])
    prepared = host.prepare_series(weights.config, series, g["horizon"], ctx, range(n))
    trace = {}
    flow(prepared, trace)
    errs = {"patch_encoder": rel_err(trace["patch_encoder"], g["patch_encoder"])}
    errs |= {f"layer_{i}": rel_err(trace[f"layer_{i}"], g[f"layer_{i}"]) for i in range(len(weights.layers))}
    worst = max(errs, key=errs.get)
    print(f"\n[dataflow] {name}: worst {worst} {errs[worst]:.2e}; layer_23 {errs['layer_23']:.2e}")
    assert errs[worst] < REL, errs
    lo, cnt = prepared.readout_start, prepared.readout_count
    head = Dataflow(weights)(prepared)
    assert rel_err(head, g["head"][:, lo : lo + cnt]) < REL
    data = host.rescale(weights.config, head, prepared)
    assert rel_err(data, g["raw21"]) < REL


def test_readout_cut_is_exact(weights, flow):
    """Cutting to the read-out patches after the last time attention changes nothing (fp32)."""
    g = golden("single.pt")
    series, _, n, ctx = host.series_from_array(g["context"])
    prepared = host.prepare_series(weights.config, series, 64, ctx, range(n))
    torch.testing.assert_close(Dataflow(weights)(prepared), flow(prepared), rtol=1e-6, atol=1e-6)


def _network(weights):
    return Dataflow(weights)


@pytest.mark.parametrize(
    "case",
    ["single_raw21", "single_levels9", "single_gift9", "e2e_raw21", "e2e_gift9", "e2e_gift21"],
)
def test_predict_against_goldens(weights, case):
    """``host.predict`` around the dataflow against the reference ``predict`` outputs (final quantiles)."""
    cfg = weights.config
    name, key = case.split("_", 1)
    g = golden(f"{name}.pt")
    levels = {
        "raw21": cfg.quantile_levels,
        "levels9": host_levels9(),
        "gift9": host_levels9(),
        "gift21": cfg.quantile_levels,
    }[key]
    gift = key.startswith("gift")
    got = host.predict(
        cfg,
        _network(weights),
        g["context"],
        g["horizon"],
        levels,
        trim_leading_missing=gift,
        symmetric_averaging=gift,
        clip_nonnegative=gift,
        snap_integers=gift,
    )
    ref = g[key]
    err = rel_err(got, ref)
    print(f"\n[predict] {case}: rel {err:.2e}")
    assert err < REL
    if name == "e2e" and key == "raw21":
        series, _, n, ctx = host.series_from_array(g["context"])
        prepared = host.prepare_series(cfg, series, g["horizon"], ctx, range(n))
        assert rel_err(Dataflow(weights)(prepared), g["head_readout"]) < REL


def host_levels9():
    return (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def test_predict_multivariate_and_tails(model, weights):
    """4 series x 8 variates (real group attention, G = 8) and tail levels (0.005, 0.995) against the live reference."""
    g = torch.Generator().manual_seed(7)
    t = torch.arange(2048, dtype=torch.float32)
    x = (5 + torch.sin(2 * np.pi * t / 24) + 0.3 * torch.randn(4, 8, 2048, generator=g)).to(torch.float32)
    levels = (0.005, 0.1, 0.5, 0.9, 0.995)
    ref = model.predict(x, horizon=64, quantile_levels=list(levels)).quantiles
    got = host.predict(weights.config, _network(weights), x, 64, levels)
    assert got.shape == ref.shape == (4, 8, 64, 5)
    assert rel_err(got, ref) < REL


@pytest.mark.parametrize(
    "name",
    ["time_layer_clean", "time_layer_padded", "group_layer_g1", "group_layer_g4", "group_layer_g4_padded"],
)
def test_layer_modules(weights, flow, name):
    """The dataflow's layers on the random-input module goldens (real tokens only)."""
    m = golden("modules.pt")[name]
    time_mask, group_mask, real, gs = host.structured_masks(m["group_ids"], m["variate_type"], m["mask"])
    s = m["x"].shape[1]
    prepared = host.T0HostInputs(
        rows=m["x"].shape[0],
        num_patches=s,
        context_patches=s,
        readout_start=0,
        readout_count=s,
        group_size=gs,
        tokens=torch.zeros(0),
        patch_type=torch.zeros(0),
        time_mask=time_mask,
        group_mask=group_mask,
        real_tokens=real,
        rope=host.rope_tables(s, weights.config),
        loc=torch.zeros(0),
        scale=torch.zeros(0),
        horizon=0,
        target_rows=(),
    )
    layer = weights.layers[0 if name.startswith("time") else 2]
    x = m["x"]
    if name.startswith("time"):
        attn = x + flow.time_attention(layer.attention, x, prepared)
    else:
        attn = x + flow.group_attention(layer.attention, x, prepared, group_mask)
    out = attn + flow.mlp(layer.mlp, attn)
    assert rel_err(attn[real], m["attention"][real]) < REL
    assert rel_err(out[real], m["out"][real]) < REL


def test_patch_encoder_mlp_head_modules(weights, flow):
    m = golden("modules.pt")
    pe = m["patch_encoder"]
    t_index = (torch.arange(32, dtype=torch.float32) / 32).expand(*pe["values"].shape)
    tokens = torch.cat([pe["values"], t_index, (pe["mask"] == host.VALID).float()], dim=-1)
    ptype = host.reduce_patch_metadata(pe["variate_type"], pe["mask"]).clamp(min=0)
    from models.experimental.t0_forecast.tests.golden.dataflow import residual_block

    got = residual_block(weights.patch_encoder, tokens) + weights.type_embeddings[ptype]
    assert rel_err(got, pe["out"]) < REL
    assert rel_err(flow.mlp(weights.layers[0].mlp, m["mlp"]["x"]), m["mlp"]["out"]) < REL
    assert rel_err(flow.head(m["head"]["x"]), m["head"]["out"]) < REL


def test_predict_paths(weights):
    """Branch coverage golden: context 2000 / horizon 40, a partially PAD group (explicit mask), trimmed series."""
    cfg = weights.config
    g = golden("paths.pt")
    net = _network(weights)
    got = host.predict(cfg, net, g["uni_context"], g["uni_horizon"], cfg.quantile_levels)
    assert got.shape == g["uni_raw21"].shape and rel_err(got, g["uni_raw21"]) < REL
    series, _, n, ctx = host.series_from_array(g["multi_context"], g["multi_mask"])
    prepared = host.prepare_series(cfg, series, 64, ctx, range(n))
    assert prepared.group_size == 3 and prepared.group_mask is not None and prepared.time_mask is not None
    got = host.predict(cfg, net, g["multi_context"], 64, cfg.quantile_levels, mask=g["multi_mask"])
    assert rel_err(got, g["multi_raw21"]) < REL
    assert rel_err(net(prepared), g["multi_head_readout"]) < REL
    settings = dict(trim_leading_missing=True, symmetric_averaging=True, clip_nonnegative=True, snap_integers=True)
    got = host.predict(cfg, net, g["multi_context"], 64, cfg.quantile_levels, **settings)
    assert rel_err(got, g["multi_gift21"]) < REL

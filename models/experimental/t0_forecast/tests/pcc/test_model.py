# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC of TTNN t0-gamma on real weights against the fp32 reference goldens.

Metrics on the read-out patches ``[rows, n, 32, 21]`` in asinh space, against a hook on ``m.head`` (before
``sinh(q) * scale + loc``), plus the data-space quantiles (``tests/helpers.quantile_metrics``):
    pcc_normalized   PCC over everything; dominated by each row's level
    pcc_spread       PCC of ``h - mean_q(h)``, the quantile spread around each step's centre: catches a collapse
                     of the quantiles that pcc_normalized cannot see on trending series
    pcc_row_p01 / pcc_row_min   per-row PCC of ``h`` (1st percentile / minimum over rows)
    pcc_final        data-space quantiles after the host post-processing (raw21: the 21 trained levels);
                     pcc_final_spread the same minus their mean over the quantile axis
Each case also prints the collapsed output's metrics (every quantile = its mean over the quantile axis), which
every gate floor must reject.

Cases and the paths they take:
    single          1 x 2048 -> 64               clean causal time attention, G = 1, S = 66 -> 96
    single_long     1 x 8192 -> 1024             S = 288, 32 read-out patches (the far horizon)
    paper16         16 x 2048 -> 64              the first 16 series of the paper golden (realistic mix)
    multivariate    4 series x 8 variates        real group attention (G = 8, no mask)
    long2           2 x 8192 -> 1024             S = 288 (tile aligned, xPos centre 143), 32 read-out patches
    e2e             64 x 2048 -> 64              leading NaN -> PAD patches (explicit time mask), gaps
                                                 (MISSING), count rows; GIFT-Eval settings (x and -x)
    paper16_performance                          bfp8 weights, LoFi
"""

from __future__ import annotations

import pytest

from models.experimental.t0_forecast.tests.helpers import (
    FLOORS,
    below,
    collapsed,
    final_metrics,
    golden,
    misses,
    pcc,
    quantile_metrics,
    weights,
)
from models.experimental.t0_forecast.tt.common import TtT0Precision
from models.experimental.t0_forecast.tt.model import TtT0

GIFT_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_MODELS: dict = {}


def _tt(device, precision: str) -> TtT0:
    # One model per precision per device for the whole module (weights upload once).
    key = (id(device), precision)
    if key not in _MODELS:
        _MODELS.clear()
        _MODELS[key] = TtT0(device, weights(), TtT0Precision.named(precision))
    return _MODELS[key]


def _forward(tt: TtT0, context, horizon, **kwargs):
    """One forward: ``(prepared, normalized [R, n, 32, 21], raw21 data space)``."""
    import ttnn

    prepared = tt.prepare_inputs(context, horizon, **kwargs)
    inputs = tt.upload_inputs(prepared)
    out = tt.forward_device(inputs)
    normalized = tt.normalized(out, prepared)
    final = tt.postprocess(out, prepared)
    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)
    return prepared, normalized, final


def check(case: str, floors: dict, head_ref, normalized, raw_ref=None, final=None, strict: bool = True) -> dict:
    """Score, print (with the collapsed output's scores) and assert ``floors``; returns the metrics. ``strict``: every
    asinh-space gate metric alone must reject the collapsed output (the realistic bench inputs); otherwise the floors
    together must (smooth coverage inputs, where pcc_normalized alone cannot see a collapse). ``pcc_final`` is never
    required to reject it alone (a collapsed forecast scores near 1 in data space; ``pcc_final_spread`` does)."""
    got = quantile_metrics(head_ref, normalized)
    bad = quantile_metrics(head_ref, collapsed(head_ref))
    if raw_ref is not None:
        got.update(final_metrics(raw_ref, final))
        bad.update(final_metrics(raw_ref, collapsed(raw_ref)))
    show = lambda m: ", ".join(f"{k} {v:.6f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items())
    print(f"\n[pcc] {case}: {show(got)}\n[pcc] {case} collapsed: {show(bad)}")
    missed = misses(got, floors)
    assert not missed, f"{case}: {'; '.join(missed)}"
    for key, floor in floors.items():
        if strict and key in bad and key.startswith("pcc_") and key != "pcc_final":
            assert below(bad[key], floor), f"{case}: the collapsed output passes {key} ({bad[key]:.6f})"
    if floors:
        assert misses(bad, floors), f"{case}: the collapsed output passes the gate"
    return got


CASES = {
    # name: (golden file, rows / groups to take, precision, floors, read-out slice of the full head capture)
    "single": ("single.pt", None, "accuracy", "single", (63, 2)),
    "single_long": ("single_long.pt", None, "accuracy", "single_long", (255, 32)),
    "paper16": ("bench_paper.pt", 16, "accuracy", "accuracy", None),
    "multivariate": ("bench_multivariate.pt", 4, "accuracy", "accuracy", None),
    "long2": ("bench_long_context.pt", 2, "accuracy", "accuracy", None),
    "paper16_performance": ("bench_paper.pt", 16, "performance", "performance", None),
    # D4 floors.
    "single_performance": ("single.pt", None, "performance", "single_performance", (63, 2)),
    "single_long_performance": ("single_long.pt", None, "performance", "single_long_performance", (255, 32)),
}


@pytest.fixture(scope="module")
def device(request):
    """One device for the module, so the weights upload once per precision."""
    import ttnn

    dev = ttnn.open_device(device_id=0, l1_small_size=0)
    yield dev
    _MODELS.clear()
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", list(CASES))
def test_forward(device, case):
    name, take, precision, floors, readout = CASES[case]
    g = golden(name)
    context = g["context"] if take is None else g["context"][:take]
    rows = context.shape[0] * (context.shape[1] if context.ndim == 3 else 1)
    head_ref = g["head"][:, readout[0] : sum(readout)] if readout else g["head_readout"][:rows]
    raw_ref = g["raw21"] if take is None else g["raw21"][:take]
    tt = _tt(device, precision)
    prepared, normalized, final = _forward(tt, context, g["horizon"])
    assert normalized.shape == head_ref.shape and final.shape == raw_ref.shape
    print(f"\n[case] {case}: rows {rows}, S {prepared.num_patches}, G {prepared.group_size}")
    check(case, FLOORS[floors], head_ref, normalized, raw_ref, final)


def test_e2e_masked_paths(device):
    """PAD patches from trimming (explicit time mask), MISSING gaps, counts: plain raw21 and its head."""
    g = golden("e2e.pt")
    tt = _tt(device, "accuracy")
    prepared, normalized, final = _forward(tt, g["context"], g["horizon"])
    assert prepared.time_mask is None  # untrimmed NaN are MISSING, not PAD: the causal path
    check("e2e", FLOORS["coverage"], g["head_readout"], normalized, g["raw21"], final, strict=False)


def test_rope_cache_eviction(device):
    """Device inputs keep their RoPE tables after the per-shape cache evicts them (more shapes than entries)."""
    from models.experimental.t0_forecast.tt.model import _MAX_CONSTANTS

    g = golden("single.pt")
    tt = _tt(device, "accuracy")
    prepared = tt.prepare_inputs(g["context"], 64)
    inputs = tt.upload_inputs(prepared)
    shapes = range(10, 10 + _MAX_CONSTANTS // 4 + 2)  # 4 tables per shape: crosses the cache size
    for k in shapes:
        other = tt.upload_inputs(tt.prepare_inputs(g["context"][:, : 32 * k], 64))
        tt.deallocate_inputs(other)
    assert len(tt._constants) <= _MAX_CONSTANTS
    assert not any(key[1] == prepared.num_patches for key in tt._constants), "the first shape must be evicted"
    out = tt.forward_device(inputs)
    normalized = tt.normalized(out, prepared)
    tt.deallocate_inputs(inputs)
    check("rope_cache_eviction", FLOORS["single"], g["head"][:, 63:65], normalized, strict=False)


@pytest.mark.parametrize(
    "precision,levels",
    [("accuracy", "gift9"), ("accuracy", "gift21"), ("performance", "gift9"), ("performance", "gift21")],
)
def test_e2e_gift_settings(device, precision, levels):
    """GIFT-Eval settings: trim (PAD patches -> masked time attention), symmetric averaging (a second forward on
    the negated series), clip and snap, at the 9 GIFT levels and the 21 trained levels. Scored (D4) on both forwards'
    head read-outs against ``gift_head_readouts`` and on the final quantiles (``pcc_final``, ``pcc_final_spread``)
    with the coverage floors; the collapsed output must fail them."""
    import torch

    from models.experimental.t0_forecast.tt import host

    g = golden("e2e.pt")
    tt = _tt(device, precision)
    trimmed = host.trim_leading(host.series_from_array(g["context"])[0], 2048)
    prepared = host.prepare_series(tt.config, trimmed, 64, 2048, range(64))
    assert prepared.time_mask is not None and not bool(prepared.real_tokens.all())
    q = GIFT_LEVELS if levels == "gift9" else tt.config.quantile_levels
    heads = []

    def network(p):
        out = tt.network(p)
        heads.append(out)
        return out

    settings = dict(trim_leading_missing=True, symmetric_averaging=True, clip_nonnegative=True, snap_integers=True)
    got = host.predict(tt.config, network, g["context"], g["horizon"], q, **settings)
    ref_heads = g["gift_head_readouts"]
    assert len(heads) == len(ref_heads) and all(h.shape == r.shape for h, r in zip(heads, ref_heads))
    floors = FLOORS["coverage" if precision == "accuracy" else "coverage_performance"]
    check(f"e2e_{levels}_{precision}", floors, torch.cat(ref_heads), torch.cat(heads), g[levels], got, strict=False)


def test_branch_paths(device):
    """Context 2000 (a partial PAD patch), horizon 40 (two patches, 40 steps kept); 4 series x 3 variates with an
    explicit PAD row (group mask: real and all-PAD rows in one series) and trimmed series (time mask)."""
    g = golden("paths.pt")
    tt = _tt(device, "accuracy")
    q = tt.config.quantile_levels
    _, normalized, final = _forward(tt, g["uni_context"], g["uni_horizon"])
    assert final.shape == g["uni_raw21"].shape
    check("paths_uni_h40", FLOORS["coverage"], g["uni_head_readout"], normalized, g["uni_raw21"], final, strict=False)
    prepared = tt.prepare_inputs(g["multi_context"], 64, mask=g["multi_mask"])
    assert prepared.group_size == 3 and prepared.group_mask is not None and prepared.time_mask is not None
    normalized = tt.network(prepared)
    final = tt.predict(g["multi_context"], 64, q, mask=g["multi_mask"])
    check(
        "paths_multi_masked",
        FLOORS["coverage"],
        g["multi_head_readout"],
        normalized,
        g["multi_raw21"],
        final,
        strict=False,
    )
    settings = dict(trim_leading_missing=True, symmetric_averaging=True, clip_nonnegative=True, snap_integers=True)
    p_gift = pcc(g["multi_gift21"], tt.predict(g["multi_context"], 64, q, **settings))
    print(f"\n[pcc] paths gift21 (GIFT-Eval settings): pcc_final {p_gift:.6f}")
    assert p_gift >= 0.99

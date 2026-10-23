# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC of TTNN t0-gamma on real weights against the fp32 reference goldens.

    pcc_normalized   the monotone head's quantiles in asinh space at the read-out patches ([rows, n, 32, 21]),
                     against a hook on ``m.head`` (before ``sinh(q) * scale + loc``)
    pcc_final        data-space quantiles after the host post-processing (raw21: the 21 trained levels)

Cases and the paths they take:
    single          1 x 2048 -> 64               clean causal time attention, G = 1, S = 66 -> 96
    paper16         16 x 2048 -> 64              the first 16 series of the paper golden
    multivariate    4 series x 8 variates        real group attention (G = 8, no mask)
    long2           2 x 8192 -> 1024             S = 288 (tile aligned, xPos centre 143), 32 read-out patches
    e2e             64 x 2048 -> 64              leading NaN -> PAD patches (explicit time mask), gaps
                                                 (MISSING), count rows; GIFT-Eval settings (x and -x)
    paper16_performance                          bfp8 weights, LoFi
"""

from __future__ import annotations

import pytest

from models.experimental.t0_forecast.tests.helpers import golden, pcc, weights
from models.experimental.t0_forecast.tt.common import TtT0Precision
from models.experimental.t0_forecast.tt.model import TtT0

GIFT_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FLOORS = {"accuracy": (0.999, 0.99), "performance": (0.995, 0.98)}  # (pcc_normalized, pcc_final)

_MODELS: dict = {}


def _tt(device, precision: str) -> TtT0:
    # One model per precision per device for the whole module (weights upload once).
    key = (id(device), precision)
    if key not in _MODELS:
        _MODELS.clear()
        _MODELS[key] = TtT0(device, weights(), TtT0Precision.named(precision))
    return _MODELS[key]


def _forward(tt: TtT0, context, horizon):
    """One forward: ``(normalized [R, n, 32, 21], raw21 data space)``."""
    import ttnn

    prepared = tt.prepare_inputs(context, horizon)
    inputs = tt.upload_inputs(prepared)
    out = tt.forward_device(inputs)
    normalized = tt.normalized(out, prepared)
    final = tt.postprocess(out, prepared)
    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)
    return prepared, normalized, final


CASES = {
    # name: (golden file, rows / groups to take, precision)
    "single": ("single.pt", None, "accuracy"),
    "paper16": ("bench_paper.pt", 16, "accuracy"),
    "multivariate": ("bench_multivariate.pt", 4, "accuracy"),
    "long2": ("bench_long_context.pt", 2, "accuracy"),
    "paper16_performance": ("bench_paper.pt", 16, "performance"),
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
    name, take, precision = CASES[case]
    g = golden(name)
    context = g["context"] if take is None else g["context"][:take]
    rows = context.shape[0] * (context.shape[1] if context.ndim == 3 else 1)
    head_ref = g["head"][:, 63:65] if name == "single.pt" else g["head_readout"][:rows]
    raw_ref = g["raw21"] if take is None else g["raw21"][:take]
    tt = _tt(device, precision)
    prepared, normalized, final = _forward(tt, context, g["horizon"])
    assert normalized.shape == head_ref.shape and final.shape == raw_ref.shape
    p_norm, p_final = pcc(head_ref, normalized), pcc(raw_ref, final)
    floor_norm, floor_final = FLOORS[precision]
    print(
        f"\n[pcc] {case}: rows {rows}, S {prepared.num_patches}, G {prepared.group_size}, "
        f"time mask {prepared.time_mask is not None}: pcc_normalized {p_norm:.6f}, pcc_final {p_final:.6f}"
    )
    assert p_norm >= floor_norm and p_final >= floor_final


def test_e2e_masked_paths(device):
    """PAD patches from trimming (explicit time mask), MISSING gaps, counts: plain raw21 and its head."""
    g = golden("e2e.pt")
    tt = _tt(device, "accuracy")
    series_prepared = tt.prepare_inputs(g["context"], g["horizon"])
    assert series_prepared.time_mask is None  # untrimmed NaN are MISSING, not PAD
    _, normalized, final = _forward(tt, g["context"], g["horizon"])
    p_norm, p_final = pcc(g["head_readout"], normalized), pcc(g["raw21"], final)
    print(f"\n[pcc] e2e plain: pcc_normalized {p_norm:.6f}, pcc_final {p_final:.6f}")
    assert p_norm >= 0.999 and p_final >= 0.99


@pytest.mark.parametrize("levels", ["gift9", "gift21"])
def test_e2e_gift_settings(device, levels):
    """GIFT-Eval settings: trim (PAD patches -> masked time attention), symmetric averaging (a second forward on
    the negated series), clip and snap, at the 9 GIFT levels and the 21 trained levels."""
    from models.experimental.t0_forecast.tt import host

    g = golden("e2e.pt")
    tt = _tt(device, "accuracy")
    trimmed = host.trim_leading(host.series_from_array(g["context"])[0], 2048)
    prepared = host.prepare_series(tt.config, trimmed, 64, 2048, range(64))
    assert prepared.time_mask is not None and not bool(prepared.real_tokens.all())
    q = GIFT_LEVELS if levels == "gift9" else tt.config.quantile_levels
    got = tt.predict(
        g["context"],
        g["horizon"],
        q,
        trim_leading_missing=True,
        symmetric_averaging=True,
        clip_nonnegative=True,
        snap_integers=True,
    )
    value = pcc(g[levels], got)
    print(f"\n[pcc] e2e {levels} (GIFT-Eval settings): pcc_final {value:.6f}")
    assert value >= 0.99


def test_branch_paths(device):
    """Context 2000 (a partial PAD patch), horizon 40 (two patches, 40 steps kept); 4 series x 3 variates with an
    explicit PAD row (group mask: real and all-PAD rows in one series) and trimmed series (time mask)."""
    g = golden("paths.pt")
    tt = _tt(device, "accuracy")
    q = tt.config.quantile_levels
    _, normalized, final = _forward(tt, g["uni_context"], g["uni_horizon"])
    p_uni = pcc(g["uni_head_readout"], normalized), pcc(g["uni_raw21"], final)
    assert final.shape == g["uni_raw21"].shape
    prepared = tt.prepare_inputs(g["multi_context"], 64, mask=g["multi_mask"])
    assert prepared.group_size == 3 and prepared.group_mask is not None and prepared.time_mask is not None
    normalized = tt.network(prepared)
    final = tt.predict(g["multi_context"], 64, q, mask=g["multi_mask"])
    p_multi = pcc(g["multi_head_readout"], normalized), pcc(g["multi_raw21"], final)
    settings = dict(trim_leading_missing=True, symmetric_averaging=True, clip_nonnegative=True, snap_integers=True)
    p_gift = pcc(g["multi_gift21"], tt.predict(g["multi_context"], 64, q, **settings))
    print(
        f"\n[pcc] paths: uni h40 {p_uni[0]:.6f} / {p_uni[1]:.6f}, multi masked {p_multi[0]:.6f} / {p_multi[1]:.6f}, gift {p_gift:.6f}"
    )
    assert min(p_uni[0], p_multi[0]) >= 0.999 and min(p_uni[1], p_multi[1], p_gift) >= 0.99

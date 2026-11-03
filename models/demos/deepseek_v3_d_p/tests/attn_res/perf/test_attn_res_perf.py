# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 AttnRes performance: a boundary layer and an ordinary layer, end to end, traced and untraced.

Galaxy (SP8xTP4) runs K3's prefill chunk and is gated. LoudBox (SP2xTP4, 1280 tokens) runs the same walk at Galaxy's
TP width and 640 rows per chip, so every per-chip shape matches Galaxy's and only the SP axis is shorter; it reports
without gating until it has CI references.

There is no QuietBox arm: QuietBox 2 has no layout AttnRes can run at a Galaxy-like shape.
  * TP2: at a 3584-wide shard `attn_res_gather_softmax` needs 1658880 B of L1 per core against 1572864 B.
  * TP4 on 2D torus-X: the four chips are wired 2x2, so a 1x4 TORUS_X request comes up unwrapped.
  * TP4 on a 1D ring: `attn_res_gather_softmax` requires a 2D fabric.

The walk starts where the last pipeline rank starts (24/24/24/21 over 4 Galaxies), with the sealed snapshots the
upstream ranks hand over. Two layer kinds cover what AttnRes costs across the model:

  * **Boundary layer** (the rank's first): the most expensive AttnRes layer of the stack, and every AttnRes
    program it has — the 1-site `inter_block` over the inherited set for the read that straddles the rank
    boundary, the pre-attention read, the seal (the largest concat of any full block), the 24-site
    `inter_block` at the largest sealed count of any full block, past the statistics fold's threshold, and the
    post-attention read. The last block has one more snapshot but only 9 layers, so it batches 19 sites, not 24.
  * **Ordinary layer** (the ones after it): two reads, each settling the module write the walk deferred into
    it. 85 of the 93 layers are this, and it is about 60% of AttnRes time over the model, which the boundary
    layer alone weights at under 10%.

AttnRes runs through the production code path: `TtAttnResResidual` over `TtAttnResWalk`, built the way
`TtKimiK3Transformer`'s residual factory builds it on a non-first rank, and driven in `TtKimiK3Block.forward`'s
call order. The attention and FFN are stand-ins: a softmax, which AttnRes never issues, so their programs are
dropped from the device time and mark where each residual call starts and ends.

The gates are end-to-end time on the host clock, with no profiler attached. The host clock sees whole forwards,
so each layer is a difference between forwards timed interleaved: the boundary layer is a 1-layer forward minus
the same forward with the AttnRes calls taken out, and the ordinary layer is what the next `ORDINARY_LAYERS`
layers add over it, divided by their count; one ordinary layer alone is too small a difference of two ~5 ms
eager forwards to time reliably. Traced replays captured
forwards; untraced dispatches them eagerly, the way the K3 runtime runs prefill. Device time per layer is
reported beside it: the sum over programs of the slowest chip's realtime-profiler window, with the per-op split.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections import defaultdict

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import (
    EXACT_BOX,
    HIDDEN_SIZE,
    L1_SMALL_SIZE,
    PER_CHIP_TOKENS,
    generator,
    place,
    random_hidden,
    random_queries,
)
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_xy_device_params,
    tp_axis_is_wrapped,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import BLOCK_SIZE, TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(1800)]

# The first measured layer: the last rank's first, under the 24/24/24/21 split of K3's 93 layers.
LAYER = 72
RANK_LAYERS = 93 - LAYER
INHERITED_SNAPSHOTS = LAYER // BLOCK_SIZE
# The rank's first layer, then this many ordinary layers after it, averaged into one ordinary-layer number.
ORDINARY_LAYERS = 8
LAYERS = ("boundary", "ordinary")
_SAMPLES = 5
_WALL_SAMPLES = 30
# Set to report without gating, e.g. while re-baselining.
_REPORT_ONLY = os.environ.get("ATTN_RES_PERF_REPORT_ONLY") == "1"
_LOUDBOX_PERF_SKU = "bh_loudbox"
_GALAXY = "SP8xTP4"
# Galaxy end-to-end time (us) of each layer's AttnRes on the CI pool (bh_sc1_high_power). Untraced is host-dispatch
# bound, with the device idle between AttnRes programs, so it follows the host CPU and gets the wider band. Galaxies
# off the pool can sit outside the band; run them with ATTN_RES_PERF_REPORT_ONLY=1.
_GALAXY_REFERENCE_E2E_US = {
    "traced": {"boundary": 1700.0, "ordinary": 140.0},
    "untraced": {"boundary": 4140.0, "ordinary": 284.0},
}
# LoudBox references, per SKU: a run opts in with ATTN_RES_PERF_SKU, so a box of another SKU never
# gates against them. To gate a layout, add its CI numbers here and drop it from _REPORT_ONLY_LAYOUTS.
_PERF_REFERENCE_E2E_US = {
    _LOUDBOX_PERF_SKU: {},
}
# LoudBox SP2xTP4 has no CI references yet: it reports but is not gated.
_REPORT_ONLY_LAYOUTS = ("SP2xTP4",)
_MARGIN = {"traced": 0.05, "untraced": 0.10}
# The stand-in module: never issued by AttnRes, so its programs are identifiable by kernel path.
_MODULE_KERNEL = "normalization/softmax"


def _perf_reference_e2e_us(layout: str, mode: str) -> dict[str, float]:
    """Return each layer's end-to-end reference (us) for a gated layout, on the ATTN_RES_PERF_SKU it opts in to."""
    if layout == _GALAXY:
        return _GALAXY_REFERENCE_E2E_US[mode]
    skus = sorted(sku for sku, references in _PERF_REFERENCE_E2E_US.items() if layout in references)
    if not skus:
        raise ValueError(f"{layout} has no end-to-end reference")
    sku = os.environ.get("ATTN_RES_PERF_SKU")
    if sku not in skus:
        raise ValueError(f"set ATTN_RES_PERF_SKU to one of {skus} to opt in to this hardware-specific performance gate")
    return _PERF_REFERENCE_E2E_US[sku][layout][mode]


def _assert_within_margin(what: str, measured_us: float, reference_us: float, margin: float) -> None:
    lower = reference_us * (1.0 - margin)
    upper = reference_us * (1.0 + margin)
    assert lower <= measured_us <= upper, (
        f"{what} {measured_us:.1f} us is outside performance range "
        f"[{lower:.1f}, {upper:.1f}] us "
        f"(reference {reference_us:.1f} us ± {margin:.0%})"
    )


@pytest.mark.parametrize("mode", ["traced", "untraced"])
def test_attn_res_performance_uses_two_sided_margin(mode, monkeypatch, expect_error) -> None:
    # The Galaxy gate needs no ATTN_RES_PERF_SKU opt-in.
    monkeypatch.delenv("ATTN_RES_PERF_SKU", raising=False)
    margin = _MARGIN[mode]
    for layer, reference_us in _perf_reference_e2e_us(_GALAXY, mode).items():
        _assert_within_margin(layer, reference_us * (1 - 0.99 * margin), reference_us, margin)
        _assert_within_margin(layer, reference_us * (1 + 0.99 * margin), reference_us, margin)
        with expect_error(AssertionError, "outside performance range"):
            _assert_within_margin(layer, reference_us * (0.99 - margin), reference_us, margin)
        with expect_error(AssertionError, "outside performance range"):
            _assert_within_margin(layer, reference_us * (1.01 + margin), reference_us, margin)


@pytest.mark.parametrize("layout", _REPORT_ONLY_LAYOUTS)
def test_attn_res_report_only_layouts_have_no_reference(layout, expect_error) -> None:
    # A reference added for a report-only layout would never gate: drop the layout from _REPORT_ONLY_LAYOUTS.
    with expect_error(ValueError, "has no end-to-end reference"):
        _perf_reference_e2e_us(layout, "traced")


def _op_name(kernel_sources) -> str:
    names = set()
    for source in kernel_sources:
        match = re.search(r"operations/(.+?)/device/", source.replace("\\", "/"))
        names.add(match.group(1) if match else source.rsplit("/", 1)[-1])
    return " + ".join(sorted(names))


def _profiled(mesh_device, fn) -> list[dict]:
    """The programs `fn` dispatches, in dispatch order, each at its slowest chip's duration."""

    def run():
        fn()
        ttnn.synchronize_device(mesh_device)

    _, per_program = profile_realtime_program_merged(mesh_device, run)
    return [
        {"op": _op_name(entry["kernel_sources"]), "ns": entry["duration_ns"]}
        for _, entry in sorted(per_program.items())
    ]


class _Prepared:
    """A forward ready to time: called eagerly, or captured once and replayed."""

    def __init__(self, mesh_device, fn, traced):
        self.mesh_device, self.trace_id = mesh_device, None
        fn()  # compile
        fn()
        ttnn.synchronize_device(mesh_device)
        if not traced:
            self.call = fn
            return
        # The caller only holds this object once construction returns, so a failure past this point must
        # close the capture and release the trace here, or it would leave the mesh unusable for later params.
        self.trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            try:
                fn()
            finally:
                ttnn.end_trace_capture(mesh_device, self.trace_id, cq_id=0)
            ttnn.execute_trace(mesh_device, self.trace_id, cq_id=0, blocking=True)
        except BaseException:
            self.release()
            raise
        self.call = lambda: ttnn.execute_trace(mesh_device, self.trace_id, cq_id=0, blocking=False)

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)


def _interleaved_walls_us(mesh_device, calls) -> dict[str, list[float]]:
    """Host clock (us) of each call, from an idle device through device sync, interleaved so drift cancels."""
    walls = {name: [] for name in calls}
    for _ in range(_WALL_SAMPLES):
        for name, call in calls.items():
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            call()
            ttnn.synchronize_device(mesh_device)
            walls[name].append((time.perf_counter() - start) * 1e6)
    return walls


def _split(programs, num_layers):
    """One forward's programs -> each layer's AttnRes programs.

    The forward dispatches `[setup] [m] open(0) [m] read(0) [m] open(1) ... read(L-1) [m]`. What precedes
    the first stand-in is test setup. Layer `l` is the two segments after stand-ins `2l` and `2l + 1`. The
    walk ends in `discard`, which only frees, so nothing follows the last stand-in.
    """
    segments, current, started = [], [], False
    for p in programs:
        if _MODULE_KERNEL in p["op"]:
            if started:
                segments.append(current)
            current, started = [], True
            continue
        current.append(p)
    segments.append(current)
    assert len(segments) == 2 * num_layers + 1, f"expected {2 * num_layers + 1} segments, got {len(segments)}"
    trailing = [p["op"] for p in segments[-1]]
    assert not trailing, f"programs after the last stand-in, not attributed to a layer: {trailing}"
    return [segments[2 * l] + segments[2 * l + 1] for l in range(num_layers)]


def _log_ops(name, programs):
    by_op = defaultdict(lambda: [0, 0.0])
    for p in programs:
        by_op[p["op"]][0] += 1
        by_op[p["op"]][1] += p["ns"]
    logger.info(f"{name} layer, per op (median sample):")
    for op, (count, ns) in sorted(by_op.items(), key=lambda kv: -kv[1][1]):
        logger.info(f"  {ns / 1e3:8.1f} us  x{count:<3d} {op}")


_TRACE_REGION_SIZE = 128 * 1024 * 1024


@pytest.mark.parametrize("mode", ["traced", "untraced"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=L1_SMALL_SIZE, trace_region_size=_TRACE_REGION_SIZE, **EXACT_BOX),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
        # Unwrapped 2D, the fabric the AttnRes model suite runs on LoudBox; `attn_res_gather_softmax` needs 2D.
        pytest.param(
            (2, 4),
            fabric2d_device_params(l1_small_size=L1_SMALL_SIZE, trace_region_size=_TRACE_REGION_SIZE, **EXACT_BOX),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="SP2xTP4-fabric-2d",
        ),
        # No QuietBox arm: it has no layout AttnRes can run at a Galaxy-like shape. See the module docstring.
    ],
    indirect=True,
)
def test_attn_res_layer_perf(mesh_device, device_params, mode):
    """Gate the AttnRes end-to-end time of a boundary layer and an ordinary layer on LoudBox and Galaxy,
    and report their device time."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    layout = f"SP{op.sp_factor}xTP{op.tp_factor}"
    report_only = layout in _REPORT_ONLY_LAYOUTS
    if layout == _GALAXY:
        require_high_power(f"kimi_k3 AttnRes {layout} perf")
    # Before measuring, so a gated run without its SKU opt-in fails fast.
    references = None if report_only else _perf_reference_e2e_us(layout, mode)
    require_realtime_profiler("the AttnRes device-time report")
    num_tokens = PER_CHIP_TOKENS * op.sp_factor

    rng = generator(0)
    q_pre = [op.to_query(q) for q in random_queries(rng, RANK_LAYERS)]
    q_post = [op.to_query(q) for q in random_queries(rng, RANK_LAYERS)]
    q_out = op.to_query(random_queries(rng, 1)[0])
    live_base = place(op, random_hidden(rng, num_tokens).unsqueeze(0).unsqueeze(0))
    sealed_base = place(op, torch.randn(1, INHERITED_SNAPSHOTS, num_tokens, HIDDEN_SIZE, generator=rng))
    module = lambda h: ttnn.softmax(h, dim=-1)

    def forward(num_layers):
        """The rank's residual over its first `num_layers` layers, in `TtKimiK3Block.forward`'s call order."""
        # The handoff the upstream rank sent: the live stream and the sealed snapshots, owned by the walk.
        inherited = ttnn.clone(sealed_base)
        hidden = module(live_base)
        residual = TtAttnResResidual(
            TtAttnResWalk(op, hidden, q_pre, q_post, q_out, RANK_LAYERS, inherited_block_residual=inherited)
        )
        assert residual.num_sealed == INHERITED_SNAPSHOTS
        for local_idx in range(num_layers):
            hidden = residual.open(local_idx)
            residual.write(module(hidden))
            residual.release(hidden)
            pre_ffn = residual.read()
            residual.write(module(pre_ffn))
            residual.release(pre_ffn)
        # `discard`, not `handoff`: mid-rank the next read settles the last deferred write, and `baseline` has no
        # counterpart to handoff's add, which would land in the boundary layer.
        residual.discard()

    def baseline(num_layers):
        """`forward` without AttnRes: the same setup and stand-ins, so the difference is AttnRes alone."""
        held = [ttnn.clone(sealed_base), module(live_base)]
        for _ in range(2 * num_layers):
            held.append(module(held[-1]))
        for tensor in held:
            ttnn.deallocate(tensor)

    walked = 1 + ORDINARY_LAYERS
    variants = {
        "forward_1": lambda: forward(1),
        "baseline_1": lambda: baseline(1),
        f"forward_{walked}": lambda: forward(walked),
        f"baseline_{walked}": lambda: baseline(walked),
    }
    prepared = {}
    try:
        for name, fn in variants.items():
            prepared[name] = _Prepared(mesh_device, fn, traced=mode == "traced")
        walls = _interleaved_walls_us(mesh_device, {name: p.call for name, p in prepared.items()})
        _profiled(mesh_device, prepared[f"forward_{walked}"].call)  # exclude first-replay jitter
        samples = []
        for _ in range(_SAMPLES):
            # Let the previous pass's records drain, so none land in this window.
            ttnn.synchronize_device(mesh_device)
            time.sleep(0.5)
            samples.append(_profiled(mesh_device, prepared[f"forward_{walked}"].call))
    finally:
        for p in prepared.values():
            p.release()
        for tensor in (live_base, sealed_base, q_out, *q_pre, *q_post):
            ttnn.deallocate(tensor)

    wall = {name: statistics.median(values) for name, values in walls.items()}
    first = wall["forward_1"] - wall["baseline_1"]
    ordinary = (wall[f"forward_{walked}"] - wall[f"baseline_{walked}"] - first) / ORDINARY_LAYERS
    e2e_us = {"boundary": first, "ordinary": ordinary}

    # Per sample: the boundary layer, and the ordinary layers' mean, in device time (us).
    per_sample = []
    for programs in samples:
        layers = _split(programs, walked)
        layer_us = [sum(p["ns"] for p in layer) / 1e3 for layer in layers]
        per_sample.append(((layer_us[0], statistics.mean(layer_us[1:])), layers))
    totals = [sum(us) for us, _ in per_sample]
    median_layers = per_sample[totals.index(sorted(totals)[len(totals) // 2])][1]
    device_us = {name: statistics.median(us[i] for us, _ in per_sample) for i, name in enumerate(LAYERS)}

    _log_ops("boundary", median_layers[0])
    _log_ops("ordinary", median_layers[1])

    margin = _MARGIN[mode]
    fabric = ttnn.get_fabric_config().name
    logger.info(
        f"AttnRes layers {LAYER}-{LAYER + ORDINARY_LAYERS} (S = {INHERITED_SNAPSHOTS} -> {INHERITED_SNAPSHOTS + 1}), "
        f"{num_tokens} tokens, {layout} {fabric}, {mode}:"
    )
    within_band = {}
    for name, label in zip(LAYERS, (f"layer {LAYER}", f"mean of {LAYER + 1}-{LAYER + ORDINARY_LAYERS}")):
        line = f"  {name:8s} ({label}) e2e {e2e_us[name]:9.1f} us"
        if references is not None:
            reference = references[name]
            within_band[name] = abs(e2e_us[name] - reference) <= reference * margin
            line += f"   ref {reference:9.1f} ±{margin:.0%}  {'ok' if within_band[name] else 'OUT OF BAND'}"
        logger.info(line)
        logger.info(f"    device {device_us[name]:9.1f} us   (sum of slowest-chip program windows)")
    logger.info(
        "  host clock medians: "
        + ", ".join(f"{name} {us:.1f}" for name, us in wall.items())
        + f" us (of {_WALL_SAMPLES} each, interleaved)"
    )

    result = {
        "mode": mode,
        "layout": layout,
        "fabric_config": fabric,
        "mesh": list(mesh_device.shape),
        "tokens": num_tokens,
        "first_layer": LAYER,
        "ordinary_layers": ORDINARY_LAYERS,
        "inherited_snapshots": INHERITED_SNAPSHOTS,
        "e2e_us": e2e_us,
        "device_us": device_us,
        "wall_us": walls,
        "device_samples_us": [us for us, _ in per_sample],
        "reference_e2e_us": references,
        "margin_pct": margin * 100.0,
    }
    print("ATTN_RES_PERF=" + json.dumps(result, sort_keys=True))
    tp_wrapped = layout != _GALAXY or tp_axis_is_wrapped(mesh_device)
    gated = not report_only and not _REPORT_ONLY and tp_wrapped
    emit_perf_metrics(
        f"kimi_k3_attn_res_layer_{mode}_{layout}",
        f"Kimi-K3 AttnRes layer perf ({mode}, {layout})",
        {**result, "gated": gated, "within_band": all(within_band.values()) if gated else None},
    )
    if report_only:
        return
    if _REPORT_ONLY:
        pytest.skip("ATTN_RES_PERF_REPORT_ONLY=1; not gating")
    if not tp_wrapped:
        pytest.skip(f"TP axis not wrapped: the {layout} references assume the TP ring")
    for name in LAYERS:
        _assert_within_margin(f"{layout} {mode} {name} layer e2e", e2e_us[name], references[name], margin)

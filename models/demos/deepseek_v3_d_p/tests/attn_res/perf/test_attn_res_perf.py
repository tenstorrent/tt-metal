# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 AttnRes performance on Galaxy: a boundary layer and an ordinary layer, end to end, traced and untraced.

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
layers add over it, divided by their count. A single ordinary layer is the difference of two ~5 ms eager
forwards and moved by 17% between runs; averaged over 8 it holds to a few percent. Traced replays captured
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
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params, tp_axis_is_wrapped
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
# Galaxy end-to-end time (us) of each layer's AttnRes on the CI pool (bh_sc1_high_power). Traced and untraced
# share the reference and the band: production prefill runs untraced, and the untraced walk is held to the
# traced time. Within a box the spread is under 2%; bh-glx-120-b07u02 measures the traced boundary layer ~6%
# above CI, so a run off the CI pool may need ATTN_RES_PERF_REPORT_ONLY=1.
_REFERENCE_E2E_US = {"boundary": 1656.0, "ordinary": 141.0}
_MARGIN = 0.05
# The stand-in module: never issued by AttnRes, so its programs are identifiable by kernel path.
_MODULE_KERNEL = "normalization/softmax"


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
        self.trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, self.trace_id, cq_id=0)
        self.call = lambda: ttnn.execute_trace(mesh_device, self.trace_id, cq_id=0, blocking=False)
        ttnn.execute_trace(mesh_device, self.trace_id, cq_id=0, blocking=True)

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
    """One forward's programs -> (each layer's AttnRes programs, the handoff's).

    The forward dispatches `[setup] [m] open(0) [m] read(0) [m] open(1) ... read(L-1) [m] handoff`. What
    precedes the first stand-in is test setup. Layer `l` is the two segments after stand-ins `2l` and `2l + 1`.
    The handoff settles the last deferred write with a plain add, which mid-rank is folded into the next layer's
    read instead, so it is kept apart.
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
    return [segments[2 * l] + segments[2 * l + 1] for l in range(num_layers)], segments[-1]


def _log_ops(name, programs):
    by_op = defaultdict(lambda: [0, 0.0])
    for p in programs:
        by_op[p["op"]][0] += 1
        by_op[p["op"]][1] += p["ns"]
    logger.info(f"{name} layer, per op (median sample):")
    for op, (count, ns) in sorted(by_op.items(), key=lambda kv: -kv[1][1]):
        logger.info(f"  {ns / 1e3:8.1f} us  x{count:<3d} {op}")


@pytest.mark.parametrize("mode", ["traced", "untraced"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=L1_SMALL_SIZE, trace_region_size=128 * 1024 * 1024, **EXACT_BOX),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
def test_attn_res_layer_perf(mesh_device, device_params, mode):
    """Gate the AttnRes end-to-end time of a boundary layer and an ordinary layer, and report their device time."""
    require_high_power("kimi_k3 AttnRes perf")
    require_realtime_profiler("the AttnRes device-time report")
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    assert (num_tokens, op.tp_factor) == (5120, 4), "must run Galaxy's K3 prefill chunk"

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
        live, sealed = residual.handoff()
        for tensor in (live, *sealed):
            ttnn.deallocate(tensor)

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
        layers, handoff = _split(programs, walked)
        layer_us = [sum(p["ns"] for p in layer) / 1e3 for layer in layers]
        per_sample.append(((layer_us[0], statistics.mean(layer_us[1:])), layers, sum(p["ns"] for p in handoff) / 1e3))
    totals = [sum(us) for us, _, _ in per_sample]
    median_layers = per_sample[totals.index(sorted(totals)[len(totals) // 2])][1]
    device_us = {name: statistics.median(us[i] for us, _, _ in per_sample) for i, name in enumerate(LAYERS)}
    handoff_us = statistics.median(handoff for _, _, handoff in per_sample)

    _log_ops("boundary", median_layers[0])
    _log_ops("ordinary", median_layers[1])

    references, margin = _REFERENCE_E2E_US, _MARGIN
    logger.info(
        f"AttnRes layers {LAYER}-{LAYER + ORDINARY_LAYERS} (S = {INHERITED_SNAPSHOTS} -> {INHERITED_SNAPSHOTS + 1}), "
        f"{num_tokens} tokens, SP8xTP4 torus-XY, {mode}:"
    )
    failures = []
    for name, label in zip(LAYERS, (f"layer {LAYER}", f"mean of {LAYER + 1}-{LAYER + ORDINARY_LAYERS}")):
        line = f"  {name:8s} ({label}) e2e {e2e_us[name]:9.1f} us"
        reference = references[name]
        if reference is not None:
            lower, upper = reference * (1 - margin), reference * (1 + margin)
            ok = lower <= e2e_us[name] <= upper
            line += f"   ref {reference:9.1f} ±{margin:.0%}  {'ok' if ok else 'OUT OF BAND'}"
            if not ok:
                failures.append(f"{name} layer e2e {e2e_us[name]:.1f} us outside [{lower:.1f}, {upper:.1f}] us")
        logger.info(line)
        logger.info(f"    device {device_us[name]:9.1f} us   (sum of slowest-chip program windows)")
    logger.info(
        "  host clock medians: "
        + ", ".join(f"{name} {us:.1f}" for name, us in wall.items())
        + f" us (of {_WALL_SAMPLES} each, interleaved)"
    )
    logger.info(f"  handoff add {handoff_us:.1f} us   (in the forwards, cancels between them)")

    result = {
        "mode": mode,
        "fabric_config": ttnn.get_fabric_config().name,
        "mesh": list(mesh_device.shape),
        "tokens": num_tokens,
        "first_layer": LAYER,
        "ordinary_layers": ORDINARY_LAYERS,
        "inherited_snapshots": INHERITED_SNAPSHOTS,
        "e2e_us": e2e_us,
        "device_us": device_us,
        "wall_us": walls,
        "device_samples_us": [us for us, _, _ in per_sample],
        "handoff_add_us": handoff_us,
        "reference_e2e_us": references,
        "margin_pct": margin * 100.0,
    }
    print("ATTN_RES_PERF=" + json.dumps(result, sort_keys=True))
    gated = not _REPORT_ONLY and tp_axis_is_wrapped(mesh_device) and all(v is not None for v in references.values())
    emit_perf_metrics(
        f"kimi_k3_attn_res_layer_{mode}_SP8xTP4",
        f"Kimi-K3 AttnRes layer perf ({mode}, SP8xTP4)",
        {**result, "gated": gated, "within_band": not failures if gated else None},
    )
    if gated:
        assert not failures, f"AttnRes {mode}:\n  " + "\n  ".join(failures)

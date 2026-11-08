# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 AttnRes op-level device performance: each of the three fused ops on its own gate.

`test_attn_res_perf.py` gates whole layers, where a slowdown in one op can hide behind a speedup in
another. This gates each op's device kernel duration: the slowest chip's realtime-profiler window for
the op's program, the median over `_ITERATIONS` runs.

The ops run as the boundary layer of a non-first pipeline rank issues them (`TtAttnResWalk`), at K3's
shapes: the 5120-token chunk over the SP axis (640 rows per chip on 8x4, 1280 on a 4x4 stage) and
`d = 7168` over TP4. Each iteration is two batches, each followed by a read of its site 0:

    owed:  attn_res_block_stats -> attn_res_block_mix -> attn_res_gather_softmax
           the read owed across the rank boundary, one site over the six inherited snapshots; it
           settles nothing, as the walk opens on the live stream the upstream rank handed over
    block: attn_res_block_stats -> attn_res_block_mix -> attn_res_gather_softmax
           the boundary block's 24 sites over those and the sealed live stream, and an ordinary
           read of it, settling a write

with one persistent gather buffer per sealed depth, exchange scratch and semaphores held across
iterations the way `TtAttnRes` holds them. The read after each batch is also what makes reusing the
gather buffers safe.

The 4x4 arm needs the Galaxy carved to a subtorus (see `harness.SUBTORUS_4X4_ENV`):
    TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31
    TT_MESH_GRAPH_DESC_PATH=models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto
"""

from __future__ import annotations

import json
import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.ci_failure import assert_perf_gate
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.attn_res.device_determinism import deterministic_input
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import EXACT_BOX, HIDDEN_SIZE, L1_SMALL_SIZE
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params, tp_axis_is_wrapped
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import EPS, STATS_FIDELITY
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links, per_axis_topology
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(900)]

GLOBAL_TOKENS = 5120
TP_AXIS = 1
SP_AXIS = 0
# The boundary layer's sealed set: six inherited snapshots and the sealed live stream.
PLANES = (6, 1)
# Each batch as (snapshots it scores, a prefix of PLANES; read sites), in dispatch order.
BATCHES = {"owed": (1, 1), "block": (2, 24)}
_WARMUP = 3
_ITERATIONS = 20
# Set to report without gating, e.g. while re-baselining.
_REPORT_ONLY = os.environ.get("ATTN_RES_PERF_REPORT_ONLY") == "1"
# Band around each op's reference. block_stats' window runs until the slowest peer's statistics land, so it also
# times how far the chips are apart, and that differs by machine: two CI Galaxies put its median at 139.0 and 119.0
# us, each tight on its own. Its band covers both and still catches a regression of more than ~15%.
_MARGIN = {"attn_res_block_stats": 0.15, "attn_res_block_mix": 0.05, "attn_res_gather_softmax": 0.05}
# Each op's program, by the kernel directory it is built from.
OPS = ("attn_res_block_stats", "attn_res_block_mix", "attn_res_gather_softmax")
# One iteration's programs in dispatch order, each gated on its own.
STEPS = tuple(f"{batch}/{op}" for batch in BATCHES for op in OPS)

# Device kernel duration (us) per step, as CI hardware (bh_sc1_high_power) measures it. None reports without gating.
#
# Until tt_metal_tracker-ao3i.10 the boundary was one 25-site batch over C = 7 with the owed read closed; the CI pool
# measured it (runs 41962421284 and 42058182913, medians over 20 synced blocks: block_stats 139.0 / 119.0, block_mix
# 275.5 / 275.6, gather_softmax 67.5 / 68.2; run 42340900388: 137.8 / 275.0 / 69.0).
#
# SP8xTP4 references are derived, not measured on the pool: each step's median on one off-pool Galaxy at 130 W
# (bh-glx-120-b07u08, 2026-11-08, 7 runs) times that Galaxy's pool ratio for the same op, from the old 25-site code run
# there interleaved (3 runs: block_mix 292.1, gather_softmax 69.1 us) against run 42340900388. That Galaxy runs
# block_mix ~6% slower than the pool and gather_softmax at the pool's speed. Medians there: block/block_mix 290.3,
# block/gather_softmax 68.6, owed/block_mix 95.2 us. block_stats and the owed gather_softmax settle per process into
# one of several levels (block_stats 76 / 117 / 139 us there, 119 / 139 on the pool; owed gather_softmax 54 / 77 /
# 113 us), as both wait on their peers, so no ratio holds for them: block/block_stats keeps the pool's mean of its two
# levels, and the owed step's two only report.
#
# SP4xTP4 has no CI job: it was measured on a Galaxy off the CI pool (bh-glx-120-b07u02, 2026-11-05), which runs the
# end-to-end boundary layer of `test_attn_res_perf.py` at ~416 us against its 394 us CI reference, and is scaled by
# 394 / 416. Local medians, synced: block_stats 209.9 / 218.4 / 208.4, block_mix 568.4 / 570.2 / 570.2,
# gather_softmax 120.2 / 121.2 / 123.4.
# Locally, run with ATTN_RES_PERF_REPORT_ONLY=1.
_CI_OVER_LOCAL = 394.0 / 416.0
_REFERENCE_US = {
    "SP8xTP4": {
        "owed/attn_res_block_stats": None,
        "owed/attn_res_block_mix": 89.6,
        "owed/attn_res_gather_softmax": None,
        "block/attn_res_block_stats": 129.0,
        "block/attn_res_block_mix": 273.4,
        "block/attn_res_gather_softmax": 68.5,
    },
    "SP4xTP4": {
        "owed/attn_res_block_stats": None,
        "owed/attn_res_block_mix": None,
        "owed/attn_res_gather_softmax": None,
        "block/attn_res_block_stats": round(209.9 * _CI_OVER_LOCAL, 1),
        "block/attn_res_block_mix": round(570.2 * _CI_OVER_LOCAL, 1),
        "block/attn_res_gather_softmax": round(121.2 * _CI_OVER_LOCAL, 1),
    },
}


def _assert_within_margin(what: str, measured_us: float, reference_us: float, margin: float) -> None:
    lower, upper = reference_us * (1.0 - margin), reference_us * (1.0 + margin)
    assert_perf_gate(
        lower <= measured_us <= upper,
        (
            f"{what} {measured_us:.1f} us is outside performance range [{lower:.1f}, {upper:.1f}] us "
            f"(reference {reference_us:.1f} us ± {margin:.0%})"
        ),
    )


def _op(step: str) -> str:
    return step.split("/", 1)[1]


def _mapper(mesh_device, sp_dim, tp_dim):
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = sp_dim, tp_dim
    return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))


def _to_device(mesh_device, tensor, mapper, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)


def _semaphore(mesh_device, cores):
    return ttnn.create_global_semaphore(mesh_device, ttnn.CoreRangeSet([cores]), 0)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=L1_SMALL_SIZE, **EXACT_BOX),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
        pytest.param(
            (4, 4),
            torus_xy_device_params(l1_small_size=L1_SMALL_SIZE, **EXACT_BOX),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="SP4xTP4-torus-xy",
        ),
    ],
    indirect=True,
)
def test_attn_res_op_perf(mesh_device, device_params):
    """Gate each AttnRes op's device kernel duration at K3's boundary-layer shapes."""
    mesh_shape = tuple(mesh_device.shape)
    tp_factor, sp_factor = mesh_shape[TP_AXIS], mesh_shape[SP_AXIS]
    layout = f"SP{sp_factor}xTP{tp_factor}"
    require_high_power(f"kimi_k3 AttnRes op {layout} perf")
    references = _REFERENCE_US[layout]
    assert set(references) == set(STEPS), f"{layout} references name {sorted(references)}, not {sorted(STEPS)}"
    require_realtime_profiler("the AttnRes op perf gate")
    num_links = get_num_links(mesh_device, TP_AXIS)
    topology = per_axis_topology()[TP_AXIS]
    grid = mesh_device.compute_with_storage_grid_size()

    stream_mapper = _mapper(mesh_device, 2, 3)
    sealed = [
        _to_device(mesh_device, deterministic_input(1, planes, GLOBAL_TOKENS, HIDDEN_SIZE, seed=i), stream_mapper)
        for i, planes in enumerate(PLANES)
    ]

    def score_columns(num_sites):
        columns = torch.zeros(1, 1, HIDDEN_SIZE, 32)
        columns[..., 1 : 1 + num_sites] = deterministic_input(
            HIDDEN_SIZE, num_sites, seed=len(PLANES), scale=0.02, dtype=torch.float32
        )
        return _to_device(mesh_device, columns, _mapper(mesh_device, None, 2))

    def gathered(num_snapshots):
        zeros = torch.zeros(1, sum(PLANES[:num_snapshots]), GLOBAL_TOKENS, 32 * tp_factor)
        return _to_device(mesh_device, zeros, _mapper(mesh_device, 2, None), ttnn.float32)

    running_sum, pending = (
        _to_device(mesh_device, deterministic_input(1, 1, GLOBAL_TOKENS, HIDDEN_SIZE, seed=seed), stream_mapper)
        for seed in (10, 11)
    )
    # Per batch: its sealed set, score columns, persistent gather buffer, sites, and the write its read
    # settles: none for the owed read, one for the block's.
    batches = [
        (sealed[:num_snapshots], score_columns(num_sites), gathered(num_snapshots), num_sites, batch_pending)
        for (num_snapshots, num_sites), batch_pending in zip(BATCHES.values(), (None, pending))
    ]
    stats_semaphore = _semaphore(
        mesh_device,
        ttnn.CoreRange(ttnn.CoreCoord(grid.x - num_links, grid.y - 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1)),
    )
    query = _to_device(
        mesh_device, deterministic_input(1, 1, 1, HIDDEN_SIZE, seed=12, scale=0.05), _mapper(mesh_device, None, 3)
    )
    exchange = _to_device(
        mesh_device, torch.zeros(1, 2 * tp_factor, GLOBAL_TOKENS, 1), _mapper(mesh_device, 2, None), ttnn.float32
    )
    read_semaphore = _semaphore(
        mesh_device, ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))
    )

    def block():
        for batch_sealed, columns, batch_gathered, num_sites, batch_pending in batches:
            stats = ttnn.experimental.deepseek_prefill.attn_res_block_stats(
                batch_sealed,
                columns,
                gathered=batch_gathered,
                cluster_axis=TP_AXIS,
                semaphore=stats_semaphore,
                topology=topology,
                num_links=num_links,
                compute_kernel_config=STATS_FIDELITY,
            )
            partials, shifts, masses = ttnn.experimental.deepseek_prefill.attn_res_block_mix(
                batch_sealed, stats, num_sites=num_sites, eps=EPS
            )
            outputs = ttnn.experimental.deepseek_prefill.attn_res_gather_softmax(
                partials,
                running_sum,
                shifts,
                masses,
                query,
                exchange,
                read_semaphore,
                site=0,
                cluster_axis=TP_AXIS,
                topology=topology,
                inv_hidden_size=1.0 / HIDDEN_SIZE,
                eps=EPS,
                pending=batch_pending,
            )
            for tensor in (partials, shifts, masses, *outputs):
                ttnn.deallocate(tensor)

    def run():
        # Synced per block: `attn_res_block_stats`' window runs until its slowest peer's statistics land, so
        # without a common start it would also time how far the chips drifted apart over the blocks before.
        for _ in range(_ITERATIONS):
            ttnn.synchronize_device(mesh_device)
            block()

    for _ in range(_WARMUP):
        block()
    ttnn.synchronize_device(mesh_device)
    _, per_program = profile_realtime_program_merged(mesh_device, run)

    # The window's programs in dispatch order, each matched to its step by position and checked by its op.
    durations_us = {step: [] for step in STEPS}
    position = 0
    for entry in per_program.values():
        sources = [source.replace("\\", "/") for source in entry["kernel_sources"]]
        matched = [op for op in OPS if any(f"/{op}/" in source for source in sources)]
        assert len(matched) <= 1, f"a program matches several ops: {matched}"
        if matched:
            step = STEPS[position % len(STEPS)]
            assert step.endswith("/" + matched[0]), f"program {position} is {matched[0]}, expected step {step}"
            durations_us[step].append(entry["duration_ns"] / 1e3)
            position += 1
    for step, samples in durations_us.items():
        assert len(samples) == _ITERATIONS, f"expected {_ITERATIONS} {step} programs in the window, got {len(samples)}"
    device_us = {step: statistics.median(samples) for step, samples in durations_us.items()}

    fabric = ttnn.get_fabric_config().name
    logger.info(
        f"AttnRes ops, (C, sites) per batch "
        f"{ {batch: (sum(PLANES[:n]), sites) for batch, (n, sites) in BATCHES.items()} }, {GLOBAL_TOKENS} tokens "
        f"({GLOBAL_TOKENS // sp_factor}/chip), {layout} {fabric}, {num_links} links, median of {_ITERATIONS}:"
    )
    within_band = {}
    for step in STEPS:
        samples, margin = durations_us[step], _MARGIN[_op(step)]
        line = f"  {step:32s} {device_us[step]:8.1f} us   (min {min(samples):.1f}, max {max(samples):.1f})"
        if references[step] is not None:
            within_band[step] = abs(device_us[step] - references[step]) <= references[step] * margin
            line += f"   ref {references[step]:8.1f} ±{margin:.0%}  {'ok' if within_band[step] else 'OUT OF BAND'}"
        logger.info(line)

    result = {
        "layout": layout,
        "fabric_config": fabric,
        "mesh": list(mesh_shape),
        "tokens": GLOBAL_TOKENS,
        "batches": {
            batch: {"num_candidates": sum(PLANES[:n]), "num_sites": sites} for batch, (n, sites) in BATCHES.items()
        },
        "num_links": num_links,
        "device_us": device_us,
        "device_samples_us": durations_us,
        "reference_us": references,
        "margin_pct": {op: margin * 100.0 for op, margin in _MARGIN.items()},
    }
    print("ATTN_RES_OP_PERF=" + json.dumps(result, sort_keys=True))
    tp_wrapped = tp_axis_is_wrapped(mesh_device)
    gated_steps = [step for step in STEPS if references[step] is not None]
    gated = not _REPORT_ONLY and tp_wrapped
    emit_perf_metrics(
        f"kimi_k3_attn_res_ops_{layout}",
        f"Kimi-K3 AttnRes op perf ({layout})",
        {**result, "gated": gated, "within_band": all(within_band.values()) if gated else None},
    )
    if _REPORT_ONLY:
        pytest.skip("ATTN_RES_PERF_REPORT_ONLY=1; not gating")
    if not tp_wrapped:
        pytest.skip(f"TP axis not wrapped: the {layout} references assume the TP ring")
    for step in gated_steps:
        _assert_within_margin(f"{layout} {step}", device_us[step], references[step], _MARGIN[_op(step)])

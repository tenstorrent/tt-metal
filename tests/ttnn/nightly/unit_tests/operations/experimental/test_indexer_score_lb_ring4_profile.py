# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DEVICE-SIDE profiling of the ring-fused indexer_score (root-cause why fused > AG+score).

Unlike test_indexer_score_lb_ring4_perf.py (host wall-clock, confounded by dispatch overhead), this reads the
in-process tracy device profiler (ttnn.ReadDeviceProfiler + get_latest_programs_perf_data) to get the PURE
on-device kernel duration per program, per chip -- no subprocess, no mesh-close teardown throw.

Decision tree from the three device durations (per chip, take max across chips = the bottleneck device):
  D_fused ~= D_ag                  -> compute fully hidden (IDEAL: max(AG,score))
  D_fused ~= D_ag + D_score        -> zero overlap
  D_fused  > D_ag + D_score        -> co-scheduling CONTENTION (something got slower)

Also dumps the raw generated/profiler/.logs/profile_log_device.csv path so a per-core span parse can separate
the AG-worker-core span from the compute-core span inside the single fused program.

Run:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
  INDEXER_FUSED_PROFILE=1 scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score_lb_ring4_profile.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    glx_config,
    QB_CASES,
    QB_IDS,
    QB_DIM,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score_lb_ring4_ag_equiv import (
    _open_ring4_ccl,
    _close_ring4_ccl,
    _build_ring4_fused_inputs,
    _persistent_buffer,
    _reset,
    SP_AXIS,
    T,
)

pytestmark = [
    pytest.mark.skipif(os.environ.get("INDEXER_FUSED_PROFILE") != "1", reason="set INDEXER_FUSED_PROFILE=1 to run"),
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() < 8, reason="ring-of-4 needs the 8-chip LoudBox (2x4)"),
]

_ITERS = 5
_NLINKS = int(os.environ.get("INDEXER_FUSED_NLINKS", "1"))
_KCHUNK = int(os.environ.get("INDEXER_FUSED_KCHUNK", "0"))  # override k_chunk_size (0 = glx default)


def _cfg(heads):
    c = glx_config(heads)
    if _KCHUNK:
        return ttnn.IndexerScoreProgramConfig(
            q_chunk_size=c.q_chunk_size, k_chunk_size=_KCHUNK, head_group_size=c.head_group_size
        )
    return c


def _read_durations(submesh):
    """Return {runtime_id: {'dur': ns_max_over_chips, 'per_chip': {chip: (start,end,dur)}}} for the latest read."""
    ttnn.ReadDeviceProfiler(submesh)
    perf_by_chip = ttnn.get_latest_programs_perf_data()
    out = {}
    for chip, programs in perf_by_chip.items():
        for program in programs:
            uid = program.program_execution_uid
            rid = uid.runtime_id
            for name, result in program.program_analyses_results.items():
                slot = out.setdefault((rid, name), {"per_chip": {}})
                slot["per_chip"][chip] = (int(result.start_timestamp), int(result.end_timestamp), int(result.duration))
    for slot in out.values():
        slot["dur_max"] = max(v[2] for v in slot["per_chip"].values())
        # global span across chips (min start .. max end) -- for the fused single program this is the wall span
        slot["span"] = max(v[1] for v in slot["per_chip"].values()) - min(v[0] for v in slot["per_chip"].values())
    return out


def _run_profiled(label, fn, submesh, stall_group):
    """Run fn _ITERS times (blocking each), then read the profiler; return the per-program duration table."""
    for _ in range(_ITERS):
        fn()
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
    return _read_durations(submesh)


def _peak(tab):
    """Max device kernel-duration (us) across every program/analysis in the table (bottleneck chip)."""
    return max((slot["dur_max"] for slot in tab.values()), default=0) / 1000.0


def _matched_T(heads, grid_x, grid_y):
    """Rescale K's T so a STANDALONE indexer_score does the SAME bands-per-column as the FUSED op.

    The fused op reserves reserved_cols columns of the grid for the AG workers, so its compute runs on
    (grid_x - reserved_cols) columns while a standalone score gets all grid_x. Bands are dealt one-per-column,
    so the bottleneck column carries ceil(bands / cols) bands -- i.e. the fused op does ~grid_x/(grid_x-1) MORE
    work per core purely from the reserved column. To compare apples-to-apples we scale the standalone's T up so
    its bottleneck column matches the fused op's, then per-core work is identical and the fused-vs-matched delta
    is the PURE co-schedule cost (DRAM contention + last-shard tail), with the core-count difference removed.

    Returns (T_matched, diag) where diag documents the band arithmetic for the log."""
    kc_tiles = _cfg(heads).k_chunk_size // 32
    bands = (T // 32) // kc_tiles
    reserved = (_NLINKS * 2 + grid_y - 1) // grid_y
    fused_cols = grid_x - reserved
    std_cols = min(bands, grid_x)
    fused_perc = (bands + fused_cols - 1) // fused_cols  # bands on the fused bottleneck column
    std_perc = (bands + std_cols - 1) // std_cols  # bands on the standalone bottleneck column (native T)
    bands_matched = fused_perc * grid_x  # even split over grid_x cols -> std bottleneck == fused_perc
    t_matched = bands_matched * kc_tiles * 32
    diag = (
        f"grid={grid_x}x{grid_y} bands={bands} kc_tiles={kc_tiles} | native: std {std_cols}cols→{std_perc}/col "
        f"vs fused {fused_cols}cols→{fused_perc}/col | matched T {T}→{t_matched} "
        f"(bands {bands}→{bands_matched} = {bands_matched // grid_x}/col over {grid_x} cols == fused {fused_perc}/col)"
    )
    return t_matched, diag


def _summ(label, table):
    """Log every distinct (runtime_id, analysis) program duration in the table (ns -> us)."""
    logger.info(f"===== {label} device programs =====")
    rows = []
    for (rid, name), slot in sorted(table.items()):
        durs = sorted(v[2] for v in slot["per_chip"].values())
        rows.append((rid, name, slot["dur_max"], durs))
    for rid, name, dmax, durs in rows:
        percc = ", ".join(f"{d/1000:.1f}" for d in durs)
        logger.info(f"  rid={rid} [{name}] max={dmax/1000:.1f}us  per-chip us=[{percc}]")
    return rows


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_ring4_fused_device_profile(case_id, heads, block_cyclic):
    """Device-profiler kernel durations (AG-only vs score-only vs FUSED, per chip) to root-cause the overlap:
    fused ~= max(AG, score) means the smaller phase is hidden; fused ~= AG + score means zero overlap."""
    submesh, parent, _unused_sems, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_dev, w_dev, k_local, k_gathered, bc, sems = _build_ring4_fused_inputs(submesh, heads, block_cyclic)

        def fused_once():
            _reset(sems)
            ttnn.experimental.indexer_score_dsa_fused(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                sems,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=_NLINKS,
                ag_sub_device_id=subdevice_id,
                program_config=_cfg(heads),
                **bc,
            )

        def ag_once():
            _reset(sems)
            ttnn.experimental.ring_attention_all_gather_async(
                [k_local],
                persistent_output_buffer=[k_gathered],
                dim=2,
                multi_device_global_semaphore=sems,
                cluster_axis=SP_AXIS,
                mesh_device=submesh,
                num_links=_NLINKS,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                subdevice_id=subdevice_id,
            )

        def score_once():
            ttnn.experimental.indexer_score_dsa(
                q_dev, k_gathered, w_dev, cluster_axis=SP_AXIS, program_config=_cfg(heads), **bc
            )

        # warmup (compile) + flush profiler so the first real read is clean
        fused_once()
        ag_once()
        score_once()
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        _read_durations(submesh)

        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"########## {case_id}-{layout} (heads={heads}) ##########")
        ag_tab = _run_profiled("AG-only", ag_once, submesh, stall_group)
        _summ("AG-only", ag_tab)
        score_tab = _run_profiled("score-only", score_once, submesh, stall_group)
        _summ("score-only", score_tab)
        fused_tab = _run_profiled("FUSED", fused_once, submesh, stall_group)
        fused_rows = _summ("FUSED", fused_tab)

        # Work-per-core-matched score-only: rescale K's T so a STANDALONE score does the same bands-per-column
        # as the fused op (see _matched_T). Removes the reserved-AG-column core-count penalty from the fused-vs-
        # score gap, isolating the pure co-schedule cost. Contiguous only (T need not divide chunk_global; the
        # per-band compute cost is layout-independent with aligned KC), so we run it once on the contiguous pass.
        matched_us = None
        if not block_cyclic:
            grid = submesh.compute_with_storage_grid_size()
            t_matched, diag = _matched_T(heads, grid.x, grid.y)
            logger.info(f">>> work-per-core match [{case_id}]: {diag}")
            k_matched = _persistent_buffer(submesh, torch.zeros(1, 1, t_matched, QB_DIM, dtype=torch.bfloat16))

            def matched_score_once():
                ttnn.experimental.indexer_score_dsa(
                    q_dev, k_matched, w_dev, cluster_axis=SP_AXIS, program_config=_cfg(heads)
                )

            matched_score_once()  # warmup/compile
            ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
            _read_durations(submesh)
            matched_tab = _run_profiled("score-matched", matched_score_once, submesh, stall_group)
            _summ("score-matched", matched_tab)
            matched_us = _peak(matched_tab)

        ag_us = _peak(ag_tab)
        score_us = _peak(score_tab)
        fused_us = _peak(fused_tab)
        matched_str = (
            f" | score-matched(=fused work/core)={matched_us:.1f} "
            f"fused-vs-matched={fused_us - matched_us:+.1f}us  <== pure co-schedule cost (core-count controlled)"
            if matched_us is not None
            else ""
        )
        logger.info(
            f">>> [{case_id}-{layout}] DEVICE us: AG={ag_us:.1f} score={score_us:.1f} "
            f"fused={fused_us:.1f} | ideal max(AG,score)={max(ag_us,score_us):.1f} "
            f"| zero-overlap floor(AG+score)={ag_us+score_us:.1f} "
            f"| fused-vs-floor={fused_us-(ag_us+score_us):+.1f}us fused-vs-ideal={fused_us-max(ag_us,score_us):+.1f}us"
            + matched_str
        )
        logs = os.path.join(os.getcwd(), "generated/profiler/.logs/profile_log_device.csv")
        logger.info(f">>> raw per-core CSV: {logs} (exists={os.path.exists(logs)})")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)

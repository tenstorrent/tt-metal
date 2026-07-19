# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
QB4 (4-device Blackhole QuietBox) perf comparison: the FUSED ring indexer_score_dsa (SP all-gather
co-scheduled with the score) vs the SEPARATE two-op baseline (standalone ring_attention all-gather THEN a
standalone indexer_score_dsa). The point is to measure how much of the all-gather is HIDDEN behind the
per-column scoring in the fused op.

Topology: FABRIC_1D_RING + ttnn.Topology.Ring on a directly-opened (1, 4) mesh (sp = 4, a natural ring size).

Apples-to-apples compute grid
-----------------------------
The QuietBox Blackhole compute grid is 11x10 (110 cores). The fused op reserves ONE column for the AG
workers, leaving a 10x10 = 100-core compute rectangle for the indexer. To make the SEPARATE baseline
comparable, the standalone indexer is pinned to the SAME 10x10 rectangle via the (benchmarking-only)
IndexerScoreProgramConfig.max_core_grid_x = 10 cap, so both indexers run byte-identical work-splits and the
only difference measured is the AG overlap. (Without the cap the standalone would grab all 11 columns and look
artificially fast, hiding the real question.)

Sequence lengths / shapes (DEPLOYED, to match the perf gate's device kernel duration)
-------------------------------------------------------------------------------------
The deployed per-device shape is the TP=1/SP=32 resharded "short" grid-fill config (glm5_tp1/dsv32_tp1):
Sq = 160 queries/device (5120-query chunk / SP=32), ALL index heads resident (TP=1) -> glm5 = 32 heads,
dsv32 = 64 heads, head_dim 128, head_group_size 0. q_chunk = 32 (QC=1 -> 5 q-groups -> grid-fill num_blocks=2).
T = 56320 keys (1760 tiles), slab-aligned as history 55680 + chunk 640 (chunk_local = Sq = 160, cl_t = 5 tiles).
At this shape the standalone indexer reproduces the deployed ~0.345 ms (glm5) / ~0.636 ms (dsv32) device kernel
duration and ~70/76% matmul util. k_chunk: 'prod' = short_config KC (256/128), 'aligned' = KC | cl_t=5 (160).

Measurement
-----------
Per-program DEVICE KERNEL DURATION [ns] via the in-process device profiler (ttnn.ReadDeviceProfiler +
ttnn.get_latest_programs_perf_data), taken as MAX across the 4 ring chips (the bottleneck device = real
latency), summed across a composite op's programs. Requires a profiler (Tracy) build and the env:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
The test SKIPS (does not fail) if profiling is not enabled, after still checking correctness.

Run:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
    scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa_qb4_perf.py
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    assert_indexer_match,
    indexer_score_dsa_ref,
    _global_inputs,
    _to_slab,
    QB_DIM,
)

DRAM = ttnn.DRAM_MEMORY_CONFIG

pytestmark = [
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() < 4, reason="needs a 4-device Blackhole box"),
]

# ---- ring-of-4 geometry (sp = 4) ------------------------------------------------------------------
RING = 4
SP_AXIS = 1  # the length-4 axis of the (1, 4) mesh == cluster_axis == block-cyclic SP axis

# DEPLOYED per-device shape = the TP=1/SP=32 resharded "short" grid-fill config (test_indexer_score.py:
# SHORT_SQ, short_config, glm5_tp1/dsv32_tp1): Sq = 160 queries/device (a 5120-query chunk / SP=32), ALL index
# heads resident (TP=1, no head shard) -> glm5 = 32 heads, dsv32 = 64 heads, head_dim 128, head_group_size = 0.
SQ = 160  # per-device queries (== SHORT_SQ); 5 q-tile-rows -> QC=1, group_count=5, grid-fill num_blocks=2
QC_TILES = 1  # q_chunk_size = 32 (Sq=160 is not a multiple of 64, so QC must be 1)
CHUNK = RING * SQ  # 640 global prefill chunk (chunk_local = SQ = 160 per SP shard; cl_t = 5 tiles)
T = 56320  # deployed GLX all-gathered keys (1760 tiles) -- the fullest-causal short shape
HISTORY = T - CHUNK  # 55680; 55680 % CHUNK == 0 -> slab-aligned (no straddle)
SLL = T // RING  # 14080 per-device local shard keys
CL_T = SQ // 32  # 5 -- per-SP-shard chunk in tiles (block-cyclic alignment target)

COMPUTE_COLS = 10  # 11x10 grid, 1 column reserved for the fused AG -> 100-core compute both paths

# heads = the DEPLOYED per-device index_n_heads (TP=1, all heads resident): glm5 32, dsv32 64. At Sq=160 x
# 32/64 heads the per-device work equals the deployed shape, so the standalone indexer lands at the deployed
# ~0.345 ms (glm5) / ~0.636 ms (dsv32) device kernel duration and ~70/76% matmul utilization.
CASES = [("glm5", 32), ("dsv32", 64)]
CASE_IDS = [c[0] for c in CASES]

# k_chunk per (mode, heads). cl_t = SQ/32 = 5 tiles.
#   'prod'    = the deployed short_config KC (glm5 32h -> 256 = KC8, dsv32 64h -> 128 = KC4): compute-optimum for
#               the standalone, but KC does NOT divide cl_t=5 -> block-cyclic band straddle -> the FUSED overlap
#               back-loads its tail (factory warns).
#   'aligned' = KC | cl_t=5 (KC=5 -> k_chunk 160, both models): fused overlap-optimal, slightly slower standalone.
KC_MODES = {"prod": {32: 256, 64: 128}, "aligned": {32: 160, 64: 160}}

_KERNEL_KEY = "DEVICE KERNEL DURATION [ns]"
_TRISC1_KEY = "DEVICE TRISC1 KERNEL DURATION [ns]"  # the math (FPU) thread's span
_PROFILER_ON = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"

# Analytical FPU (matrix-unit) utilization, the same model as tests/ttnn/unit_tests/benchmarks/test_benchmark.py:
#   ideal_cycles = (m*k*n) / (32*32*32) * cycle_per_tile / num_cores ; util = ideal_cycles / trisc1_cycles.
# The indexer's FPU work is the q.k^T matmul: per (bottleneck) device m*k*n = heads * Sq * D * T. q/k are bf16
# -> HiFi2 -> 32 cycles per 32x32x32 tile-matmul. Blackhole runs at 1350 MHz (1.35 cycles/ns).
# WARNING: on a PROFILER build the kernels compile with -DPROFILE_KERNEL=1, whose in-loop markers inflate the
# measured TRISC1/kernel duration ~4x, so the FPU_util% printed by the profiler-based comparison below is
# ~4x too LOW (a measurement artifact, NOT the real device efficiency). test_qb4_indexer_true_latency measures
# the TRUE latency with the profiler OFF (trace loop) -> the real ~60-72% FPU util. The AG-hidden % from the
# comparison is unaffected: every op is inflated identically, so the RELATIVE numbers hold.
_BH_FREQ_GHZ = 1.35  # cycles per ns (1350 MHz)
_HIFI2_CYCLES_PER_TILE = 32


def _ideal_fpu_cycles_per_core(heads, num_cores):
    """Ideal HiFi2 matrix-unit cycles per core for the indexer q.k^T matmul on the bottleneck device (full T)."""
    mkn = heads * SQ * QB_DIM * T  # m=Sq, k=D, n=T, folded over heads
    return mkn / (32 * 32 * 32) * _HIFI2_CYCLES_PER_TILE / num_cores


def _per_sp_ref_sq(q_g, k_g, w_g, sp, history, sq):
    """Per-SP-rank full-head DSA reference for a per-device query length `sq` (each rank's chunk_start =
    history + rank*sq), concatenated along seq. Same as test_indexer_score._per_sp_ref but sq-parametrized
    (that helper hardcodes QB_SQ=640; here sq=SQ=160)."""
    refs = []
    for r in range(sp):
        sl = slice(r * sq, (r + 1) * sq)
        refs.append(indexer_score_dsa_ref(q_g[:, :, sl, :], k_g, w_g[:, :, sl, :], history + r * sq))
    return torch.cat(refs, dim=2)


def _open_ccl(mesh_shape):
    """Open `mesh_shape` directly with a RING (torus) 1D fabric, a worker sub-device, and 2 ccl semaphores."""
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
        grid = mesh.compute_with_storage_grid_size()
        ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        worker_sub_device = ttnn.SubDevice([ccl_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        stall_group = [worker_sub_device_id]
        mgr = mesh.create_sub_device_manager([worker_sub_device], 0)
        mesh.load_sub_device_manager(mgr)
        mesh.set_sub_device_stall_group(stall_group)
        ccl_semaphores = [ttnn.create_global_semaphore(mesh, ccl_crs, 0) for _ in range(2)]
        return mesh, ccl_semaphores, worker_sub_device_id, stall_group
    except Exception:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_ccl(mesh):
    try:
        try:
            mesh.reset_sub_device_stall_group()
            mesh.clear_loaded_sub_device_manager()
        finally:
            ttnn.close_mesh_device(mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _prog_max_us(perf_by_chip, key):
    """Per-program `key` analysis: max across chips (bottleneck device), summed across programs -> us (or None)."""
    per_prog = {}
    for _chip, programs in perf_by_chip.items():
        for p in programs:
            uid = p.program_execution_uid
            pk = (uid.runtime_id, uid.trace_id, uid.trace_id_counter)
            res = p.program_analyses_results.get(key)
            if res is not None:
                per_prog[pk] = max(per_prog.get(pk, 0), int(res.duration))
    return sum(per_prog.values()) / 1000.0 if per_prog else None


def _measure(mesh, stall_group, run_fn, iters=8, warmup=3):
    """Dispatch `run_fn` (one op) warmup+iters times; return {"kernel": [us...], "trisc1": [us...]} per iter.

    Returns None if the profiler is off / returns nothing. Each measured iter reads immediately after its own
    sync so get_latest_programs_perf_data() holds only that dispatch's programs. trisc1 is absent for pure
    data-movement ops (the all-gather has no math thread) -> its list stays empty -> FPU util reads as 0."""
    for _ in range(warmup):
        run_fn()
    ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
    if _PROFILER_ON:
        ttnn.ReadDeviceProfiler(mesh)  # flush warmup out of the buffer
    out = {"kernel": [], "trisc1": []}
    for _ in range(iters):
        run_fn()
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        if not _PROFILER_ON:
            continue
        ttnn.ReadDeviceProfiler(mesh)
        perf = ttnn.get_latest_programs_perf_data()
        k_us = _prog_max_us(perf, _KERNEL_KEY)
        t_us = _prog_max_us(perf, _TRISC1_KEY)
        if k_us is not None:
            out["kernel"].append(k_us)
        if t_us is not None:
            out["trisc1"].append(t_us)
    return out if out["kernel"] else None


def _summ(samples):
    """min / median of a us-sample list, or None."""
    if not samples:
        return None
    return min(samples), statistics.median(samples)


def _fpu_util(trisc1_samples, heads, num_cores):
    """Analytical HiFi2 FPU utilization: ideal matmul cycles / measured math-thread (TRISC1) cycles."""
    if not trisc1_samples:
        return 0.0  # no math thread (e.g. the all-gather) -> 0% FPU
    trisc1_cycles = statistics.median(trisc1_samples) * 1000.0 * _BH_FREQ_GHZ  # us -> ns -> cycles
    return _ideal_fpu_cycles_per_core(heads, num_cores) / trisc1_cycles if trisc1_cycles else 0.0


@pytest.mark.parametrize("num_links", [2], ids=["nl2"])
@pytest.mark.parametrize("kc_mode", ["prod", "aligned"], ids=["kc_prod", "kc_aligned"])
@pytest.mark.parametrize("case_id, heads", CASES, ids=CASE_IDS)
def test_qb4_ring_indexer_vs_separate(case_id, heads, kc_mode, num_links):
    """Fused ring indexer_score_dsa vs (standalone ring_attention AG) + (standalone indexer_score_dsa), on a
    ring-of-4. Both indexers pinned to the SAME 10x10 (100-core) grid with the SAME k_chunk (same job/core).
    Measures device kernel duration + FPU util and how much of the AG is hidden; asserts both match the ref.
    Swept over the production KC (16/8) and the fused-aligned KC (20/10) -- see KC_MODES."""
    k_chunk = KC_MODES[kc_mode][heads]
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((1, RING))
    try:
        # ---- host tensors + reference ------------------------------------------------------------
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK, T, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK)  # block-cyclic physical layout the reader inverts
        ref = _per_sp_ref_sq(q_g, k_nat, w_g, RING, HISTORY, SQ)

        shard = ttnn.ShardTensorToMesh(mesh, dim=2)  # SP-shard seq over the 4 devices
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        k_local = ttnn.from_torch(k_bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)

        replicate = ttnn.ReplicateTensorToMesh(mesh)
        # Fused: gathered buffer seeded with ZEROS (AG fills remote bands; reader dual-sources local from k_local).
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        # Separate baseline: full block-cyclic K replicated on every device for the standalone scorer.
        k_full = ttnn.from_torch(k_bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
        # Separate baseline: a fresh AG output buffer to time the standalone all-gather into.
        ag_out_buf = ttnn.from_torch(
            torch.zeros_like(k_nat), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )

        bc = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=SQ)
        fused_cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0)
        # Standalone indexer pinned to the fused op's post-reservation 10-column compute rectangle.
        sep_cfg = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0, max_core_grid_x=COMPUTE_COLS
        )

        # ---- op closures -------------------------------------------------------------------------
        def run_fused():
            return ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Ring,
                num_links=num_links,
                ag_sub_device_id=subdevice_id,
                program_config=fused_cfg,
                **bc,
            )

        def run_ag():
            return ttnn.experimental.ring_attention_all_gather_async(
                [k_local],
                persistent_output_buffer=[ag_out_buf],
                dim=2,
                multi_device_global_semaphore=ccl_semaphores,
                cluster_axis=SP_AXIS,
                mesh_device=mesh,
                num_links=num_links,
                memory_config=DRAM,
                topology=ttnn.Topology.Ring,
                subdevice_id=subdevice_id,
            )

        def run_indexer():
            return ttnn.experimental.indexer_score_dsa(
                q_dev,
                k_full,
                w_dev,
                cluster_axis=SP_AXIS,
                program_config=sep_cfg,
                **bc,
            )

        # ---- correctness (so the perf comparison is meaningful) ----------------------------------
        fused_out = run_fused()
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        fused_t = ttnn.to_torch(fused_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
        assert_indexer_match(fused_t, ref, CHUNK, T, check_neg=True)

        sep_out = run_indexer()
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        sep_t = ttnn.to_torch(sep_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
        assert_indexer_match(sep_t, ref, CHUNK, T, check_neg=True)
        logger.info(f"[{case_id} nl{num_links}] fused + standalone both match the per-SP reference")

        # ---- measurement -------------------------------------------------------------------------
        fused_m = _measure(mesh, stall_group, run_fused)
        ag_m = _measure(mesh, stall_group, run_ag)
        idx_m = _measure(mesh, stall_group, run_indexer)

        if not (fused_m and ag_m and idx_m):
            pytest.skip(
                "device profiler not enabled -- correctness verified; rerun with "
                "TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1"
            )

        fused_min, fused_med = _summ(fused_m["kernel"])
        ag_min, ag_med = _summ(ag_m["kernel"])
        idx_min, idx_med = _summ(idx_m["kernel"])

        # Analytical HiFi2 FPU (matrix-unit) utilization per op (100 compute cores; AG has no math thread -> 0%).
        fused_fpu = _fpu_util(fused_m["trisc1"], heads, COMPUTE_COLS * 10)
        ag_fpu = _fpu_util(ag_m["trisc1"], heads, COMPUTE_COLS * 10)
        idx_fpu = _fpu_util(idx_m["trisc1"], heads, COMPUTE_COLS * 10)

        two_op_floor = ag_med + idx_med  # separate baseline: AG then score, no overlap
        ideal = max(ag_med, idx_med)  # AG fully hidden behind compute (or vice-versa)
        ag_exposed = fused_med - idx_med  # how much the fused op exceeds pure compute == exposed AG
        ag_hidden_frac = max(0.0, min(1.0, 1.0 - ag_exposed / ag_med)) if ag_med else 0.0
        speedup_vs_floor = two_op_floor / fused_med if fused_med else 0.0

        logger.info(
            "\n"
            f"==== QB4 ring-vs-separate  [{case_id} heads={heads} KC={k_chunk // 32}({kc_mode}) num_links={num_links}] ====\n"
            f"  T={T} ({T // 32}t)  history={HISTORY}  chunk={CHUNK}  grid=10x10 (both indexers)\n"
            f"  op                          kernel_us   FPU_util%\n"
            f"  standalone AG            : {ag_med:9.1f}   {100.0 * ag_fpu:6.1f}   (min {ag_min:.1f})\n"
            f"  standalone indexer       : {idx_med:9.1f}   {100.0 * idx_fpu:6.1f}   (min {idx_min:.1f})\n"
            f"  FUSED ring indexer       : {fused_med:9.1f}   {100.0 * fused_fpu:6.1f}   (min {fused_min:.1f})\n"
            f"  two-op floor (AG+idx)    : {two_op_floor:9.1f}\n"
            f"  ideal max(AG, idx)       : {ideal:9.1f}\n"
            f"  -> AG exposed on fused   : {ag_exposed:9.1f} us  (fused - pure compute)\n"
            f"  -> AG hidden             : {100.0 * ag_hidden_frac:6.1f} %\n"
            f"  -> fused speedup vs floor : {speedup_vs_floor:6.2f}x\n"
        )

        # Sanity: the fused op must not be slower than the no-overlap two-op floor.
        assert (
            fused_med <= two_op_floor * 1.05
        ), f"fused {fused_med:.1f}us exceeds the two-op floor {two_op_floor:.1f}us -- AG not hidden"
    finally:
        _close_ccl(mesh)


# ---- investigation: why is the standalone indexer this slow? per-RISC + k_chunk/grid sweep --------
_RISC_KEYS = {
    "NC(rdr)": "DEVICE NCRISC KERNEL DURATION [ns]",
    "BR(wrt)": "DEVICE BRISC KERNEL DURATION [ns]",
    "T0(unp)": "DEVICE TRISC0 KERNEL DURATION [ns]",
    "T1(mth)": "DEVICE TRISC1 KERNEL DURATION [ns]",
    "T2(pck)": "DEVICE TRISC2 KERNEL DURATION [ns]",
}

# (heads, [k_chunk...]); KC = k_chunk/32. Deployed short_config KC: glm5 32h = 8 (256), dsv32 64h = 4 (128).
# cl_t = 5. 'aligned' = KC | cl_t = 5 (k_chunk 160). K/Q CBs are L1-bound (KC*heads), so 64h caps KC low.
_SWEEP = [
    ("glm5", 32, [128, 160, 256]),  # KC 4, 5(=cl_t, aligned), 8(prod)
    ("dsv32", 64, [96, 128, 160]),  # KC 3, 4(prod), 5(=cl_t, aligned)
]
_SWEEP_IDS = [c[0] for c in _SWEEP]


def _measure_risc(mesh, stall_group, run_fn, iters=6, warmup=3):
    """Median us for kernel duration + every per-RISC duration (max across chips), for one op."""
    keys = {"kernel": _KERNEL_KEY, **_RISC_KEYS}
    for _ in range(warmup):
        run_fn()
    ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
    if not _PROFILER_ON:
        return None
    ttnn.ReadDeviceProfiler(mesh)
    acc = {name: [] for name in keys}
    for _ in range(iters):
        run_fn()
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        ttnn.ReadDeviceProfiler(mesh)
        perf = ttnn.get_latest_programs_perf_data()
        for name, key in keys.items():
            v = _prog_max_us(perf, key)
            if v is not None:
                acc[name].append(v)
    return {name: (statistics.median(v) if v else 0.0) for name, v in acc.items()}


@pytest.mark.parametrize("case_id, heads, k_chunks", _SWEEP, ids=_SWEEP_IDS)
def test_qb4_indexer_bottleneck_sweep(case_id, heads, k_chunks):
    """Standalone block-cyclic indexer_score_dsa only: sweep k_chunk x grid (10x10 capped vs 11x10 full) and
    print the per-RISC breakdown (reader/writer/unpack/math/pack) so the actual bottleneck thread and the
    best k_chunk are visible. 'job/core' = bands*groups per core; balanced when band_count % cols == 0."""
    if not _PROFILER_ON:
        pytest.skip("needs the device profiler env (TT_METAL_DEVICE_PROFILER=1 ...)")
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((1, RING))
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK, T, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK)
        ref = _per_sp_ref_sq(q_g, k_nat, w_g, RING, HISTORY, SQ)
        shard = ttnn.ShardTensorToMesh(mesh, dim=2)
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        k_full = ttnn.from_torch(
            k_bc,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        bc = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=SQ)
        tt_tiles = T // 32
        group_count = (SQ // 32) // QC_TILES  # Sqt / QC (q_chunk=32 -> QC=1) = 5

        rows = [
            f"==== indexer bottleneck sweep [{case_id} heads={heads}]  T={T}({tt_tiles}t) grid_y=10 groups={group_count} ====",
            "  KC  grid  bands  b/col   kernel   NC(rdr) BR(wrt) T0(unp) T1(mth) T2(pck)  bottleneck  FPU%",
        ]
        for k_chunk in k_chunks:
            kc = k_chunk // 32
            band_count = -(-tt_tiles // kc)  # ceil
            for grid_cap in (COMPUTE_COLS, 0):  # 10 cols (capped) then 0 == full 11 cols
                cols = min(band_count, grid_cap if grid_cap else 11)
                cfg = ttnn.IndexerScoreProgramConfig(
                    q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0, max_core_grid_x=grid_cap
                )

                def run():
                    return ttnn.experimental.indexer_score_dsa(
                        q_dev, k_full, w_dev, cluster_axis=SP_AXIS, program_config=cfg, **bc
                    )

                try:
                    out = run()
                    ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
                except RuntimeError as e:
                    reason = "L1-OOM" if "beyond max L1" in str(e) else "err"
                    rows.append(f"  {kc:<3} {cols}x10  {band_count:<5} {band_count / cols:<5.1f}  -- {reason} --")
                    continue
                out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
                assert_indexer_match(out_t, ref, CHUNK, T, check_neg=True)

                m = _measure_risc(mesh, stall_group, run)
                ncores = cols * 10
                fpu = _ideal_fpu_cycles_per_core(heads, ncores) / (m["T1(mth)"] * 1000.0 * _BH_FREQ_GHZ)
                risc_only = {k: v for k, v in m.items() if k != "kernel"}
                bottleneck = max(risc_only, key=risc_only.get)
                bpc = band_count / cols
                rows.append(
                    f"  {kc:<3} {cols}x10  {band_count:<5} {bpc:<5.1f}  {m['kernel']:8.1f}  "
                    f"{m['NC(rdr)']:7.1f} {m['BR(wrt)']:7.1f} {m['T0(unp)']:7.1f} {m['T1(mth)']:7.1f} {m['T2(pck)']:7.1f}  "
                    f"{bottleneck:<10} {100.0 * fpu:5.1f}"
                )
        logger.info("\n" + "\n".join(rows) + "\n")
    finally:
        _close_ccl(mesh)


# ---- TRUE latency (profiler OFF) to expose the profiler-build inflation of FPU util ---------------
def _time_trace_us(mesh, run_fn, iters=20, warmup=5, stall_group=None):
    """Per-iter device latency via a captured trace (dispatch amortized). Profiler must be OFF so the kernels
    compile without PROFILE_KERNEL markers -> real speed. Pass stall_group for fabric/CCL ops."""
    sync = (
        lambda: ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        if stall_group
        else ttnn.synchronize_device(mesh)
    )
    for _ in range(warmup):
        run_fn()
    sync()
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(iters):
        run_fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    sync()
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    sync()
    dt = time.perf_counter() - t0
    ttnn.release_trace(mesh, tid)
    return dt / iters * 1e6


@pytest.mark.parametrize("case_id, heads", CASES, ids=CASE_IDS)
def test_qb4_indexer_true_latency(case_id, heads):
    """True device latency of the standalone block-cyclic indexer via a trace loop, with the profiler OFF so
    kernels compile at real speed (no PROFILE_KERNEL). Recomputes FPU util against this true latency -- the
    profiler-build device-duration is instrumentation-inflated (~4x), which deflates the util reported by the
    other tests. Run WITHOUT the TT_METAL_DEVICE_PROFILER env."""
    if _PROFILER_ON:
        pytest.skip("run WITHOUT the profiler env so kernels compile at real speed (no PROFILE_KERNEL markers)")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, RING), trace_region_size=90_000_000)
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK, T, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK)
        ref = _per_sp_ref_sq(q_g, k_nat, w_g, RING, HISTORY, SQ)
        shard = ttnn.ShardTensorToMesh(mesh, dim=2)
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        k_full = ttnn.from_torch(
            k_bc,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        bc = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=SQ)
        tt_tiles = T // 32
        rows = [
            f"==== indexer TRUE latency (profiler OFF, trace) [{case_id} heads={heads}]  T={T}({tt_tiles}t) ====",
            "  config                grid   latency_us   ideal_FPU_us   FPU_util%",
        ]
        for kc_mode in ("prod", "aligned"):
            k_chunk = KC_MODES[kc_mode][heads]
            kc = k_chunk // 32
            for grid_cap in (COMPUTE_COLS, 0):  # 10 cols (fused-matched) then full 11 cols
                cfg = ttnn.IndexerScoreProgramConfig(
                    q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0, max_core_grid_x=grid_cap
                )

                def run():
                    return ttnn.experimental.indexer_score_dsa(
                        q_dev, k_full, w_dev, cluster_axis=SP_AXIS, program_config=cfg, **bc
                    )

                out = run()
                ttnn.synchronize_device(mesh)
                out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
                assert_indexer_match(out_t, ref, CHUNK, T, check_neg=True)

                us = _time_trace_us(mesh, run)
                cols = min(-(-tt_tiles // kc), grid_cap if grid_cap else 11)
                ideal_us = _ideal_fpu_cycles_per_core(heads, cols * 10) / _BH_FREQ_GHZ / 1000.0
                rows.append(
                    f"  KC={kc:<3}({kc_mode:<7})     {cols}x10   {us:8.1f}     {ideal_us:8.1f}       {100.0 * ideal_us / us:5.1f}"
                )
        logger.info("\n" + "\n".join(rows) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("num_links", [2], ids=["nl2"])
@pytest.mark.parametrize("kc_mode", ["prod", "aligned"], ids=["kc_prod", "kc_aligned"])
@pytest.mark.parametrize("case_id, heads", CASES, ids=CASE_IDS)
def test_qb4_ring_vs_separate_true_latency(case_id, heads, kc_mode, num_links):
    """REAL-speed fused vs separate: same comparison as test_qb4_ring_indexer_vs_separate but latency is the
    trace-loop wall-clock with the profiler OFF (no PROFILE_KERNEL inflation), so the microseconds are the true
    device numbers you would see in deployment. Run WITHOUT the TT_METAL_DEVICE_PROFILER env."""
    if _PROFILER_ON:
        pytest.skip("run WITHOUT the profiler env so kernels compile at real speed (no PROFILE_KERNEL markers)")
    k_chunk = KC_MODES[kc_mode][heads]
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((1, RING))
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK, T, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK)
        ref = _per_sp_ref_sq(q_g, k_nat, w_g, RING, HISTORY, SQ)
        shard = ttnn.ShardTensorToMesh(mesh, dim=2)
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        k_local = ttnn.from_torch(k_bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        replicate = ttnn.ReplicateTensorToMesh(mesh)
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        k_full = ttnn.from_torch(k_bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
        ag_out_buf = ttnn.from_torch(
            torch.zeros_like(k_nat), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        bc = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=SQ)
        fused_cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0)
        sep_cfg = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0, max_core_grid_x=COMPUTE_COLS
        )

        def run_fused():
            return ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Ring,
                num_links=num_links,
                ag_sub_device_id=subdevice_id,
                program_config=fused_cfg,
                **bc,
            )

        def run_ag():
            return ttnn.experimental.ring_attention_all_gather_async(
                [k_local],
                persistent_output_buffer=[ag_out_buf],
                dim=2,
                multi_device_global_semaphore=ccl_semaphores,
                cluster_axis=SP_AXIS,
                mesh_device=mesh,
                num_links=num_links,
                memory_config=DRAM,
                topology=ttnn.Topology.Ring,
                subdevice_id=subdevice_id,
            )

        def run_indexer():
            return ttnn.experimental.indexer_score_dsa(
                q_dev, k_full, w_dev, cluster_axis=SP_AXIS, program_config=sep_cfg, **bc
            )

        # correctness before timing
        fused_out = run_fused()
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        assert_indexer_match(
            ttnn.to_torch(fused_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)), ref, CHUNK, T, check_neg=True
        )

        fused_us = _time_trace_us(mesh, run_fused, stall_group=stall_group)
        ag_us = _time_trace_us(mesh, run_ag, stall_group=stall_group)
        idx_us = _time_trace_us(mesh, run_indexer, stall_group=stall_group)

        two_op_floor = ag_us + idx_us
        ideal = max(ag_us, idx_us)
        ag_exposed = fused_us - idx_us
        ag_hidden = max(0.0, min(1.0, 1.0 - ag_exposed / ag_us)) if ag_us else 0.0
        idx_fpu = _ideal_fpu_cycles_per_core(heads, COMPUTE_COLS * 10) / _BH_FREQ_GHZ / 1000.0 / idx_us
        logger.info(
            "\n"
            f"==== QB4 REAL-speed (profiler OFF) [{case_id} h={heads} KC={k_chunk // 32}({kc_mode}) nl={num_links}] ====\n"
            f"  standalone AG            : {ag_us:8.1f} us\n"
            f"  standalone indexer       : {idx_us:8.1f} us   (FPU util {100.0 * idx_fpu:.1f}%)\n"
            f"  two-op floor (AG+idx)    : {two_op_floor:8.1f} us\n"
            f"  ideal max(AG, idx)       : {ideal:8.1f} us\n"
            f"  FUSED ring indexer       : {fused_us:8.1f} us\n"
            f"  -> AG hidden             : {100.0 * ag_hidden:6.1f} %\n"
            f"  -> fused speedup vs floor : {two_op_floor / fused_us:6.2f}x\n"
        )
    finally:
        _close_ccl(mesh)


# ---- Tracy perf target: run under `scripts/run_safe_pytest.sh --profile` (python -m tracy -r) ------
# Plain dispatch loops (NO in-process profiler, NO trace, NO asserts) so the tracy ops CSV captures a clean
# DEVICE KERNEL DURATION + CORE COUNT per op. math_util is then computed OFFLINE from that CSV the same way as
# test_indexer_score_math_util: mm_flops / (cores * duration_ns*1.35 * 2048_HiFi2) at the bottleneck device.
@pytest.mark.skipif(os.environ.get("QB4_TRACY_PERF") != "1", reason="set QB4_TRACY_PERF=1 and run under --profile")
@pytest.mark.parametrize("num_links", [2], ids=["nl2"])
@pytest.mark.parametrize("kc_mode", ["prod", "aligned"], ids=["kc_prod", "kc_aligned"])
@pytest.mark.parametrize("case_id, heads", CASES, ids=CASE_IDS)
def test_qb4_tracy_perf(case_id, heads, kc_mode, num_links):
    k_chunk = KC_MODES[kc_mode][heads]
    mesh, _ccl_semaphores, _subdevice_id, stall_group = _open_ccl((1, RING))
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK, T, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK)
        shard = ttnn.ShardTensorToMesh(mesh, dim=2)
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        k_full = ttnn.from_torch(
            k_bc,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        bc = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=SQ)
        # FULL deployed grid (no cap) so the tracy DEVICE KERNEL DURATION reproduces the deployed ~0.345/0.636 ms
        # and the perf-gate's ~70/76% math_util. (The 100-core cap is only for the fused-vs-standalone comparison.)
        sep_cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0)
        # STANDALONE indexer only (tracy logs each dispatch x device; the offline CSV parse takes the bottleneck
        # device duration and computes math_util). Fused and standalone share the op code IndexerScoreDeviceOperation,
        # so profiling them together would be ambiguous in the CSV -- the fused timing comes from the trace tests.
        for _ in range(8):
            ttnn.experimental.indexer_score_dsa(
                q_dev, k_full, w_dev, cluster_axis=SP_AXIS, program_config=sep_cfg, **bc
            )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        logger.info(f"[tracy perf {case_id} h={heads} KC={k_chunk // 32}({kc_mode}) nl={num_links}] dispatched")
    finally:
        _close_ccl(mesh)

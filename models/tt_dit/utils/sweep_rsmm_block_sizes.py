# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Sweep block sizes for the fused matmul + strided reduce-scatter op
(``ttnn.experimental.minimal_matmul_strided_reduce_scatter_async``) used by
RowParallelLinear (proj_out / ff2) under FSDP, using the device profiler for
accurate kernel timing.

This complements sweep_mm_block_sizes.py (which covers plain matmul + AGMM but
NOT the reduce-scatter-matmul path). It drives the hardware-proven nightly
helper ``run_minimal_matmul_strided_reduce_scatter_impl`` so the op setup
(semaphores, RS ping-pong, sub-device, trace, PCC verification) is reused
verbatim. One profiler subprocess runs per candidate; the orchestrator parses
the signpost-bounded ops log and records mean per-iteration device time.

NOTE: this sweeps the PLAIN fused op (no addcmul / no virtual-concat). The
block-size optimum is driven by the matmul dims + grid + chunk_width; the
addcmul epilogue and virtual-concat K read-pattern are cheap and don't move the
optimum materially. Values found here are applied to the addcmul+virtual-concat
shapes in matmul.py::fused_mmrs_configs.

Usage:
    # Orchestrator: sweep all candidates for one shape
    pytest models/tt_dit/utils/sweep_rsmm_block_sizes.py::test_rsmm_sweep \\
        -k "1152_3072_6144" -x -s

    # Worker (single candidate via env): debugging
    MM_RSMM_PARAMS='{"M":1152,"K":3072,"N":6144,"mblk":10,"kblk":8,"nblk":8,
        "sbh":1,"sbw":4,"chunk":1,"workers":null,"cgx":8,"cgy":7}' \\
    pytest models/tt_dit/utils/sweep_rsmm_block_sizes.py::test_rsmm_sweep_worker \\
        -k "1152_3072_6144" -x -s
"""

import csv
import json
import os
import sys

import pytest
from loguru import logger

import ttnn
from models.tt_dit.utils.sweep_mm_block_sizes import append_csv_row, close_mesh, open_mesh, parse_ops_log

TILE = 32

# Device config: matches wh_glx_ring_sp0tp1_fsdp (RS runs on tp_axis=1, ring of 8).
WH_CFG = {
    "mesh_shape": (4, 8),
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config_payload": 4096,
    "topology": ttnn.Topology.Ring,
    "num_links": 4,
    "num_workers_per_link": 2,
    "sp_axis": 0,
    "tp_axis": 1,
    "cluster_axis": 1,
}

# (M, K, N, cgx, cgy) — the two untuned RS+MM shapes on the fused base.
# Compute grid 8x7 (RS reserves 2 rows of the full 8x9); RS offset = (0, cgy).
SHAPES = [
    (1152, 3072, 6144, 8, 7),  # SNG proj_out (xc-merged 1024+128), hot (188 hits/run)
    (128, 2304, 6144, 8, 7),  # prompt proj_out
]
SHAPE_IDS = [f"{M}_{K}_{N}_{cgx}x{cgy}" for M, K, N, cgx, cgy in SHAPES]

NUM_ITERS = 10  # trace executions per candidate (mean per-iter device time recorded)
CSV_FILE = "sweep_results_rsmm.csv"
CSV_COLUMNS = [
    "M",
    "K",
    "N",
    "core_grid",
    "M_block",
    "K_block",
    "N_block",
    "subblock_h",
    "subblock_w",
    "chunk_width",
    "num_workers",
    "device_time_ns",
    "status",
]


def _divisors(n, lo=2):
    return sorted(d for d in range(lo, n + 1) if n % d == 0)


def pick_subblock(m_block, n_block, max_vol=4):
    """fp32-dest prefers 2x2; else largest (h,w) with h|m, w|n, h*w<=max_vol."""
    if m_block % 2 == 0 and n_block % 2 == 0:
        return (2, 2)
    best, best_prod = (1, 1), 1
    for h in range(1, min(m_block, max_vol) + 1):
        if m_block % h:
            continue
        for w in range(1, min(n_block, max_vol) + 1):
            if n_block % w:
                continue
            if h * w <= max_vol and h * w > best_prod:
                best, best_prod = (h, w), h * w
    return best


def gen_candidates(M, K, N, cgx, cgy):
    """Curated (m_blk, k_blk, n_blk, sb_h, sb_w, chunk_width, num_workers) combos (tile units).

    Centered on the model's known-good blockings (M_block~10-12, K_block~8,
    N_block=8, chunk=1) with a modest neighborhood so the subprocess-per-candidate
    sweep stays bounded (~tens of combos).
    """
    M_tiles = max(1, -(-M // TILE))
    K_tiles = K // TILE

    # Trim to the high-impact axes (M_block, K_block) and fix the rest at the
    # model's known-good values, so the subprocess-per-candidate sweep (~45s
    # each) stays ~24/shape rather than hundreds. Widen later if results are flat.
    # M_block: neighborhood capped at M_tiles (one block can cover all rows).
    m_cands = sorted({b for b in (4, 6, 8, 10, 12) if b <= max(4, M_tiles)})
    # K_block: clean divisors of K_tiles near the model's value (8).
    k_all = _divisors(K_tiles, lo=4)
    k_cands = sorted({k for k in k_all if k in (8, 12, 16)} or set(k_all[:3]))
    # N_block fixed at the model's value; chunk on the two common settings;
    # num_workers auto (matches the model's None default).
    n_cands = [8]
    chunk_cands = [1, 2]
    worker_cands = [None]  # None => kernel auto-computes from RS-zone capacity

    combos = []
    for m in m_cands:
        for k in k_cands:
            for n in n_cands:
                sbh, sbw = pick_subblock(m, n)
                for chunk in chunk_cands:
                    for w in worker_cands:
                        combos.append((m, k, n, sbh, sbw, chunk, w))
    return combos


# ============================================================================
# WORKER — one candidate, profiled in subprocess
# ============================================================================


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_rsmm_sweep_worker(shape):
    """Run ONE candidate (from MM_RSMM_PARAMS env) of the fused RS+MM op."""
    from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
        run_minimal_matmul_strided_reduce_scatter_impl,
    )

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    p = json.loads(os.environ["MM_RSMM_PARAMS"])
    cfg = WH_CFG
    parent_mesh, submesh = open_mesh(cfg, trace_region_size=4194304)
    try:
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        run_minimal_matmul_strided_reduce_scatter_impl(
            submesh,
            M=p["M"],
            K=p["K"],
            N=p["N"],
            dim=3,
            num_links=cfg["num_links"],
            input_dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mem_config_input=mem,
            mem_config_mm=mem,
            mem_config_rs=mem,
            topology=cfg["topology"],
            mm_block_m=p["mblk"] * TILE,
            mm_block_k=p["kblk"] * TILE,
            mm_block_n=p["nblk"] * TILE,
            subblock_h=p["sbh"],
            subblock_w=p["sbw"],
            mm_core_grid=ttnn.CoreCoord(p["cgx"], p["cgy"]),
            num_iters=NUM_ITERS,
            enable_trace=True,
            cluster_axis=cfg["cluster_axis"],
            num_workers_per_link=p["workers"],
            num_buffers_per_channel=None,
            chunk_width_in_mm_blocks=p["chunk"],
            rs_mode="fused",
            rs_core_grid_offset=ttnn.CoreCoord(0, p["cgy"]),
            allowed_pcc=0.99,
        )
    finally:
        close_mesh(parent_mesh)


# ============================================================================
# ORCHESTRATOR — loops candidates, one profiler subprocess each
# ============================================================================


@pytest.mark.timeout(36000)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance sweep - skip on CI")
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_rsmm_sweep(shape):
    from tracy.process_model_log import run_device_profiler

    M, K, N, cgx, cgy = shape
    shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}"
    core_grid_str = f"{cgx}x{cgy}"
    combos = gen_candidates(M, K, N, cgx, cgy)

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
    print(f"\n=== RS+MM sweep {shape_id}: {len(combos)} candidates ===", flush=True)

    os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    saved_level = os.environ.get("TT_LOGGER_LEVEL")
    os.environ["TT_LOGGER_LEVEL"] = "Error"

    results = []
    try:
        for idx, (m, k, n, sbh, sbw, chunk, workers) in enumerate(combos):
            params = {
                "M": M,
                "K": K,
                "N": N,
                "mblk": m,
                "kblk": k,
                "nblk": n,
                "sbh": sbh,
                "sbw": sbw,
                "chunk": chunk,
                "workers": workers,
                "cgx": cgx,
                "cgy": cgy,
            }
            os.environ["MM_RSMM_PARAMS"] = json.dumps(params)
            subdir = f"rsmm_sweep_{shape_id}_{idx}"
            command = (
                f"pytest models/tt_dit/utils/sweep_rsmm_block_sizes.py"
                f"::test_rsmm_sweep_worker[{shape_id}] -x -s --timeout 1800"
            )
            duration_ns, status = -1.0, "FAIL"
            try:
                run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
                durs = parse_ops_log(subdir)
                if durs:
                    # Mean per-iteration device time = total device time / NUM_ITERS.
                    duration_ns = sum(durs) / NUM_ITERS
                    status = "OK"
            except Exception as e:
                logger.error(f"candidate {idx} failed: {str(e)[:160]}")

            append_csv_row(
                CSV_FILE,
                [
                    M,
                    K,
                    N,
                    core_grid_str,
                    m,
                    k,
                    n,
                    sbh,
                    sbw,
                    chunk,
                    "auto" if workers is None else workers,
                    f"{duration_ns:.0f}",
                    status,
                ],
            )
            if status == "OK":
                results.append((duration_ns, m, k, n, sbh, sbw, chunk, workers))
                print(
                    f"  [{idx+1}/{len(combos)}] M={m} K={k} N={n} sb=({sbh},{sbw}) "
                    f"chunk={chunk} workers={workers} -> {duration_ns:.0f} ns",
                    flush=True,
                )
            else:
                print(f"  [{idx+1}/{len(combos)}] {params} -> {status}", flush=True)
    finally:
        os.environ.pop("MM_RSMM_PARAMS", None)
        os.environ.pop("TT_METAL_PROFILER_MID_RUN_DUMP", None)
        if saved_level is None:
            os.environ.pop("TT_LOGGER_LEVEL", None)
        else:
            os.environ["TT_LOGGER_LEVEL"] = saved_level

    if not results:
        print("  WARN: no successful candidates", flush=True)
        return
    results.sort()
    print(f"\n  === BEST for {shape_id} ===", flush=True)
    for rank, (d, m, k, n, sbh, sbw, chunk, workers) in enumerate(results[:5], 1):
        print(
            f"    #{rank}: M_block={m} K_block={k} N_block={n} sb=({sbh},{sbw}) "
            f"chunk={chunk} workers={workers} -> {d:.0f} ns",
            flush=True,
        )

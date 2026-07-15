# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Sweep block sizes for minimal_matmul and all_gather_minimal_matmul_async
using device profiler for accurate kernel timing.

Architecture:
  - Worker test (test_mm_sweep_worker): for a (device_config, shape) pair,
    opens the mesh once and sweeps every (M_block, K_block, N_block, sb_h, sb_w)
    candidate. Uses TT_METAL_PROFILER_MID_RUN_DUMP=1 + periodic
    ttnn.ReadDeviceProfiler calls to flush the profiler buffer between combos
    without losing data.
  - Orchestrator test (test_mm_sweep): thin wrapper that invokes the worker
    via run_device_profiler and parses one ops log into the results CSV.

Usage:
    # Orchestrator: sweep one shape
    pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \\
        -k "9472_3456_5120_11x10_mm_plain-bh_4x8" -x -s

    # Worker: run directly (useful for debugging, no profiling)
    pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep_worker \\
        -k "9472_3456_5120_11x10_mm_plain-bh_4x8" -x -s

    # Standalone script
    python models/tt_dit/utils/sweep_mm_block_sizes.py \\
        --device-config bh_4x8 --shape 9472,3456,5120
"""

import argparse
import csv
import json
import os
import sys

import pytest
import torch
from loguru import logger
from tqdm import tqdm
from tracy import signpost

import ttnn

# ============================================================================
# DEVICE CONFIGURATIONS
# ============================================================================
# To add a new config: add an entry here with the required fields.
# The worker and orchestrator will pick it up automatically.

DEVICE_CONFIGS = {
    "bh_4x8": {
        "mesh_shape": (4, 8),
        "fabric_config": "FABRIC_1D_RING",
        "fabric_router_config_payload": None,  # use default (4352) to match model
        "topology": "Ring",
        "num_links": 2,
        "num_workers_per_link": 6,
        "sp_axis": 1,
        "tp_axis": 0,
        "cluster_axis": 0,
    },
    # Same galaxy, but the all-gather ring runs along the size-8 mesh axis
    # (ring_size=8) instead of the size-4 axis. Used for FLUX shapes.
    "bh_4x8_ring8": {
        "mesh_shape": (4, 8),
        "fabric_config": "FABRIC_1D_RING",
        "fabric_router_config_payload": None,
        "topology": "Ring",
        "num_links": 2,
        "num_workers_per_link": 6,
        "sp_axis": 0,
        "tp_axis": 1,
        "cluster_axis": 1,
    },
    # WH Galaxy 4x8, 4-device cluster along axis 0 (rows). Matches wh4x8links4_*
    # configs in tests/.../test_all_gather_minimal_matmul_async.py.
    "wh_4x8_ring": {
        "mesh_shape": (4, 8),
        "fabric_config": "FABRIC_1D_RING",
        "fabric_router_config_payload": 4096,
        "topology": "Ring",
        "num_links": 4,
        "num_workers_per_link": 2,
        "sp_axis": 1,
        "tp_axis": 0,
        "cluster_axis": 0,
    },
    "wh_4x8_linear": {
        "mesh_shape": (4, 8),
        "fabric_config": "FABRIC_1D",
        "fabric_router_config_payload": 4096,
        "topology": "Linear",
        "num_links": 4,
        "num_workers_per_link": 2,
        "sp_axis": 1,
        "tp_axis": 0,
        "cluster_axis": 0,
    },
}

DEFAULT_DEVICE_CONFIG = "bh_4x8"


def resolve_config(name):
    """Resolve a device config name to its dict, with ttnn enums."""
    cfg = DEVICE_CONFIGS[name]
    return {
        "mesh_shape": cfg["mesh_shape"],
        "fabric_config": getattr(ttnn.FabricConfig, cfg["fabric_config"]),
        "fabric_router_config_payload": cfg["fabric_router_config_payload"],
        "topology": getattr(ttnn.Topology, cfg["topology"]),
        "num_links": cfg["num_links"],
        "num_workers_per_link": cfg["num_workers_per_link"],
        "sp_axis": cfg["sp_axis"],
        "tp_axis": cfg["tp_axis"],
        "cluster_axis": cfg["cluster_axis"],
    }


# ============================================================================
# SHAPE TABLE
# ============================================================================

# (M, K, N, core_grid_x, core_grid_y, is_agmm, use_case)
# M, K, N are the matmul dimensions as seen by the kernel (per-device).
# Core grid matches what the model passes to get_matmul_config at runtime.
# For is_agmm=False: calls ttnn.experimental.minimal_matmul
# For is_agmm=True: calls ttnn.experimental.all_gather_minimal_matmul_async
#
# use_case controls per-shape configuration to match model behavior:
#   "plain"     - no fused activation or addcmul
#   "ff2"       - RowParallelLinear (no fused activation, label only)
#   "qkv"       - attention QKV projection (chunks=3, math_approx_mode=True)
#   "to_out"    - attention to_out projection (addcmul fused, math_approx_mode=True)
#   "ff1_gelu"  - FFN first linear (fused GELU activation)
#   "cross_attn_kv" - cross-attention to_kv (minimal_matmul_split, chunks=2, math_approx_mode=True)
# Example shapes from Wan2.2 configs in matmul.py — one per use case.
# Add model-specific shapes via register_matmul_configs() and extend this list as needed.
SHAPES = [
    # plain: basic matmul, no fused activation or addcmul (Wan2.2 720p DiT, 11x10 grid)
    (9472, 3456, 5120, 11, 10, False, "plain"),
    # ff2: RowParallelLinear — same kernel path as plain (Wan2.2 480p DiT, 11x10 grid)
    (2368, 3456, 5120, 11, 10, False, "ff2"),
    # qkv: attention QKV projection, chunks=3, approx math (Wan2.2 720p AGMM, 12x9 grid)
    (9472, 5120, 3840, 12, 9, True, "qkv"),
    # to_out: attention output with fused addcmul, approx math (Wan2.2 720p AGMM, 12x9 grid)
    (9472, 5120, 1280, 12, 9, True, "to_out"),
    # ff1_gelu: FFN first linear with fused GELU activation (Wan2.2 720p AGMM, 12x9 grid)
    (9472, 5120, 3456, 12, 9, True, "ff1_gelu"),
    # cross_attn_kv: cross-attention KV via minimal_matmul_split, chunks=2 (11x10 grid)
    (128, 5120, 2560, 11, 10, False, "cross_attn_kv"),
    # WH AGMM Wan2.2 shapes (8x8 grid), K-fractured across 4 devices.
    # All have bias (always allocated/passed by _build_op_runner).
    # N=3456 has fused GELU (exact, non-approx) to match the test config.
    (3072, 5120, 3840, 8, 8, True, "plain"),
    (3072, 5120, 1280, 8, 8, True, "plain"),
    (3072, 5120, 3456, 8, 8, True, "plain_gelu"),
]

SHAPE_IDS = [f"{M}_{K}_{N}_{cgx}x{cgy}_{'agmm' if agmm else 'mm'}_{uc}" for M, K, N, cgx, cgy, agmm, uc in SHAPES]

# Per-use-case configuration overrides applied in the worker.
USE_CASE_CONFIGS = {
    "plain": {},
    "ff2": {},
    "qkv": {
        "chunks": 3,
        "math_approx_mode": True,
    },
    "to_out": {
        "scalar": 1.0,
        "use_addcmul": True,
        "math_approx_mode": True,
    },
    "ff1_gelu": {
        "fused_activation": (ttnn.UnaryOpType.GELU, True),
    },
    # Like "plain" but with exact (non-approx) GELU fused — matches the
    # activation="gelu" config in test_all_gather_minimal_matmul_async.py.
    "plain_gelu": {
        "fused_activation": (ttnn.UnaryOpType.GELU, False),
    },
    "cross_attn_kv": {
        "chunks": 2,
        "math_approx_mode": True,
        "use_matmul_split": True,
    },
}

# Whether the sweep uses fp32 dest accumulator. With fp32 dest, the DEST tile
# capacity is halved (4 tiles instead of 8), so only subblocks with h*w == 4
# match peak compute throughput — and among those, 2x2 is strictly preferred
# over 4x1 / 1x4 (better tile reuse in the math LLK). So when fp32 dest is on
# we skip the subblock sweep entirely and always pick 2x2 (when divisible).
FP32_DEST_ACC_EN = True

# Block-size candidate methodology:
# - M/N block:  even sizes in [MN_BLOCK_MIN, MN_BLOCK_MAX]  union  divisors of
#               per-core tile count in that same range. Floor at 2 (1 is usually
#               dispatch-overhead-bound); cap at 16 because larger blocks reduce
#               pipelining. Divisors are added to give a "1 block per core"
#               option even when it's odd (e.g. 5, 15).
# - K block:    divisors of K_per_device (AGMM) / K_tiles (non-AGMM) at >=
#               K_BLOCK_MIN. K_block MUST divide K_per_device for AGMM (the ring
#               all-gather delivers K_per_device tiles in K_block-sized chunks);
#               non-divisor candidates would leave a partial chunk on the last
#               ring iteration. No upper cap — large divisors don't add padding.
MN_BLOCK_MIN, MN_BLOCK_MAX = 2, 16
K_BLOCK_MIN = 2

# L1 budget for pre-filtering block combos (KB).
# BH L1 usable ~1464 KB; conservative threshold accounts for kernel/firmware overhead.
# CB allocation: 2*M*K + 2*K*N + 2*M*N tiles (all double-buffered, bf16 = 2KB/tile)
#              + M*N tiles intermediate (single-buffered, f32 = 4KB/tile)
#              + N tiles bias (single-buffered, bf16 = 2KB/tile)
# to_out adds: M*N tiles ternary_a (bf16) + N tiles ternary_c (bf16)
L1_BUDGET_KB = 1400

CSV_FILE = "sweep_results_mm.csv"
CSV_COLUMNS = [
    "device_config",
    "op_type",
    "use_case",
    "M",
    "K",
    "N",
    "core_grid",
    "M_block",
    "K_block",
    "N_block",
    "subblock_h",
    "subblock_w",
    "device_kernel_duration_ns",
    "status",
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_mn_block_candidates(per_core_tiles):
    """Even sizes in [MN_BLOCK_MIN, MN_BLOCK_MAX] union divisors of per-core tiles in that range.

    Sized to a single core's M or N work, so that "1 block per core" appears as a
    candidate even when it's odd (e.g. 5, 15). Caps at MN_BLOCK_MAX because larger
    blocks reduce pipelining; floors at MN_BLOCK_MIN because tiny blocks are
    dispatch-bound.
    """
    evens = set(range(MN_BLOCK_MIN, MN_BLOCK_MAX + 1, 2))
    divisors = set(d for d in range(MN_BLOCK_MIN, MN_BLOCK_MAX + 1) if per_core_tiles % d == 0)
    return sorted(evens | divisors)


def get_k_block_candidates(K_per_device):
    """Divisors of K_per_device, capped at K_BLOCK_MIN floor.

    HARD constraint for AGMM: K_block must evenly divide K_per_device. The ring
    all-gather delivers K_per_device tiles per device per ring iteration, in
    K_block-sized chunks — any K_block that doesn't divide K_per_device leaves
    a partial chunk on the last iteration, which the algorithm doesn't support.

    So we restrict to divisors only. K_BLOCK_MIN excludes tiny sizes that are
    dispatch-overhead-bound; there's no upper cap because dividing K cleanly
    never adds padding even at larger block sizes.
    """
    cands = [d for d in range(K_BLOCK_MIN, K_per_device + 1) if K_per_device % d == 0]
    # Fall back to the full-K single-chunk block when K_per_device < K_BLOCK_MIN
    # (e.g. a single K-tile after the ring split, K_per_device=1): K_block =
    # K_per_device trivially divides and is the only valid option.
    return cands if cands else [K_per_device]


# Isolated all-gather (op_type="ag") knob sweep. num_links stays at the device
# config value; the AG "combo" varies the two knobs that matter for the minimal
# all-gather — encoded as a 5-tuple (num_workers_per_link, chunks_per_sync, 0,0,0)
# so it flows through the same warmup/measure/record machinery as block combos.
AG_WORKERS_PER_LINK = [1, 2, 4, 8]
AG_CHUNK_FRACTIONS = [1, 2, 4, 8]  # chunks_per_sync = max(1, MAX // frac)


def get_ag_max_chunks_per_sync(gathered_elems, ring_size, num_links):
    """Upper bound on chunks_per_sync (mirrors tests/.../get_max_chunks_per_sync)."""
    packet_elems = 2048
    return max(1, (gathered_elems // packet_elems) // (ring_size * num_links))


def generate_ag_combos(gathered_elems, ring_size, num_links):
    """(num_workers_per_link, chunks_per_sync, 0, 0, 0) combos for the AG knob sweep."""
    mx = get_ag_max_chunks_per_sync(gathered_elems, ring_size, num_links)
    chunks = sorted({max(1, mx // f) for f in AG_CHUNK_FRACTIONS})
    return [(nw, cps, 0, 0, 0) for nw in AG_WORKERS_PER_LINK for cps in chunks]


def get_per_core_dims(shape, cluster_size):
    """Compute (M_per_core, K_per_device, N_per_core) for a shape.

    Assumes force_transpose=True (the only mode the sweep currently runs):
    in0 parallelizes M across grid_x cores, in1 parallelizes N across grid_y cores.
    """
    M, K, N, cgx, cgy, is_agmm, _ = shape
    M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
    M_per_core = -(-M_tiles // cgx)  # ceiling
    N_per_core = -(-N_tiles // cgy)
    K_per_device = K_tiles // cluster_size if is_agmm else K_tiles
    return M_per_core, K_per_device, N_per_core


def estimate_l1_kb(m_blk, k_blk, n_blk, use_case="plain"):
    """Estimate L1 circular buffer footprint in KB for a given block config.

    Based on minimal_matmul_program_factory.cpp CB allocation:
      c_0 (in0):   2 * M * K tiles  (double-buffered, bf16 = 2 KB/tile)
      c_1 (in1):   2 * K * N tiles  (double-buffered, bf16)
      c_2 (out):   2 * M * N tiles  (double-buffered, bf16)
      c_3 (interm): M * N tiles     (single-buffered, f32 = 4 KB/tile)
      c_4 (bias):   N tiles         (single-buffered, bf16)
    to_out adds:
      c_5 (ternary_a): M * N tiles  (single-buffered, bf16)
      c_6 (ternary_c): N tiles      (single-buffered, bf16)
    """
    bf16_kb = 2  # KB per bf16 tile (32x32x2 bytes)
    f32_kb = 4  # KB per f32 tile (32x32x4 bytes)

    kb = (
        2 * m_blk * k_blk * bf16_kb  # c_0: in0
        + 2 * k_blk * n_blk * bf16_kb  # c_1: in1
        + 2 * m_blk * n_blk * bf16_kb  # c_2: out
        + m_blk * n_blk * f32_kb  # c_3: intermediate (f32 accumulator)
        + n_blk * bf16_kb  # c_4: bias
    )

    uc_cfg = USE_CASE_CONFIGS.get(use_case, {})
    if uc_cfg.get("use_addcmul", False):
        kb += m_blk * n_blk * bf16_kb  # c_5: ternary_a
        kb += n_blk * bf16_kb  # c_6: ternary_c

    return kb


def pick_subblock(m_block, n_block, max_dest_volume=4):
    """Pick best valid (sb_h, sb_w) where sb_h|m_block, sb_w|n_block, sb_h*sb_w <= max_dest_volume.

    For fp32 dest, (2, 2) is strictly preferred among same-product candidates
    (better math LLK tile reuse than 4x1 / 1x4), so check it first.
    """
    if FP32_DEST_ACC_EN and m_block % 2 == 0 and n_block % 2 == 0 and 4 <= max_dest_volume:
        return (2, 2)
    best = (1, 1)
    best_product = 1
    for h in range(1, min(m_block, max_dest_volume) + 1):
        if m_block % h != 0:
            continue
        for w in range(1, min(n_block, max_dest_volume) + 1):
            if n_block % w != 0:
                continue
            if h * w <= max_dest_volume and h * w > best_product:
                best = (h, w)
                best_product = h * w
    return best


def generate_kn_combos(K_per_device, N_per_core, m_block=1, use_case="plain"):
    """Generate (K_block, N_block) combos filtered by L1 budget."""
    k_candidates = get_k_block_candidates(K_per_device)
    n_candidates = get_mn_block_candidates(N_per_core)
    combos = []
    for k in k_candidates:
        for n in n_candidates:
            if estimate_l1_kb(m_block, k, n, use_case) <= L1_BUDGET_KB:
                combos.append((k, n))
    return combos


def compute_tile_counts(M, K, N):
    """Compute tile counts for the matmul dimensions (all per-device)."""
    M_tiles = max(1, -(-M // 32))  # ceiling division
    K_tiles = K // 32
    N_tiles = N // 32
    return M_tiles, K_tiles, N_tiles


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def open_mesh(cfg, trace_region_size=None):
    """Open the parent mesh and create a cluster-axis submesh.

    Returns (parent_mesh, cluster_submesh). The submesh is sized 1xN (or Nx1)
    along cluster_axis so the op runs on a single ring rather than replicating
    compute across the non-cluster axis. Workers should use the submesh; pass
    the parent to close_mesh() for cleanup.
    """
    fabric_kwargs = [
        cfg["fabric_config"],
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    ]
    if cfg["fabric_router_config_payload"] is not None:
        fabric_kwargs.append(create_fabric_router_config(cfg["fabric_router_config_payload"]))
    ttnn.set_fabric_config(*fabric_kwargs)

    rows, cols = cfg["mesh_shape"]
    device_kwargs = {}
    if trace_region_size is not None:
        device_kwargs["trace_region_size"] = trace_region_size
    parent_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), **device_kwargs)

    cluster_axis = cfg["cluster_axis"]
    submesh_shape = [1, 1]
    submesh_shape[cluster_axis] = cfg["mesh_shape"][cluster_axis]
    cluster_submesh = parent_mesh.create_submesh(ttnn.MeshShape(tuple(submesh_shape)))

    return parent_mesh, cluster_submesh


def close_mesh(parent_mesh):
    """Close submeshes then the parent mesh, and reset fabric."""
    for submesh in parent_mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(parent_mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def write_csv_header(csv_path):
    """Write CSV header if file doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def append_csv_row(csv_path, row):
    """Append a single result row to CSV."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def parse_ops_log(subdir, expected_ops=None):
    """Parse device profiler ops log between start/stop signposts.

    Returns list of per-op mean device kernel durations (ns), one per op dispatch.
    Groups by GLOBAL CALL COUNT to handle multi-device rows.
    Uses mean to match tt-perf-report behavior for collective ops (AllGather*).

    With trace mode, each device gets its own GLOBAL CALL COUNT, so grouping by
    GLOBAL CALL COUNT produces N*num_devices entries instead of N. When
    expected_ops is provided and len(results) is a multiple of it, results are
    chunked into groups and averaged to collapse per-device rows.
    """
    import numpy as np
    import pandas as pd
    from tracy.process_model_log import get_latest_ops_log_filename

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    # Filter between start/stop signposts
    signpost_rows = df[df["OP TYPE"] == "signpost"]
    start_markers = signpost_rows[signpost_rows["OP CODE"] == "start"]
    stop_markers = signpost_rows[signpost_rows["OP CODE"] == "stop"]

    if start_markers.empty or stop_markers.empty:
        logger.warning(f"No start/stop signposts found in {filename}")
        return []

    start_idx = start_markers.index[0]
    stop_idx = stop_markers.index[0]
    df = df.iloc[start_idx + 1 : stop_idx]

    # Filter to rows with valid device kernel duration
    df = df[df["OP TYPE"] != "signpost"]
    df = df[df["DEVICE KERNEL DURATION [ns]"] != "-"]
    if df.empty:
        return []

    df["DEVICE KERNEL DURATION [ns]"] = df["DEVICE KERNEL DURATION [ns]"].astype(float)

    # Group by GLOBAL CALL COUNT to collapse multi-device rows into one per op
    if "GLOBAL CALL COUNT" in df.columns:
        per_op = df.groupby("GLOBAL CALL COUNT", sort=False)["DEVICE KERNEL DURATION [ns]"].mean()
        durations = per_op.values.tolist()
    else:
        durations = df["DEVICE KERNEL DURATION [ns]"].values.tolist()

    # Trace mode: each device gets its own GLOBAL CALL COUNT, producing
    # expected_ops * num_devices entries.  Collapse by chunking and averaging.
    if expected_ops is not None and len(durations) > expected_ops and expected_ops > 0:
        if len(durations) % expected_ops == 0:
            chunk_size = len(durations) // expected_ops
            durations = np.array(durations).reshape(expected_ops, chunk_size).mean(axis=1).tolist()

    return durations


# ============================================================================
# SHARED WORKER HELPERS
# ============================================================================


def _build_op_runner(
    cfg, mesh_device, M, K, N, dtype, is_agmm, uc_cfg, core_grid, math_fidelity=ttnn.MathFidelity.HiFi2, op="agmm"
):
    """Allocate tensors + return a run_op(*combo, sync=True) closure.

    AGMM path: sharded input, dummy bias/addcmul (when use_addcmul), CCL semaphores,
    persistent output buffer.
    Non-AGMM path: replicated input/weight/bias; minimal_matmul or minimal_matmul_split.
    AG path ("ag"): sharded input + CCL semaphores; run_op(num_workers_per_link,
    chunks_per_sync, ...) runs the isolated all_gather_async that AGMM fuses.
    """
    mesh_shape = tuple(mesh_device.shape)

    # ---- Isolated all-gather (op="ag") ----------------------------------------
    # Reproduce exactly the activation gather AGMM performs: input (full_M, K)
    # sharded [sp_axis -> M, tp_axis -> K], all-gathered along cluster_axis so
    # each device reconstructs the full K. Sweeps AG knobs, not block sizes.
    if op == "ag":
        sp_axis, tp_axis = cfg["sp_axis"], cfg["tp_axis"]
        # open_mesh runs the op on a cluster-axis submesh (the non-cluster/sp axis
        # is collapsed to size 1), so full_M == M: one ring holds ONE sp position's
        # M rows. Using cfg["mesh_shape"][sp_axis] here (the full mesh's sp size)
        # over-sized M by that factor -- the tensor was built with M*sp rows but
        # sharded on a size-1 sp axis, so every device processed sp x too much
        # data (the isolated-AG time was inflated by exactly that factor).
        full_M = M * mesh_shape[sp_axis]
        tt_input = ttnn.from_torch(
            torch.randn((full_M, K), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=[sp_axis, tp_axis]),
        )
        full_grid = mesh_device.compute_with_storage_grid_size()
        ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))}
        )
        ag_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_cores, 0) for _ in range(2)]

        # Match the tuned all_gather_async setup (run_all_gather_impl in
        # test_minimal_all_gather_async.py, the same path the fused AGMM op uses):
        # set the worker sub-device stall group and pass subdevice_id, a
        # persistent DRAM output buffer, memory_config, and num_buffers_per_channel.
        # Bare all_gather_async (no persistent buffer / no stall group) falls back
        # to a per-call allocate+sync path (~710us vs the ~75us fabric roofline).
        worker_sub_device_id = ttnn.SubDeviceId(0)
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])

        # all_gather_async validates the persistent buffer against the tensor's
        # GLOBAL logical shape, so it must be (full_M, K) — not the per-device
        # (M, K). The gather runs along cluster_axis (reconstructing K), so M
        # stays sharded on sp_axis while K is replicated across the ring:
        # dims=[None, None] with [sp_axis]=0 (same mapping the addcmul path uses).
        # Per device this still lands (full_M/sp_size, K) = (M, K) in DRAM.
        out_dims = [None, None]
        out_dims[sp_axis] = 0
        persistent_output_buffer = ttnn.from_torch(
            torch.empty((full_M, K), dtype=torch.float32),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=out_dims),
        )

        def run_op(num_workers_per_link, chunks_per_sync, _n=0, _sb_h=0, _sb_w=0, sync=True):
            ttnn.experimental.all_gather_async(
                tt_input,
                persistent_output_buffer=persistent_output_buffer,
                dim=1,  # gather along K (tt_input is (full_M, K))
                cluster_axis=cfg["cluster_axis"],
                multi_device_global_semaphore=ag_semaphores,
                num_links=cfg["num_links"],
                topology=cfg["topology"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_workers_per_link=int(num_workers_per_link),
                chunks_per_sync=int(chunks_per_sync),
                num_buffers_per_channel=48,
                subdevice_id=worker_sub_device_id,
            )
            if sync:
                ttnn.synchronize_device(mesh_device)

        return run_op

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=uc_cfg.get("math_approx_mode", False),
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    fused_activation = uc_cfg.get("fused_activation", None)
    chunks = uc_cfg.get("chunks", 1)
    scalar = uc_cfg.get("scalar", None)

    def _matmul_config(m_blk, k_blk, n_blk, sb_h, sb_w):
        return ttnn.MinimalMatmulConfig(
            M_block_size=m_blk,
            K_block_size=k_blk,
            N_block_size=n_blk,
            subblock_h=sb_h,
            subblock_w=sb_w,
            compute_with_storage_grid_size=core_grid,
        )

    if is_agmm:
        sp_axis = cfg["sp_axis"]
        tp_axis = cfg["tp_axis"]
        sp_size = cfg["mesh_shape"][sp_axis]
        full_M = M * sp_size

        tt_input = ttnn.from_torch(
            torch.randn((full_M, K), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=[sp_axis, tp_axis]),
        )
        tt_weight = ttnn.from_torch(
            torch.randn((K, N), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_bias = ttnn.from_torch(
            torch.randn((1, N), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )

        # CCL infrastructure (matches model's CCLManager)
        full_grid = mesh_device.compute_with_storage_grid_size()
        ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))}
        )
        ccl_semaphore_handles = [
            ttnn.create_global_semaphore(mesh_device, ccl_cores, 0),
            ttnn.create_global_semaphore(mesh_device, ccl_cores, 0),
        ]
        persistent_output_buffer = ttnn.from_torch(
            torch.empty((M, K), dtype=torch.float32),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
        )

        addcmul_tensor1 = None
        addcmul_tensor2 = None
        if uc_cfg.get("use_addcmul", False):
            # The addcmul (fused ternary) operands are added element-wise onto the
            # AGMM output, so they must be laid out identically to it: global shape
            # (full_M, N) with M (tensor dim 0) sharded across sp_axis and N
            # replicated — matching tt_input's M-sharding. ShardTensor2dMesh dims
            # are indexed by MESH axis -> tensor dim (same convention tt_input uses),
            # so we set dims[sp_axis]=0 and leave the other mesh axis unsharded.
            # Using raw (M, N) replicated fails the op's
            # "fused_ternary_input_a shape must match output" check.
            addcmul_dims = [None, None]
            addcmul_dims[sp_axis] = 0
            addcmul_tensor1 = ttnn.from_torch(
                torch.randn((full_M, N), dtype=torch.float32),
                dtype=dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=addcmul_dims),
            )
            addcmul_tensor2 = ttnn.from_torch(
                torch.randn((full_M, N), dtype=torch.float32),
                dtype=dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=addcmul_dims),
            )

        def run_op(m_blk, k_blk, n_blk, sb_h, sb_w, sync=True):
            ttnn.experimental.all_gather_minimal_matmul_async(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=fused_activation,
                compute_kernel_config=compute_config,
                config=_matmul_config(m_blk, k_blk, n_blk, sb_h, sb_w),
                persistent_output_buffer=persistent_output_buffer,
                multi_device_global_semaphore=ccl_semaphore_handles,
                num_links=cfg["num_links"],
                topology=cfg["topology"],
                cluster_axis=cfg["cluster_axis"],
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=cfg["num_workers_per_link"],
                num_buffers_per_channel=48,
                scalar=scalar,
                addcmul_input_tensor1=addcmul_tensor1,
                addcmul_input_tensor2=addcmul_tensor2,
                chunks=chunks,
            )
            if sync:
                ttnn.synchronize_device(mesh_device)

        return run_op

    # Non-AGMM
    use_matmul_split = uc_cfg.get("use_matmul_split", False)
    tt_input = ttnn.from_torch(
        torch.randn((M, K), dtype=torch.float32),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_weight = ttnn.from_torch(
        torch.randn((K, N), dtype=torch.float32),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_bias = ttnn.from_torch(
        torch.randn((1, N), dtype=torch.float32),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op(m_blk, k_blk, n_blk, sb_h, sb_w, sync=True):
        cfg_obj = _matmul_config(m_blk, k_blk, n_blk, sb_h, sb_w)
        if use_matmul_split:
            ttnn.experimental.minimal_matmul_split(
                tt_input,
                tt_weight,
                chunks=chunks,
                dim=-1,
                bias_tensor=tt_bias,
                fused_activation=fused_activation,
                compute_kernel_config=compute_config,
                config=cfg_obj,
            )
        else:
            ttnn.experimental.minimal_matmul(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=tt_bias,
                config=cfg_obj,
                fused_activation=fused_activation,
                compute_kernel_config=compute_config,
            )
        if sync:
            ttnn.synchronize_device(mesh_device)

    return run_op


PROFILER_DUMP_EVERY = 10  # call ReadDeviceProfiler every N combos to avoid buffer overflow


def _execute_sweep(mesh_device, run_op, combos):
    """Warmup (compile + filter OOMs) + trace-based measurement, single mesh open.

    combos: list of (m_blk, k_blk, n_blk, sb_h, sb_w) tuples.

    Capture/execute/release one trace at a time so peak trace memory is bounded.
    Periodically calls ttnn.ReadDeviceProfiler to flush the device profiler
    buffer (requires TT_METAL_PROFILER_MID_RUN_DUMP=1 in the subprocess env);
    without it the buffer overflows around ~50 AGMM ops and timing data is lost.

    Returns (valid_combos, skipped_count).
    """
    # Warmup: compile programs, skip OOM silently (count and report at end).
    valid_combos = []
    skipped = 0
    with tqdm(total=len(combos), desc="Warmup", unit="combo", file=sys.stdout, leave=False) as pbar:
        for i, c in enumerate(combos):
            try:
                run_op(*c, sync=False)
                valid_combos.append(c)
            except Exception:
                skipped += 1
            pbar.update(1)
            if (i + 1) % PROFILER_DUMP_EVERY == 0:
                ttnn.synchronize_device(mesh_device)
                ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)  # flush remaining warmup data

    if not valid_combos:
        return valid_combos, skipped

    # Measured run: trace-capture + execute + release per combo. Trace execution
    # synchronizes devices before dispatch, eliminating host dispatch skew that
    # can stall fabric transfers. Single-trace-at-a-time keeps trace memory
    # bounded regardless of combo count.
    signpost("start")
    with tqdm(total=len(valid_combos), desc="Measure", unit="combo", file=sys.stdout, leave=False) as pbar:
        for i, c in enumerate(valid_combos):
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            run_op(*c, sync=False)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)

            ttnn.release_trace(mesh_device, trace_id)

            pbar.update(1)
            if (i + 1) % PROFILER_DUMP_EVERY == 0:
                ttnn.ReadDeviceProfiler(mesh_device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh_device)  # final flush of measured data

    return valid_combos, skipped


# ============================================================================
# WORKER TEST — profiled in subprocess by device profiler
# ============================================================================


def _quiet_loguru():
    """Drop loguru's default INFO/WARNING sink — keep only ERROR+ on stderr."""
    logger.remove()
    logger.add(sys.stderr, level="ERROR")


# String -> ttnn enum maps for the external (spec-file-driven) worker path. The
# fusion knobs and dtype/fidelity in agmm/sweep_shapes.json are plain strings so
# the spec file stays tool-agnostic; they're resolved to ttnn enums here.
_FUSED_ACTIVATION_MAP = {
    "gelu": (ttnn.UnaryOpType.GELU, False),  # exact
    "gelu_approx": (ttnn.UnaryOpType.GELU, True),  # fast approx
    "silu": (ttnn.UnaryOpType.SILU, False),
    "relu": (ttnn.UnaryOpType.RELU, False),
}
_DTYPE_MAP = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "float32": ttnn.float32,
}
_MATH_FIDELITY_MAP = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}

# Label under which a spec-file shape's fusion config is registered in
# USE_CASE_CONFIGS so the L1 estimator (which looks up use_addcmul by use_case)
# sees it. One shape per worker subprocess, so a fixed label is safe.
EXTERNAL_USE_CASE = "__external__"


def _run_shape_sweep(device_config, M, K, N, cgx, cgy, is_agmm, use_case, uc_cfg, dtype, math_fidelity, op=None):
    """Sweep one (device_config, shape) and record the best config.

    Shared by the parametrized worker (SHAPES table) and the external worker
    (spec-file / env-driven). Opens the mesh once, sweeps every candidate, and
    periodically flushes the device profiler buffer to avoid overflow.

    `op` selects what is swept:
      - "agmm"/"mm": (M_block, K_block, N_block, sb_h, sb_w) block-size grid.
      - "ag":        isolated all-gather; the "combo" is
                     (num_workers_per_link, chunks_per_sync, 0, 0, 0).
    (Defaults from is_agmm for the SHAPES-table path.)

    Reads optional MM_SWEEP_EXPLICIT_COMBOS env var (JSON list of 5-tuples) to
    test a specific set of combos instead of the auto-generated grid. Writes the
    combos that survived warmup to MM_SWEEP_VALID_COMBOS_FILE if set, so the
    orchestrator/runner can line CSV rows up with combos.
    """
    if op is None:
        op = "agmm" if is_agmm else "mm"
    gathers_k = op in ("agmm", "ag")  # both split K across the ring
    cfg = resolve_config(device_config)
    shape = (M, K, N, cgx, cgy, gathers_k, use_case)

    cluster_size = cfg["mesh_shape"][cfg["cluster_axis"]]
    M_per_core, K_per_device, N_per_core = get_per_core_dims(shape, cluster_size)

    explicit_combos_str = os.environ.get("MM_SWEEP_EXPLICIT_COMBOS")
    if explicit_combos_str:
        # Each entry: [m_blk, k_blk, n_blk, sb_h, sb_w]
        combos = [tuple(c) for c in json.loads(explicit_combos_str)]
        m_cands = sorted({c[0] for c in combos})
        k_cands = sorted({c[1] for c in combos})
        n_cands = sorted({c[2] for c in combos})
    elif op == "ag":
        # Isolated all-gather: sweep AG knobs, not block sizes. The op runs on a
        # cluster-axis submesh (sp axis collapsed to 1), so the per-device gathered
        # tensor is M x K -- one sp position's rows, full K. (Multiplying by the
        # full mesh's sp size here over-sized the transfer by that factor.)
        gathered_elems = M * K
        combos = generate_ag_combos(gathered_elems, cluster_size, cfg["num_links"])
        m_cands = sorted({c[0] for c in combos})  # num_workers_per_link
        k_cands = sorted({c[1] for c in combos})  # chunks_per_sync
        n_cands = [0]
    else:
        m_cands = get_mn_block_candidates(M_per_core)
        k_cands = get_k_block_candidates(K_per_device)
        n_cands = get_mn_block_candidates(N_per_core)
        combos = []
        for m_block in m_cands:
            kn_combos = generate_kn_combos(K_per_device, N_per_core, m_block=m_block, use_case=use_case)
            for k_blk, n_blk in kn_combos:
                sb_h, sb_w = pick_subblock(m_block, n_blk)
                combos.append((m_block, k_blk, n_blk, sb_h, sb_w))

    shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_{op}_{use_case}"

    # Header: clean, one-time print of candidate lists + post-filter combo total
    print(f"\n=== {shape_id} on {device_config} ===", flush=True)
    print(f"  per_core: M={M_per_core}  K_per_device={K_per_device}  N={N_per_core}", flush=True)
    if op == "ag":
        print(f"  num_workers_per_link ({len(m_cands)}): {m_cands}", flush=True)
        print(f"  chunks_per_sync ({len(k_cands)}): {k_cands}", flush=True)
    else:
        print(f"  M_blocks ({len(m_cands)}): {m_cands}", flush=True)
        print(f"  K_blocks ({len(k_cands)}): {k_cands}", flush=True)
        print(f"  N_blocks ({len(n_cands)}): {n_cands}", flush=True)
    src = " (explicit)" if explicit_combos_str else " (post-filter)"
    print(f"  combos to measure: {len(combos)}{src}", flush=True)

    if not combos:
        pytest.skip("No valid combos after filter")

    parent_mesh, mesh_device = open_mesh(cfg, trace_region_size=4194304)  # 4MB trace region (one trace at a time)
    try:
        run_op = _build_op_runner(
            cfg,
            mesh_device,
            M,
            K,
            N,
            dtype,
            is_agmm,
            uc_cfg,
            ttnn.CoreCoord(cgx, cgy),
            math_fidelity=math_fidelity,
            op=op,
        )

        valid_combos, skipped = _execute_sweep(mesh_device, run_op, combos)

        if not valid_combos:
            pytest.skip("All combos failed during warmup")

        # Write valid combos (full tuples) so orchestrator can line up CSV rows
        combos_file = os.environ.get("MM_SWEEP_VALID_COMBOS_FILE")
        if combos_file:
            with open(combos_file, "w") as f:
                json.dump([list(c) for c in valid_combos], f)

        print(f"  measured: {len(valid_combos)}  skipped (L1 OOM): {skipped}", flush=True)

    finally:
        close_mesh(parent_mesh)


def _uc_cfg_from_fusion(fusion):
    """Translate a spec-file `fusion` dict (plain JSON) into a USE_CASE_CONFIGS-style dict."""
    fusion = fusion or {}
    uc_cfg = {}
    if fusion.get("chunks", 1) > 1:
        uc_cfg["chunks"] = fusion["chunks"]
    if fusion.get("math_approx_mode"):
        uc_cfg["math_approx_mode"] = True
    if fusion.get("use_addcmul"):
        uc_cfg["use_addcmul"] = True
    if fusion.get("scalar") is not None:
        uc_cfg["scalar"] = fusion["scalar"]
    if fusion.get("use_matmul_split"):
        uc_cfg["use_matmul_split"] = True
    fa = fusion.get("fused_activation")
    if fa:
        if fa not in _FUSED_ACTIVATION_MAP:
            raise ValueError(f"Unknown fused_activation '{fa}'. Known: {sorted(_FUSED_ACTIVATION_MAP)}")
        uc_cfg["fused_activation"] = _FUSED_ACTIVATION_MAP[fa]
    return uc_cfg


@pytest.mark.timeout(7200)  # 2h — one worker now covers the full (M, K, N) grid
@pytest.mark.parametrize("device_config", list(DEVICE_CONFIGS.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_mm_sweep_worker(device_config, shape):
    """Sweep every block-size combo for a (device_config, shape) from the SHAPES table.

    Designed to be invoked via run_device_profiler with TT_METAL_PROFILER_MID_RUN_DUMP=1.
    """
    _quiet_loguru()
    M, K, N, cgx, cgy, is_agmm, use_case = shape
    uc_cfg = USE_CASE_CONFIGS[use_case]
    _run_shape_sweep(
        device_config, M, K, N, cgx, cgy, is_agmm, use_case, uc_cfg, ttnn.bfloat16, ttnn.MathFidelity.HiFi2
    )


@pytest.mark.timeout(7200)
def test_mm_sweep_worker_external():
    """Sweep every block-size combo for ONE shape supplied via MM_SWEEP_SHAPE_JSON.

    The env var holds a single spec-file record (see agmm/sweep_shapes.json):
        {"id", "op_type", "device_config", "M", "K", "N", "grid": [cgx, cgy],
         "dtype", "math_fidelity", "fusion": {...}}
    Driven by agmm/run_sweeps.py; keeps arbitrary shapes out of the SHAPES table.
    """
    _quiet_loguru()
    spec = json.loads(os.environ["MM_SWEEP_SHAPE_JSON"])

    device_config = spec.get("device_config", DEFAULT_DEVICE_CONFIG)
    M, K, N = spec["M"], spec["K"], spec["N"]
    cgx, cgy = spec["grid"]
    op = spec["op_type"]  # "agmm" | "mm" | "ag"
    is_agmm = op == "agmm"
    dtype = _DTYPE_MAP.get(spec.get("dtype", "bfloat16"), ttnn.bfloat16)
    math_fidelity = _MATH_FIDELITY_MAP.get(spec.get("math_fidelity", "HiFi2"), ttnn.MathFidelity.HiFi2)

    uc_cfg = _uc_cfg_from_fusion(spec.get("fusion"))
    # Register so estimate_l1_kb() (keyed by use_case) sees use_addcmul for this shape.
    USE_CASE_CONFIGS[EXTERNAL_USE_CASE] = uc_cfg
    _run_shape_sweep(device_config, M, K, N, cgx, cgy, is_agmm, EXTERNAL_USE_CASE, uc_cfg, dtype, math_fidelity, op=op)


# ============================================================================
# ORCHESTRATOR TEST — iterates M_blocks, invokes worker via device profiler
# ============================================================================


@pytest.mark.timeout(7200)  # 2 hours — one subprocess per shape now
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance sweep - skip on CI")
@pytest.mark.parametrize("device_config", list(DEVICE_CONFIGS.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_mm_sweep(device_config, shape):
    """Orchestrate the block size sweep for one (device_config, shape).

    Spawns test_mm_sweep_worker in ONE profiler subprocess that opens the mesh
    device once and sweeps every (M, K, N) candidate. The worker uses
    TT_METAL_PROFILER_MID_RUN_DUMP=1 + periodic ttnn.ReadDeviceProfiler calls
    to flush the profiler buffer without losing data.

    Reads optional MM_SWEEP_EXPLICIT_COMBOS='[[m, k, n, sb_h, sb_w], ...]' to
    test a specific set; otherwise the worker auto-generates candidates from
    get_mn_block_candidates / generate_kn_combos. The subblock for each combo
    is pick_subblock(...) (and for fp32 dest that's always (2, 2) when valid).
    """
    from tracy.process_model_log import run_device_profiler

    M, K, N, cgx, cgy, is_agmm, use_case = shape
    op_type = "agmm" if is_agmm else "mm"
    shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_{op_type}_{use_case}"
    core_grid_str = f"{cgx}x{cgy}"

    subdir = f"mm_sweep_{device_config}_{shape_id}"
    combos_file = f"valid_combos_{device_config}_{shape_id}.json"
    os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
    # Mid-run profiler dumps: required so the worker can flush the device
    # profiler buffer between combos without losing data.
    os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    # Quiet tt-metal C++ warnings during the sweep — only show errors. Worker
    # also reconfigures loguru to ERROR-only.
    saved_logger_level = os.environ.get("TT_LOGGER_LEVEL")
    os.environ["TT_LOGGER_LEVEL"] = "Error"

    # `-s` so the worker's print/tqdm output flows through to the user's terminal.
    command = (
        f"pytest models/tt_dit/utils/sweep_mm_block_sizes.py"
        f"::test_mm_sweep_worker[{shape_id}-{device_config}] -x -s"
    )

    write_csv_header(CSV_FILE)

    try:
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    finally:
        os.environ.pop("MM_SWEEP_VALID_COMBOS_FILE", None)
        os.environ.pop("TT_METAL_PROFILER_MID_RUN_DUMP", None)
        os.environ.pop("MM_SWEEP_EXPLICIT_COMBOS", None)
        if saved_logger_level is None:
            os.environ.pop("TT_LOGGER_LEVEL", None)
        else:
            os.environ["TT_LOGGER_LEVEL"] = saved_logger_level

    if not os.path.exists(combos_file):
        print(f"  WARN: no valid_combos file at {combos_file}", flush=True)
        return
    with open(combos_file) as f:
        valid_combos = [tuple(c) for c in json.load(f)]
    os.remove(combos_file)

    durations = parse_ops_log(subdir, expected_ops=len(valid_combos))
    if len(durations) != len(valid_combos):
        print(
            f"  WARN: expected {len(valid_combos)} ops in profiler log, got {len(durations)}",
            flush=True,
        )

    all_results = []
    for i, (m_blk, k_blk, n_blk, sb_h, sb_w) in enumerate(valid_combos):
        if i < len(durations):
            duration_ns = durations[i]
            status = "OK"
            all_results.append(
                {
                    "M_block": m_blk,
                    "K_block": k_blk,
                    "N_block": n_blk,
                    "subblock_h": sb_h,
                    "subblock_w": sb_w,
                    "duration_ns": duration_ns,
                }
            )
        else:
            duration_ns = -1
            status = "MISSING"

        append_csv_row(
            CSV_FILE,
            [
                device_config,
                op_type,
                use_case,
                M,
                K,
                N,
                core_grid_str,
                m_blk,
                k_blk,
                n_blk,
                sb_h,
                sb_w,
                f"{duration_ns:.0f}",
                status,
            ],
        )

    if not all_results:
        print(f"  WARN: no valid results", flush=True)
        return

    all_results.sort(key=lambda r: r["duration_ns"])
    best = all_results[0]
    print(
        f"  BEST: M={best['M_block']} K={best['K_block']} N={best['N_block']} "
        f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns",
        flush=True,
    )
    print("  Top 5:", flush=True)
    for rank, r in enumerate(all_results[:5], 1):
        print(
            f"    #{rank}: M={r['M_block']} K={r['K_block']} N={r['N_block']} "
            f"sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns",
            flush=True,
        )


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================


def main():
    from tracy.process_model_log import run_device_profiler

    parser = argparse.ArgumentParser(description="Sweep matmul block sizes with device profiler")
    parser.add_argument(
        "--device-config",
        type=str,
        default=DEFAULT_DEVICE_CONFIG,
        choices=list(DEVICE_CONFIGS.keys()),
        help=f"Device configuration (default: {DEFAULT_DEVICE_CONFIG})",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Filter to a single shape as M,K,N (e.g. 6144,5120,3456)",
    )
    parser.add_argument("--csv", type=str, default=CSV_FILE)
    args = parser.parse_args()

    device_config = args.device_config
    cfg = resolve_config(device_config)

    # Filter shapes
    if args.shape:
        m, k, n = [int(x) for x in args.shape.split(",")]
        shapes = [s for s in SHAPES if s[0] == m and s[1] == k and s[2] == n]
        if not shapes:
            print(f"Shape {args.shape} not found in SHAPES table")
            return
    else:
        shapes = SHAPES

    write_csv_header(args.csv)
    all_best = {}

    print(
        f"Device config: {device_config} (mesh={cfg['mesh_shape']}, "
        f"sp_axis={cfg['sp_axis']}, links={cfg['num_links']})"
    )

    # Mid-run profiler dumps so the worker can flush between combos.
    os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"

    for shape in shapes:
        M, K, N, cgx, cgy, is_agmm, use_case = shape
        op_type = "agmm" if is_agmm else "mm"
        shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_{op_type}_{use_case}"
        core_grid_str = f"{cgx}x{cgy}"

        print(f"\n{'='*80}")
        print(f"[{device_config}] {op_type} ({use_case}) Shape {M}_{K}_{N} grid={core_grid_str}")
        print(f"{'='*80}")

        subdir = f"mm_sweep_{device_config}_{shape_id}"
        combos_file = f"valid_combos_{device_config}_{shape_id}.json"
        os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
        command = (
            f"pytest models/tt_dit/utils/sweep_mm_block_sizes.py"
            f"::test_mm_sweep_worker[{shape_id}-{device_config}] -x"
        )

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
        except Exception as e:
            print(f"  Sweep FAILED: {str(e)[:200]}")
            continue
        finally:
            os.environ.pop("MM_SWEEP_VALID_COMBOS_FILE", None)

        if not os.path.exists(combos_file):
            print(f"  No valid_combos file at {combos_file}")
            continue
        with open(combos_file) as f:
            valid_combos = [tuple(c) for c in json.load(f)]
        os.remove(combos_file)

        durations = parse_ops_log(subdir, expected_ops=len(valid_combos))
        if len(durations) != len(valid_combos):
            print(f"  expected {len(valid_combos)} ops in profiler log, got {len(durations)}")

        shape_results = []
        for i, (m_blk, k_blk, n_blk, sb_h, sb_w) in enumerate(valid_combos):
            if i < len(durations):
                duration_ns = durations[i]
                status = "OK"
                shape_results.append(
                    {
                        "M_block": m_blk,
                        "K_block": k_blk,
                        "N_block": n_blk,
                        "subblock_h": sb_h,
                        "subblock_w": sb_w,
                        "duration_ns": duration_ns,
                    }
                )
            else:
                duration_ns = -1
                status = "MISSING"
            append_csv_row(
                args.csv,
                [
                    device_config,
                    op_type,
                    use_case,
                    M,
                    K,
                    N,
                    core_grid_str,
                    m_blk,
                    k_blk,
                    n_blk,
                    sb_h,
                    sb_w,
                    f"{duration_ns:.0f}",
                    status,
                ],
            )

        if shape_results:
            shape_results.sort(key=lambda r: r["duration_ns"])
            best = shape_results[0]
            print(
                f"  BEST: M={best['M_block']} K={best['K_block']} N={best['N_block']} "
                f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns"
            )
            print("  Top 5:")
            for rank, r in enumerate(shape_results[:5], 1):
                print(
                    f"    #{rank}: M={r['M_block']} K={r['K_block']} N={r['N_block']} "
                    f"sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns"
                )
            all_best[(M, K, N, cgx, cgy, op_type, use_case)] = best

    os.environ.pop("TT_METAL_PROFILER_MID_RUN_DUMP", None)

    if all_best:
        print(f"\n{'='*110}")
        print(f"SWEEP SUMMARY [{device_config}] - Best configs per shape")
        print(f"{'='*110}")
        print(
            f"{'type':>4} {'use_case':>10} {'M':>6} {'K':>6} {'N':>6} {'grid':>7} | "
            f"{'M_blk':>5} {'K_blk':>5} {'N_blk':>5} {'sb_h':>4} {'sb_w':>4} | "
            f"{'duration_ns':>12}"
        )
        print("-" * 110)
        for (M, K, N, cgx, cgy, op_type, use_case), best in sorted(all_best.items()):
            print(
                f"{op_type:>4} {use_case:>10} {M:>6} {K:>6} {N:>6} {cgx}x{cgy:>2} | "
                f"{best['M_block']:>5} {best['K_block']:>5} {best['N_block']:>5} "
                f"{best['subblock_h']:>4} {best['subblock_w']:>4} | "
                f"{best['duration_ns']:>12.0f}"
            )
        print(f"{'='*110}")
        print(f"\nFull results written to: {args.csv}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Sweep block sizes for minimal_matmul and all_gather_minimal_matmul_async
using device profiler for accurate kernel timing.

Architecture:
  - Worker test (test_mm_sweep_worker): Self-contained profiled test.
    For a given (device_config, shape, M_block), runs all valid (K_block, N_block)
    combos, calling the appropriate op directly. Invoked as subprocess by device profiler.
  - Orchestrator test (test_mm_sweep): Iterates M_blocks, spawns worker via
    run_device_profiler, parses ops log to extract per-op device kernel durations.

Usage:
    # Orchestrator: sweep one shape on BH 4x8
    pytest models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py::test_mm_sweep \\
        -k "6144_5120_3456_11x10_mm-bh_4x8" -x -s

    # Worker: run directly (useful for debugging, no profiling)
    pytest models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py::test_mm_sweep_worker \\
        -k "m4-6144_5120_3456_11x10_mm-bh_4x8" -x -s

    # Standalone script
    python models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py \\
        --device-config bh_4x8 --shape 6144,5120,3456
"""

import argparse
import csv
import json
import os

import pytest
import torch
from loguru import logger

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
SHAPES = [
    (96, 96, 192, 11, 10, False, "plain"),
    (64, 192, 384, 11, 10, False, "plain"),
    (64, 96, 192, 11, 10, False, "plain"),
    (32, 96, 192, 11, 10, False, "plain"),
    (32, 192, 384, 11, 10, False, "plain"),
    (32, 256, 5120, 11, 10, False, "plain"),
    (32, 32, 32, 11, 10, False, "plain"),
    (32, 1280, 30720, 11, 10, False, "plain"),
    (32, 3072, 10240, 11, 10, False, "plain"),
    (32, 5120, 1280, 11, 10, False, "plain"),
    (32, 10240, 10240, 11, 10, False, "plain"),
    (128, 5120, 2560, 11, 10, False, "cross_attn_kv"),
    (512, 4096, 5120, 11, 10, False, "plain"),
    (512, 5120, 5120, 11, 10, False, "plain"),
    (6144, 384, 384, 11, 10, False, "plain"),
    (6144, 384, 1152, 11, 10, False, "plain"),
    (6144, 3456, 5120, 11, 10, False, "ff2"),
    (6144, 5120, 64, 11, 10, False, "plain"),
    (6144, 5120, 1280, 12, 9, True, "to_out"),
    (6144, 5120, 3456, 11, 10, False, "plain"),
    (6144, 5120, 3456, 12, 9, True, "ff1_gelu"),
    (6144, 5120, 3840, 12, 9, True, "qkv"),
    (6240, 384, 384, 11, 10, False, "plain"),
    (6240, 384, 1152, 11, 10, False, "plain"),
    (6240, 3456, 5120, 11, 10, False, "ff2"),
    (6240, 5120, 64, 11, 10, False, "plain"),
    (6240, 5120, 1280, 12, 9, True, "to_out"),
    (6240, 5120, 3456, 11, 10, False, "plain"),
    (6240, 5120, 3456, 12, 9, True, "ff1_gelu"),
    (6240, 5120, 3840, 12, 9, True, "qkv"),
    (14400, 384, 384, 11, 10, False, "plain"),
    (14400, 384, 1152, 11, 10, False, "plain"),
    (14400, 3456, 5120, 11, 10, False, "ff2"),
    (14400, 5120, 64, 11, 10, False, "plain"),
    (14400, 5120, 1280, 12, 9, True, "to_out"),
    (14400, 5120, 3456, 11, 10, False, "plain"),
    (14400, 5120, 3456, 12, 9, True, "ff1_gelu"),
    (14400, 5120, 3840, 12, 9, True, "qkv"),
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
    "cross_attn_kv": {
        "chunks": 2,
        "math_approx_mode": True,
        "use_matmul_split": True,
    },
}

# Block sweep range
MAX_BLOCK = 64

# Base block sizes to always include (union with divisors).
# Covers powers of 2 plus common non-power-of-2 values found in known-best configs.
BASE_BLOCK_SIZES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]

# AGMM-specific restricted candidate sets to reduce profiler overhead.
# K: divisors only (all known-best K_blocks divide K_tiles on 12x9 grid).
# N: restricted set covering all known-best N_block values (includes 3, 6 from configs).
AGMM_N_BLOCK_CANDIDATES = [1, 2, 3, 4, 6, 8, 12, 16]

# L1 budget for pre-filtering block combos (KB).
# BH L1 usable ~1464 KB; conservative threshold accounts for kernel/firmware overhead.
# CB allocation: 2*M*K + 2*K*N + 2*M*N tiles (all double-buffered, bf16 = 2KB/tile)
#              + M*N tiles intermediate (single-buffered, f32 = 4KB/tile)
#              + N tiles bias (single-buffered, bf16 = 2KB/tile)
# to_out adds: M*N tiles ternary_a (bf16) + N tiles ternary_c (bf16)
L1_BUDGET_KB = 1400

# Max combos per profiler subprocess to avoid DRAM profiler buffer overflow.
# AGMM ops generate many more profiler markers per op (fabric, all-gather, etc.)
# so need smaller batches. Non-AGMM can handle more.
PROFILER_BATCH_SIZE_AGMM = 8
PROFILER_BATCH_SIZE_MM = 256

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


def get_divisors(n, max_val=MAX_BLOCK):
    """Return sorted list of divisors of n, each <= max_val."""
    if n <= 0:
        return [1]
    return sorted(i for i in range(1, min(n, max_val) + 1) if n % i == 0)


def get_block_candidates(n_tiles, max_val=MAX_BLOCK):
    """Return sorted candidate block sizes: divisors of n_tiles union BASE_BLOCK_SIZES, each <= min(n_tiles, max_val)."""
    cap = min(n_tiles, max_val)
    divisors = set(i for i in range(1, cap + 1) if n_tiles % i == 0)
    base = set(b for b in BASE_BLOCK_SIZES if b <= cap)
    return sorted(divisors | base)


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
    """Pick best valid (sb_h, sb_w) where sb_h|m_block, sb_w|n_block, sb_h*sb_w <= max_dest_volume."""
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


def generate_subblock_combos(m_block, n_block, max_dest_volume=4):
    """Generate all valid (sb_h, sb_w) where sb_h|m_block, sb_w|n_block, sb_h*sb_w <= max_dest_volume."""
    combos = []
    for h in range(1, min(m_block, max_dest_volume) + 1):
        if m_block % h != 0:
            continue
        for w in range(1, min(n_block, max_dest_volume) + 1):
            if n_block % w != 0:
                continue
            if h * w <= max_dest_volume:
                combos.append((h, w))
    return combos


def generate_kn_combos(K_tiles, N_tiles, m_block=1, use_case="plain", is_agmm=False):
    """Generate (K_block, N_block) combos filtered by L1 budget.

    For non-AGMM: divisors union BASE_BLOCK_SIZES for both K and N.
    For AGMM: K divisors only (no BASE union), N from AGMM_N_BLOCK_CANDIDATES.
    This reduces AGMM combos to avoid profiler DRAM buffer overflow.
    """
    if is_agmm:
        k_candidates = get_divisors(K_tiles)
        cap = min(N_tiles, MAX_BLOCK)
        n_candidates = sorted(b for b in AGMM_N_BLOCK_CANDIDATES if b <= cap)
    else:
        k_candidates = get_block_candidates(K_tiles)
        n_candidates = get_block_candidates(N_tiles)
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
    """Open a mesh device with the given resolved config."""
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
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), **device_kwargs)


def close_mesh(mesh_device):
    """Close mesh device and reset fabric."""
    ttnn.close_mesh_device(mesh_device)
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
# WORKER TEST — profiled in subprocess by device profiler
# ============================================================================


@pytest.mark.timeout(3000)
@pytest.mark.parametrize("device_config", list(DEVICE_CONFIGS.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("m_block", range(1, MAX_BLOCK + 1), ids=[f"m{i}" for i in range(1, MAX_BLOCK + 1)])
def test_mm_sweep_worker(device_config, shape, m_block):
    """Run all (K_block, N_block) combos for a given device config, shape, and M_block.

    Designed to be invoked via run_device_profiler as a subprocess.
    Emits start/stop signposts around the measured region.
    """
    cfg = resolve_config(device_config)
    M, K, N, cgx, cgy, is_agmm, use_case = shape
    uc_cfg = USE_CASE_CONFIGS[use_case]

    M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)

    # Explicit combos override: JSON list of [k_blk, n_blk, sb_h, sb_w] tuples
    # When set, m_block must match the expected value and we skip normal generation.
    explicit_combos_str = os.environ.get("MM_SWEEP_EXPLICIT_COMBOS")
    if explicit_combos_str:
        explicit_combos = json.loads(explicit_combos_str)
        # Each entry is [k_blk, n_blk, sb_h, sb_w]; we only use k_blk/n_blk here,
        # sb is passed separately via the run_op closure below.
        kn_combos = [(c[0], c[1]) for c in explicit_combos]
        # Override pick_subblock with explicit subblocks
        _explicit_subblocks = {(c[0], c[1]): (c[2], c[3]) for c in explicit_combos}
    else:
        _explicit_subblocks = None
        m_candidates = get_block_candidates(M_tiles)
        if m_block not in m_candidates:
            pytest.skip(f"m_block={m_block} not a candidate for M_tiles={M_tiles}")

        kn_combos = generate_kn_combos(K_tiles, N_tiles, m_block=m_block, use_case=use_case, is_agmm=is_agmm)
        if not kn_combos:
            pytest.skip("No valid (K_block, N_block) combos after L1 filter")

        # Optional batch slicing to avoid profiler DRAM buffer overflow on AGMM
        batch_start = int(os.environ.get("MM_SWEEP_BATCH_START", 0))
        batch_end = int(os.environ.get("MM_SWEEP_BATCH_END", len(kn_combos)))
        kn_combos = kn_combos[batch_start:batch_end]
        if not kn_combos:
            pytest.skip(f"Empty batch [{batch_start}:{batch_end}]")

    op_type = "agmm" if is_agmm else "mm"
    if explicit_combos_str:
        logger.info(
            f"Worker [{device_config}] {op_type} ({use_case}): M={M} K={K} N={N} grid={cgx}x{cgy} "
            f"m_block={m_block}, {len(kn_combos)} EXPLICIT combos"
        )
    else:
        logger.info(
            f"Worker [{device_config}] {op_type} ({use_case}): M={M} K={K} N={N} grid={cgx}x{cgy} "
            f"m_block={m_block}, {len(kn_combos)} K/N combos (batch [{batch_start}:{batch_end}])"
        )

    mesh_device = open_mesh(cfg, trace_region_size=90112 if is_agmm else None)
    try:
        core_grid = ttnn.CoreCoord(cgx, cgy)
        dtype = ttnn.bfloat16

        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=uc_cfg.get("math_approx_mode", False),
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        fused_activation = uc_cfg.get("fused_activation", None)
        chunks = uc_cfg.get("chunks", 1)
        scalar = uc_cfg.get("scalar", None)

        if is_agmm:
            # ----- AGMM path: sharded input + CCL infrastructure -----
            sp_axis = cfg["sp_axis"]
            tp_axis = cfg["tp_axis"]
            sp_size = cfg["mesh_shape"][sp_axis]

            # M is per-device; create full tensor for mesh sharding
            full_M = M * sp_size
            shard_dims = [sp_axis, tp_axis]
            tt_input = ttnn.from_torch(
                torch.randn((full_M, K), dtype=torch.float32),
                dtype=dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
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

            # Use full compute grid for semaphores (matching model's CCLManager)
            full_grid = mesh_device.compute_with_storage_grid_size()
            ccl_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))}
            )

            # Create 2 semaphores (matching model's CCLManager ag_ping_pong pattern)
            ccl_semaphore_handles = [
                ttnn.create_global_semaphore(mesh_device, ccl_cores, 0),
                ttnn.create_global_semaphore(mesh_device, ccl_cores, 0),
            ]

            # 4D buffer without mesh_mapper (matching model's CCLManager)
            persistent_output_buffer = ttnn.from_torch(
                torch.empty((1, 1, M, K), dtype=torch.float32),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=mesh_device,
            )

            # Allocate dummy addcmul tensors if needed (to_out use case)
            addcmul_tensor1 = None
            addcmul_tensor2 = None
            if uc_cfg.get("use_addcmul", False):
                addcmul_tensor1 = ttnn.from_torch(
                    torch.randn((M, N), dtype=torch.float32),
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]
                    ),
                )
                addcmul_tensor2 = ttnn.from_torch(
                    torch.randn((M, N), dtype=torch.float32),
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]
                    ),
                )

            def run_op(k_blk, n_blk, sync=True):
                if _explicit_subblocks and (k_blk, n_blk) in _explicit_subblocks:
                    sb_h, sb_w = _explicit_subblocks[(k_blk, n_blk)]
                else:
                    sb_h, sb_w = pick_subblock(m_block, n_blk)
                matmul_config = ttnn.MinimalMatmulConfig(
                    M_block_size=m_block,
                    K_block_size=k_blk,
                    N_block_size=n_blk,
                    subblock_h=sb_h,
                    subblock_w=sb_w,
                    compute_with_storage_grid_size=core_grid,
                )
                ttnn.experimental.all_gather_minimal_matmul_async(
                    tt_input,
                    tt_weight,
                    bias_tensor=tt_bias,
                    fused_activation=fused_activation,
                    compute_kernel_config=compute_config,
                    config=matmul_config,
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

        else:
            # ----- Non-AGMM path: replicated tensors + minimal_matmul -----
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

            if use_matmul_split:

                def run_op(k_blk, n_blk):
                    if _explicit_subblocks and (k_blk, n_blk) in _explicit_subblocks:
                        sb_h, sb_w = _explicit_subblocks[(k_blk, n_blk)]
                    else:
                        sb_h, sb_w = pick_subblock(m_block, n_blk)
                    matmul_config = ttnn.MinimalMatmulConfig(
                        M_block_size=m_block,
                        K_block_size=k_blk,
                        N_block_size=n_blk,
                        subblock_h=sb_h,
                        subblock_w=sb_w,
                        compute_with_storage_grid_size=core_grid,
                    )
                    ttnn.experimental.minimal_matmul_split(
                        tt_input,
                        tt_weight,
                        chunks=chunks,
                        dim=-1,
                        bias_tensor=tt_bias,
                        fused_activation=fused_activation,
                        compute_kernel_config=compute_config,
                        config=matmul_config,
                    )
                    ttnn.synchronize_device(mesh_device)

            else:

                def run_op(k_blk, n_blk):
                    if _explicit_subblocks and (k_blk, n_blk) in _explicit_subblocks:
                        sb_h, sb_w = _explicit_subblocks[(k_blk, n_blk)]
                    else:
                        sb_h, sb_w = pick_subblock(m_block, n_blk)
                    matmul_config = ttnn.MinimalMatmulConfig(
                        M_block_size=m_block,
                        K_block_size=k_blk,
                        N_block_size=n_blk,
                        subblock_h=sb_h,
                        subblock_w=sb_w,
                        compute_with_storage_grid_size=core_grid,
                    )
                    ttnn.experimental.minimal_matmul(
                        input_tensor=tt_input,
                        weight_tensor=tt_weight,
                        bias_tensor=tt_bias,
                        config=matmul_config,
                        fused_activation=fused_activation,
                        compute_kernel_config=compute_config,
                    )
                    ttnn.synchronize_device(mesh_device)

        # Warmup: compile all programs, skip combos that OOM
        valid_combos = []
        for k_blk, n_blk in kn_combos:
            try:
                run_op(k_blk, n_blk)
                valid_combos.append((k_blk, n_blk))
            except Exception as e:
                logger.warning(f"Skipping K_block={k_blk} N_block={n_blk}: {e}")

        if not valid_combos:
            pytest.skip("All K/N combos failed during warmup")

        skipped = len(kn_combos) - len(valid_combos)
        if skipped:
            logger.info(f"Warmup done: {len(valid_combos)} valid, {skipped} skipped (L1 OOM)")
        else:
            logger.info(f"Warmup done: all {len(valid_combos)} combos valid")

        # Write valid combos file so orchestrator knows which ran
        combos_file = os.environ.get("MM_SWEEP_VALID_COMBOS_FILE")
        if combos_file:
            with open(combos_file, "w") as f:
                json.dump(valid_combos, f)

        # Measured run — only valid combos
        from tracy import signpost

        if is_agmm:
            # Capture a trace per combo (ops already compiled from warmup).
            # Trace execution synchronizes all devices before dispatching,
            # eliminating host dispatch skew that can stall fabric transfers.
            trace_ids = []
            for k_blk, n_blk in valid_combos:
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                run_op(k_blk, n_blk, sync=False)
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(mesh_device)
                trace_ids.append(trace_id)

            # Only trace executions appear between signposts
            signpost("start")
            for trace_id in trace_ids:
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")

            for trace_id in trace_ids:
                ttnn.release_trace(mesh_device, trace_id)
        else:
            signpost("start")
            for k_blk, n_blk in valid_combos:
                run_op(k_blk, n_blk)
            signpost("stop")

        logger.info(f"Worker done: {len(valid_combos)} combos measured")

    finally:
        close_mesh(mesh_device)


# ============================================================================
# SUBBLOCK SWEEP WORKER — sweeps subblocks for a fixed (M, K, N) block config
# ============================================================================


@pytest.mark.timeout(3000)
@pytest.mark.parametrize("device_config", list(DEVICE_CONFIGS.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_mm_subblock_sweep_worker(device_config, shape):
    """Sweep subblock sizes for fixed (M_block, K_block, N_block) read from env vars.

    Designed to be invoked by the orchestrator's pass 2 via run_device_profiler.
    Block sizes are passed via MM_SWEEP_M_BLOCK, MM_SWEEP_K_BLOCK, MM_SWEEP_N_BLOCK env vars.
    """
    m_block = int(os.environ["MM_SWEEP_M_BLOCK"])
    k_block = int(os.environ["MM_SWEEP_K_BLOCK"])
    n_block = int(os.environ["MM_SWEEP_N_BLOCK"])

    cfg = resolve_config(device_config)
    M, K, N, cgx, cgy, is_agmm, use_case = shape
    uc_cfg = USE_CASE_CONFIGS[use_case]

    sb_combos = generate_subblock_combos(m_block, n_block)
    if len(sb_combos) <= 1:
        pytest.skip("Only one valid subblock combo")

    logger.info(
        f"Subblock worker [{device_config}] ({use_case}): M={M} K={K} N={N} "
        f"blocks=({m_block},{k_block},{n_block}), {len(sb_combos)} subblock combos"
    )

    mesh_device = open_mesh(cfg)
    try:
        core_grid = ttnn.CoreCoord(cgx, cgy)
        dtype = ttnn.bfloat16

        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=uc_cfg.get("math_approx_mode", False),
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        fused_activation = uc_cfg.get("fused_activation", None)
        chunks = uc_cfg.get("chunks", 1)
        scalar = uc_cfg.get("scalar", None)

        if is_agmm:
            sp_axis = cfg["sp_axis"]
            tp_axis = cfg["tp_axis"]
            sp_size = cfg["mesh_shape"][sp_axis]

            full_M = M * sp_size
            shard_dims = [sp_axis, tp_axis]
            tt_input = ttnn.from_torch(
                torch.randn((full_M, K), dtype=torch.float32),
                dtype=dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
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

            # Use full compute grid for semaphores (matching model's CCLManager)
            full_grid = mesh_device.compute_with_storage_grid_size()
            ccl_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))}
            )

            # Create 2 semaphores (matching model's CCLManager ag_ping_pong pattern)
            ccl_semaphore_handles = [
                ttnn.create_global_semaphore(mesh_device, ccl_cores, 0),
                ttnn.create_global_semaphore(mesh_device, ccl_cores, 0),
            ]

            # 4D buffer without mesh_mapper (matching model's CCLManager)
            persistent_output_buffer = ttnn.from_torch(
                torch.empty((1, 1, M, K), dtype=torch.float32),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=mesh_device,
            )

            addcmul_tensor1 = None
            addcmul_tensor2 = None
            if uc_cfg.get("use_addcmul", False):
                addcmul_tensor1 = ttnn.from_torch(
                    torch.randn((M, N), dtype=torch.float32),
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]
                    ),
                )
                addcmul_tensor2 = ttnn.from_torch(
                    torch.randn((M, N), dtype=torch.float32),
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]
                    ),
                )

            def run_op(sb_h, sb_w):
                matmul_config = ttnn.MinimalMatmulConfig(
                    M_block_size=m_block,
                    K_block_size=k_block,
                    N_block_size=n_block,
                    subblock_h=sb_h,
                    subblock_w=sb_w,
                    compute_with_storage_grid_size=core_grid,
                )
                ttnn.experimental.all_gather_minimal_matmul_async(
                    tt_input,
                    tt_weight,
                    bias_tensor=tt_bias,
                    fused_activation=fused_activation,
                    compute_kernel_config=compute_config,
                    config=matmul_config,
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
                ttnn.synchronize_device(mesh_device)

        else:
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

            if use_matmul_split:

                def run_op(sb_h, sb_w):
                    matmul_config = ttnn.MinimalMatmulConfig(
                        M_block_size=m_block,
                        K_block_size=k_block,
                        N_block_size=n_block,
                        subblock_h=sb_h,
                        subblock_w=sb_w,
                        compute_with_storage_grid_size=core_grid,
                    )
                    ttnn.experimental.minimal_matmul_split(
                        tt_input,
                        tt_weight,
                        chunks=chunks,
                        dim=-1,
                        bias_tensor=tt_bias,
                        fused_activation=fused_activation,
                        compute_kernel_config=compute_config,
                        config=matmul_config,
                    )
                    ttnn.synchronize_device(mesh_device)

            else:

                def run_op(sb_h, sb_w):
                    matmul_config = ttnn.MinimalMatmulConfig(
                        M_block_size=m_block,
                        K_block_size=k_block,
                        N_block_size=n_block,
                        subblock_h=sb_h,
                        subblock_w=sb_w,
                        compute_with_storage_grid_size=core_grid,
                    )
                    ttnn.experimental.minimal_matmul(
                        input_tensor=tt_input,
                        weight_tensor=tt_weight,
                        bias_tensor=tt_bias,
                        config=matmul_config,
                        fused_activation=fused_activation,
                        compute_kernel_config=compute_config,
                    )
                    ttnn.synchronize_device(mesh_device)

        # Warmup — skip combos that OOM
        valid_combos = []
        for sb_h, sb_w in sb_combos:
            try:
                run_op(sb_h, sb_w)
                valid_combos.append((sb_h, sb_w))
            except Exception as e:
                logger.warning(f"Skipping subblock ({sb_h},{sb_w}): {e}")

        if not valid_combos:
            pytest.skip("All subblock combos failed during warmup")

        logger.info(f"Subblock warmup done: {len(valid_combos)}/{len(sb_combos)} valid")

        combos_file = os.environ.get("MM_SWEEP_VALID_COMBOS_FILE")
        if combos_file:
            with open(combos_file, "w") as f:
                json.dump(valid_combos, f)

        from tracy import signpost

        signpost("start")
        for sb_h, sb_w in valid_combos:
            run_op(sb_h, sb_w)
        signpost("stop")

        logger.info(f"Subblock worker done: {len(valid_combos)} combos measured")

    finally:
        close_mesh(mesh_device)


# ============================================================================
# ORCHESTRATOR TEST — iterates M_blocks, invokes worker via device profiler
# ============================================================================


@pytest.mark.timeout(21600)  # 6 hours — AGMM sweeps with batching are slow (mesh open + compile per batch)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance sweep - skip on CI")
@pytest.mark.parametrize("device_config", list(DEVICE_CONFIGS.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_mm_sweep(device_config, shape):
    """Orchestrate the block size sweep for one (device_config, shape).

    For each valid M_block, invokes test_mm_sweep_worker via the device profiler
    as a subprocess, then parses the ops log to extract device kernel durations.
    """
    from tracy.process_model_log import run_device_profiler

    cfg = resolve_config(device_config)
    M, K, N, cgx, cgy, is_agmm, use_case = shape
    M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
    op_type = "agmm" if is_agmm else "mm"
    shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_{op_type}_{use_case}"
    core_grid_str = f"{cgx}x{cgy}"

    # Explicit combos mode: set MM_SWEEP_EXPLICIT_COMBOS='[[m,k,n,sb_h,sb_w],...]'
    # to test only specific configs. Skips normal sweep and pass 2.
    explicit_combos_env = os.environ.get("MM_SWEEP_EXPLICIT_COMBOS")
    if explicit_combos_env:
        explicit_list = json.loads(explicit_combos_env)
        # Group by m_block
        from collections import defaultdict

        by_m = defaultdict(list)
        for combo in explicit_list:
            by_m[combo[0]].append(combo)
        m_blocks = sorted(by_m.keys())
        logger.info(f"EXPLICIT COMBOS MODE: {len(explicit_list)} combos across {len(m_blocks)} M_blocks for {shape_id}")
    else:
        explicit_list = None
        by_m = None
        m_blocks = get_block_candidates(M_tiles)

        # Log total combo counts (K/N combos vary per m_block due to L1 filter)
        sample_kn = generate_kn_combos(K_tiles, N_tiles, m_block=m_blocks[0], use_case=use_case, is_agmm=is_agmm)
        if not sample_kn and not any(
            generate_kn_combos(K_tiles, N_tiles, m_block=m, use_case=use_case, is_agmm=is_agmm) for m in m_blocks
        ):
            pytest.skip(f"No valid (K_block, N_block) combos for K_tiles={K_tiles}, N_tiles={N_tiles}")

        logger.info(
            f"Sweep [{device_config}] {op_type} ({use_case}) {shape_id}: "
            f"M_tiles={M_tiles}, K_tiles={K_tiles}, N_tiles={N_tiles}, "
            f"{len(m_blocks)} M_blocks (divisors+base), L1-filtered K/N combos"
        )

    write_csv_header(CSV_FILE)
    all_results = []

    batch_size = PROFILER_BATCH_SIZE_AGMM if is_agmm else PROFILER_BATCH_SIZE_MM

    for m_block in m_blocks:
        if explicit_list is not None:
            # Explicit mode: use the combos for this m_block, pass as env var to worker
            m_combos = by_m[m_block]
            # kn_combos for orchestrator tracking: (k, n) pairs
            kn_combos = [(c[1], c[2]) for c in m_combos]
            # Explicit subblock lookup: (k, n) -> (sb_h, sb_w)
            explicit_sb = {(c[1], c[2]): (c[3], c[4]) for c in m_combos}
            # Worker needs [k, n, sb_h, sb_w] format
            worker_explicit = [[c[1], c[2], c[3], c[4]] for c in m_combos]
            os.environ["MM_SWEEP_EXPLICIT_COMBOS"] = json.dumps(worker_explicit)
            logger.info(f"  M_block={m_block}: {len(kn_combos)} explicit combos")
        else:
            explicit_sb = None
            os.environ.pop("MM_SWEEP_EXPLICIT_COMBOS", None)
            kn_combos = generate_kn_combos(K_tiles, N_tiles, m_block=m_block, use_case=use_case, is_agmm=is_agmm)
            if not kn_combos:
                logger.info(f"  M_block={m_block}: all K/N combos exceed L1 budget, skipping")
                continue

        # Split into batches to avoid profiler DRAM buffer overflow
        batches = [
            (b_start, min(b_start + batch_size, len(kn_combos))) for b_start in range(0, len(kn_combos), batch_size)
        ]
        n_batches = len(batches)
        batch_label = f" ({n_batches} batches)" if n_batches > 1 else ""

        logger.info(f"  M_block={m_block}: profiling {len(kn_combos)} combos (L1-filtered){batch_label}...")

        m_block_durations = []
        m_block_valid_combos = []
        m_block_failed = False

        for batch_idx, (b_start, b_end) in enumerate(batches):
            batch_combos = kn_combos[b_start:b_end]
            batch_suffix = f"_b{batch_idx}" if n_batches > 1 else ""
            subdir = f"mm_sweep_{device_config}_{shape_id}_m{m_block}{batch_suffix}"
            combos_file = f"valid_combos_{device_config}_{shape_id}_m{m_block}{batch_suffix}.json"
            os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
            if explicit_list is None:
                os.environ["MM_SWEEP_BATCH_START"] = str(b_start)
                os.environ["MM_SWEEP_BATCH_END"] = str(b_end)
            command = (
                f"pytest models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py"
                f"::test_mm_sweep_worker[m{m_block}-{shape_id}-{device_config}] -x"
            )

            if n_batches > 1:
                logger.info(f"    batch {batch_idx + 1}/{n_batches}: combos [{b_start}:{b_end}]")

            try:
                run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])

                # Read valid combos from worker first (may be subset due to runtime L1 OOM)
                if os.path.exists(combos_file):
                    with open(combos_file) as f:
                        valid_combos = [tuple(c) for c in json.load(f)]
                    os.remove(combos_file)
                else:
                    valid_combos = batch_combos

                durations = parse_ops_log(subdir, expected_ops=len(valid_combos))

                if len(durations) != len(valid_combos):
                    logger.warning(
                        f"  M_block={m_block}{batch_suffix}: expected {len(valid_combos)} ops, "
                        f"got {len(durations)} in profiler log"
                    )

                m_block_durations.extend(durations)
                m_block_valid_combos.extend(valid_combos)

            except Exception as e:
                err_msg = str(e)
                if len(err_msg) > 200:
                    err_msg = err_msg[:200] + "..."
                logger.warning(f"  M_block={m_block}{batch_suffix}: FAILED - {err_msg}")
                # Record failures for this batch
                for k_blk, n_blk in batch_combos:
                    sb_h, sb_w = (
                        explicit_sb[(k_blk, n_blk)]
                        if explicit_sb and (k_blk, n_blk) in explicit_sb
                        else pick_subblock(m_block, n_blk)
                    )
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
                            m_block,
                            k_blk,
                            n_blk,
                            sb_h,
                            sb_w,
                            -1,
                            f"FAIL: {err_msg[:80]}",
                        ],
                    )
                m_block_failed = True

        # Clean up batch env vars
        os.environ.pop("MM_SWEEP_BATCH_START", None)
        os.environ.pop("MM_SWEEP_BATCH_END", None)
        os.environ.pop("MM_SWEEP_EXPLICIT_COMBOS", None)

        # Record results for all successful batches
        skipped = len(kn_combos) - len(m_block_valid_combos)
        if skipped and not m_block_failed:
            logger.info(f"  M_block={m_block}: {skipped} combos skipped (runtime L1 OOM)")

        for i, (k_blk, n_blk) in enumerate(m_block_valid_combos):
            sb_h, sb_w = (
                explicit_sb[(k_blk, n_blk)]
                if explicit_sb and (k_blk, n_blk) in explicit_sb
                else pick_subblock(m_block, n_blk)
            )
            if i < len(m_block_durations):
                duration_ns = m_block_durations[i]
                status = "OK"
                all_results.append(
                    {
                        "M_block": m_block,
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
                    m_block,
                    k_blk,
                    n_blk,
                    sb_h,
                    sb_w,
                    f"{duration_ns:.0f}",
                    status,
                ],
            )

        logger.info(
            f"  M_block={m_block}: done, " f"{min(len(m_block_durations), len(m_block_valid_combos))} results recorded"
        )

    # Pass 1 summary
    if not all_results:
        logger.warning(f"No valid results for [{device_config}] {shape_id}")
        return

    all_results.sort(key=lambda r: r["duration_ns"])
    best = all_results[0]
    logger.info(
        f"Pass 1 BEST for [{device_config}] {shape_id}: "
        f"M={best['M_block']} K={best['K_block']} N={best['N_block']} "
        f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns"
    )
    logger.info("Pass 1 top 5:")
    for rank, r in enumerate(all_results[:5], 1):
        logger.info(
            f"  #{rank}: M={r['M_block']} K={r['K_block']} N={r['N_block']} "
            f"sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns"
        )

    # Pass 2: subblock sweep on best (M_block, K_block, N_block)
    # Skip if explicit combos were provided (subblocks already specified).
    best_m, best_k, best_n = best["M_block"], best["K_block"], best["N_block"]
    sb_combos = generate_subblock_combos(best_m, best_n) if not explicit_list else []

    if len(sb_combos) > 1:
        logger.info(f"Pass 2: sweeping {len(sb_combos)} subblock combos for " f"blocks=({best_m},{best_k},{best_n})...")

        os.environ["MM_SWEEP_M_BLOCK"] = str(best_m)
        os.environ["MM_SWEEP_K_BLOCK"] = str(best_k)
        os.environ["MM_SWEEP_N_BLOCK"] = str(best_n)

        subdir = f"mm_sweep_{device_config}_{shape_id}_subblock"
        combos_file = f"valid_combos_{device_config}_{shape_id}_subblock.json"
        os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
        command = (
            f"pytest models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py"
            f"::test_mm_subblock_sweep_worker[{shape_id}-{device_config}] -x"
        )

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])

            if os.path.exists(combos_file):
                with open(combos_file) as f:
                    valid_sb_combos = [tuple(c) for c in json.load(f)]
                os.remove(combos_file)
            else:
                valid_sb_combos = sb_combos

            durations = parse_ops_log(subdir, expected_ops=len(valid_sb_combos))

            if len(durations) != len(valid_sb_combos):
                logger.warning(f"  Subblock sweep: expected {len(valid_sb_combos)} ops, got {len(durations)}")

            sb_results = []
            for i, (sb_h, sb_w) in enumerate(valid_sb_combos):
                if i < len(durations):
                    duration_ns = durations[i]
                    sb_results.append({"subblock_h": sb_h, "subblock_w": sb_w, "duration_ns": duration_ns})
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
                            best_m,
                            best_k,
                            best_n,
                            sb_h,
                            sb_w,
                            f"{duration_ns:.0f}",
                            "OK_SB",
                        ],
                    )

            if sb_results:
                sb_results.sort(key=lambda r: r["duration_ns"])
                best_sb = sb_results[0]
                if best_sb["duration_ns"] < best["duration_ns"]:
                    best["subblock_h"] = best_sb["subblock_h"]
                    best["subblock_w"] = best_sb["subblock_w"]
                    best["duration_ns"] = best_sb["duration_ns"]

                logger.info("Pass 2 subblock results:")
                for rank, r in enumerate(sb_results, 1):
                    logger.info(f"  #{rank}: sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns")

        except Exception as e:
            logger.warning(f"  Subblock sweep FAILED: {str(e)[:200]}")
    else:
        logger.info("Pass 2: skipped (only one valid subblock combo)")

    # Final result
    logger.info(
        f"FINAL BEST for [{device_config}] {shape_id}: "
        f"M={best['M_block']} K={best['K_block']} N={best['N_block']} "
        f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns"
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
    parser.add_argument("--max-block", type=int, default=MAX_BLOCK, help="Max block size in tiles (default: 64)")
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

    for M, K, N, cgx, cgy, is_agmm, use_case in shapes:
        M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
        op_type = "agmm" if is_agmm else "mm"
        shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_{op_type}_{use_case}"
        core_grid_str = f"{cgx}x{cgy}"

        m_blocks = get_block_candidates(M_tiles, args.max_block)

        # Check if any M_block has valid K/N combos
        has_combos = any(
            generate_kn_combos(K_tiles, N_tiles, m_block=m, use_case=use_case, is_agmm=is_agmm) for m in m_blocks
        )
        if not has_combos:
            print(f"Skipping {shape_id}: no valid K/N combos after L1 filter")
            continue

        print(f"\n{'='*80}")
        print(
            f"[{device_config}] {op_type} ({use_case}) Shape {M}_{K}_{N} grid={core_grid_str}: "
            f"M_tiles={M_tiles} K_tiles={K_tiles} N_tiles={N_tiles}"
        )
        print(f"  {len(m_blocks)} M_blocks (divisors+base), L1-filtered K/N combos")
        print(f"{'='*80}")

        shape_results = []

        batch_size = PROFILER_BATCH_SIZE_AGMM if is_agmm else PROFILER_BATCH_SIZE_MM

        for m_block in m_blocks:
            kn_combos = generate_kn_combos(K_tiles, N_tiles, m_block=m_block, use_case=use_case, is_agmm=is_agmm)
            if not kn_combos:
                print(f"  M_block={m_block}: all K/N combos exceed L1 budget, skipping")
                continue

            # Split into batches to avoid profiler DRAM buffer overflow
            batches = [
                (b_start, min(b_start + batch_size, len(kn_combos))) for b_start in range(0, len(kn_combos), batch_size)
            ]
            n_batches = len(batches)
            batch_label = f" ({n_batches} batches)" if n_batches > 1 else ""

            print(f"  M_block={m_block}: profiling {len(kn_combos)} combos{batch_label}...", end=" ", flush=True)

            m_block_results = 0

            for batch_idx, (b_start, b_end) in enumerate(batches):
                batch_combos = kn_combos[b_start:b_end]
                batch_suffix = f"_b{batch_idx}" if n_batches > 1 else ""
                subdir = f"mm_sweep_{device_config}_{shape_id}_m{m_block}{batch_suffix}"
                combos_file = f"valid_combos_{device_config}_{shape_id}_m{m_block}{batch_suffix}.json"
                os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
                os.environ["MM_SWEEP_BATCH_START"] = str(b_start)
                os.environ["MM_SWEEP_BATCH_END"] = str(b_end)
                command = (
                    f"pytest models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py"
                    f"::test_mm_sweep_worker[m{m_block}-{shape_id}-{device_config}] -x"
                )

                try:
                    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])

                    # Read valid combos from worker first (may be subset due to runtime L1 OOM)
                    if os.path.exists(combos_file):
                        with open(combos_file) as f:
                            valid_combos = [tuple(c) for c in json.load(f)]
                        os.remove(combos_file)
                    else:
                        valid_combos = batch_combos

                    durations = parse_ops_log(subdir, expected_ops=len(valid_combos))

                    for i, (k_blk, n_blk) in enumerate(valid_combos):
                        sb_h, sb_w = pick_subblock(m_block, n_blk)
                        if i < len(durations):
                            duration_ns = durations[i]
                            shape_results.append(
                                {
                                    "M_block": m_block,
                                    "K_block": k_blk,
                                    "N_block": n_blk,
                                    "subblock_h": sb_h,
                                    "subblock_w": sb_w,
                                    "duration_ns": duration_ns,
                                }
                            )
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
                                    m_block,
                                    k_blk,
                                    n_blk,
                                    sb_h,
                                    sb_w,
                                    f"{duration_ns:.0f}",
                                    "OK",
                                ],
                            )
                            m_block_results += 1
                        else:
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
                                    m_block,
                                    k_blk,
                                    n_blk,
                                    sb_h,
                                    sb_w,
                                    -1,
                                    "MISSING",
                                ],
                            )

                except Exception as e:
                    print(f"batch {batch_idx} FAILED: {str(e)[:100]}...", end=" ", flush=True)
                    for k_blk, n_blk in batch_combos:
                        sb_h, sb_w = pick_subblock(m_block, n_blk)
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
                                m_block,
                                k_blk,
                                n_blk,
                                sb_h,
                                sb_w,
                                -1,
                                "FAIL",
                            ],
                        )

            # Clean up batch env vars
            os.environ.pop("MM_SWEEP_BATCH_START", None)
            os.environ.pop("MM_SWEEP_BATCH_END", None)
            print(f"{m_block_results} results OK")

        if shape_results:
            shape_results.sort(key=lambda r: r["duration_ns"])
            best = shape_results[0]
            print(
                f"  Pass 1 BEST: M={best['M_block']} K={best['K_block']} N={best['N_block']} "
                f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns"
            )
            print("  Pass 1 top 5:")
            for rank, r in enumerate(shape_results[:5], 1):
                print(
                    f"    #{rank}: M={r['M_block']} K={r['K_block']} N={r['N_block']} "
                    f"sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns"
                )

            # Pass 2: subblock sweep on best (M_block, K_block, N_block)
            best_m, best_k, best_n = best["M_block"], best["K_block"], best["N_block"]
            sb_combos = generate_subblock_combos(best_m, best_n)

            if len(sb_combos) > 1:
                print(
                    f"  Pass 2: sweeping {len(sb_combos)} subblock combos for "
                    f"blocks=({best_m},{best_k},{best_n})..."
                )

                os.environ["MM_SWEEP_M_BLOCK"] = str(best_m)
                os.environ["MM_SWEEP_K_BLOCK"] = str(best_k)
                os.environ["MM_SWEEP_N_BLOCK"] = str(best_n)

                subdir = f"mm_sweep_{device_config}_{shape_id}_subblock"
                combos_file = f"valid_combos_{device_config}_{shape_id}_subblock.json"
                os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
                command = (
                    f"pytest models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py"
                    f"::test_mm_subblock_sweep_worker[{shape_id}-{device_config}] -x"
                )

                try:
                    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])

                    if os.path.exists(combos_file):
                        with open(combos_file) as f:
                            valid_sb_combos = [tuple(c) for c in json.load(f)]
                        os.remove(combos_file)
                    else:
                        valid_sb_combos = sb_combos

                    durations = parse_ops_log(subdir, expected_ops=len(valid_sb_combos))

                    if len(durations) != len(valid_sb_combos):
                        print(f"  Subblock sweep: expected {len(valid_sb_combos)} ops, got {len(durations)}")

                    sb_results = []
                    for i, (sb_h, sb_w) in enumerate(valid_sb_combos):
                        if i < len(durations):
                            duration_ns = durations[i]
                            sb_results.append({"subblock_h": sb_h, "subblock_w": sb_w, "duration_ns": duration_ns})
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
                                    best_m,
                                    best_k,
                                    best_n,
                                    sb_h,
                                    sb_w,
                                    f"{duration_ns:.0f}",
                                    "OK_SB",
                                ],
                            )

                    if sb_results:
                        sb_results.sort(key=lambda r: r["duration_ns"])
                        best_sb = sb_results[0]
                        if best_sb["duration_ns"] < best["duration_ns"]:
                            best["subblock_h"] = best_sb["subblock_h"]
                            best["subblock_w"] = best_sb["subblock_w"]
                            best["duration_ns"] = best_sb["duration_ns"]

                        print("  Pass 2 subblock results:")
                        for rank, r in enumerate(sb_results, 1):
                            print(f"    #{rank}: sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns")

                except Exception as e:
                    print(f"  Subblock sweep FAILED: {str(e)[:200]}")
            else:
                print("  Pass 2: skipped (only one valid subblock combo)")

            # Final best for this shape
            print(
                f"  FINAL BEST: M={best['M_block']} K={best['K_block']} N={best['N_block']} "
                f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns"
            )
            all_best[(M, K, N, cgx, cgy, op_type, use_case)] = best

    # Print summary
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

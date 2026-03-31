# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Sweep block sizes for all_gather_minimal_matmul_async (fused AG+MM)
on Llama 70B FF2 shapes using device profiler for accurate kernel timing.

Target: Wormhole Galaxy (8x4 mesh, 32 devices)

Architecture:
  - Worker test (test_mm_sweep_worker): Self-contained profiled test.
    For a given (device_config, shape, M_block), runs all valid (K_block, N_block)
    combos, calling the fused op directly. Invoked as subprocess by device profiler.
  - Subblock sweep worker (test_mm_subblock_sweep_worker): Sweeps all valid subblock
    sizes for a fixed (M_block, K_block, N_block) configuration (Pass 2).
  - Orchestrator test (test_mm_sweep): Iterates M_blocks, spawns worker via
    run_device_profiler, parses ops log to extract per-op device kernel durations.
    Runs Pass 1 (block sweep) then Pass 2 (subblock sweep on best blocks).

Usage:
  # Orchestrator: one (device_config, shape) — ISL 4k/8k/…/128k, 6x8 grid, wh_galaxy (3 links)
  pytest "tests/ttnn/unit_tests/operations/ccl/sweep_llama70b_agmm_block_sizes.py::test_mm_sweep[4096_3584_2048_6x8_agmm-device_config=wh_galaxy]" -x -s
  pytest "tests/ttnn/unit_tests/operations/ccl/sweep_llama70b_agmm_block_sizes.py::test_mm_sweep[131072_3584_2048_6x8_agmm-device_config=wh_galaxy]" -x -s

  # Worker: one M_block slice (debug / no profiler)
  pytest "tests/ttnn/unit_tests/operations/ccl/sweep_llama70b_agmm_block_sizes.py::test_mm_sweep_worker[m8-8192_3584_2048_6x8_agmm-device_config=wh_galaxy]" -x -s

  # Standalone script
  python tests/ttnn/unit_tests/operations/ccl/sweep_llama70b_agmm_block_sizes.py \
    --device-config wh_galaxy --shape 8192,3584,2048

  # Llama-style 8×8×8 baseline (compare Pass-1/2 sweep winners vs fixed policy): see
  # tests/ttnn/unit_tests/operations/ccl/test_agmm_llama_baseline_8_8_8.py

Llama 70B FF2 shapes (per device):
  - M = seq_len (4096 / 8192 / 16384 / 32768 / 65536 / 131072 for 4k–128k ISL sweeps)
  - K = 3584 (intermediate_dim / TP / 2 = 28672 / 4 / 2, gated MLP down projection)
  - N = 2048 (hidden_dim / TP = 8192 / 4)
  - All-gather on K dimension: 896 per device -> 3584 after ring of 4
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
    "wh_galaxy": {
        "mesh_shape": (8, 4),
        "fabric_config": "FABRIC_1D_RING",
        "fabric_router_config_payload": 7168,
        "topology": "Ring",
        "num_links": 3,
        "num_workers_per_link": 2,
        "cluster_axis": 1,
        "ring_size": 4,
    },
    # 7-wide compute grid: only num_links=1 divides 7 (force_transpose uses grid.x for links).
    "wh_galaxy_7col": {
        "mesh_shape": (8, 4),
        "fabric_config": "FABRIC_1D_RING",
        "fabric_router_config_payload": 7168,
        "topology": "Ring",
        "num_links": 1,
        "num_workers_per_link": 7,
        "cluster_axis": 1,
        "ring_size": 4,
    },
}

DEFAULT_DEVICE_CONFIG = "wh_galaxy"


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
        "cluster_axis": cfg["cluster_axis"],
        "ring_size": cfg["ring_size"],
    }


# ============================================================================
# SHAPE TABLE - Llama 70B FF2 shapes
# ============================================================================


# (M, K, N, core_grid_x, core_grid_y)
# M, K, N are the matmul dimensions as seen by the kernel (per-device).
# Core grid matches what the model passes to MinimalMatmulConfig at runtime.
# Sweep coverage: 4k / 8k / 16k / 32k / 64k / 128k ISL, all 6x8 + wh_galaxy (3 links, 2 workers/link).
# Large M: DRAM scales ~M×K; 128k is ~2× 64k activation footprint — watch OOM and host RAM.
# M_block=1 is skipped (SWEEP_MIN_M_BLOCK_TILES) to avoid fabric/router issues seen on fused AGMM.
def _shape_tuple_id(shape):
    M, K, N, cgx, cgy = shape
    return f"{M}_{K}_{N}_{cgx}x{cgy}_agmm"


# How the sweep runs (short):
#   • Fixed per row: K=3584, N=2048, compute grid 6×8, device_config wh_galaxy (unless you add pairs).
#   • M is one of SWEEP_PAIRS (4k…128k rows). Tile counts: M_tiles=ceil(M/32), K_tiles=K/32, N_tiles=N/32.
#   Pass 1 — block sweep:
#   • M_block: every divisor of M_tiles with 2 <= M_block <= 64 (skip 1).
#   • (K_block, N_block): every pair of divisors of K_tiles and N_tiles (each <= 64), with K_block <= K_tiles/ring_size.
#   • For each combo, device profiler measures fused op; CSV logs pick_subblock(M_block, N_block) (max h×w<=4).
#   Pass 2 — subblock sweep:
#   • Take best (M_block, K_block, N_block) from Pass 1; try every (subblock_h, subblock_w) with h|M_block,
#     w|N_block, h*w<=4; profiler measures each.

# (device_config_name, (M, K, N, core_x, core_y))
SWEEP_PAIRS = [
    ("wh_galaxy", (4096, 3584, 2048, 6, 8)),
    ("wh_galaxy", (8192, 3584, 2048, 6, 8)),
    ("wh_galaxy", (16384, 3584, 2048, 6, 8)),
    ("wh_galaxy", (32768, 3584, 2048, 6, 8)),
    ("wh_galaxy", (65536, 3584, 2048, 6, 8)),
    ("wh_galaxy", (131072, 3584, 2048, 6, 8)),
]

SWEEP_PAIR_IDS = [f"{_shape_tuple_id(s)}-device_config={dc}" for dc, s in SWEEP_PAIRS]

# Unique shapes for CLI / filters (stable order)
SHAPES = []
_seen_shapes = set()
for _, s in SWEEP_PAIRS:
    if s not in _seen_shapes:
        _seen_shapes.add(s)
        SHAPES.append(s)
SHAPE_IDS = [_shape_tuple_id(s) for s in SHAPES]

# Block sweep range
MAX_BLOCK = 64
# M_block=1 (smallest divisor) has caused fabric router sync / apparent deadlocks on large
# fused AGMM; remaining M_blocks then fail in a bad mesh state. Skip for sweeps.
SWEEP_MIN_M_BLOCK_TILES = 2

CSV_FILE = "sweep_results_llama70b_mm.csv"
CSV_COLUMNS = [
    "device_config",
    "op_type",
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


def generate_kn_combos(K_tiles, N_tiles, ring_size=4):
    """Generate all valid (K_block, N_block) divisor combos in [1, MAX_BLOCK].

    Filters out K_block values that would cause division by zero in the kernel
    (K_block must be <= K_tiles / ring_size to avoid K_blocks_per_device = 0).
    """
    K_blocks_per_device = K_tiles // ring_size
    k_divs = [k for k in get_divisors(K_tiles) if k <= K_blocks_per_device]
    n_divs = get_divisors(N_tiles)
    return [(k, n) for k in k_divs for n in n_divs]


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


def open_mesh(cfg):
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
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))


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


def parse_ops_log(subdir, num_devices=32):
    """Parse device profiler ops log between start/stop signposts.

    Returns list of per-op max device kernel durations (ns), one per op dispatch.
    Groups by row order (every num_devices consecutive rows = one op) since
    GLOBAL CALL COUNT is unique per device-row in multi-device profiling.
    """
    import pandas as pd
    import numpy as np
    from tracy.process_model_log import get_latest_ops_log_filename

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    signpost_rows = df[df["OP TYPE"] == "signpost"]
    start_markers = signpost_rows[signpost_rows["OP CODE"] == "start"]
    stop_markers = signpost_rows[signpost_rows["OP CODE"] == "stop"]

    if start_markers.empty or stop_markers.empty:
        logger.warning(f"No start/stop signposts found in {filename}")
        return []

    start_idx = start_markers.index[0]
    stop_idx = stop_markers.index[0]
    df = df.iloc[start_idx + 1 : stop_idx]

    df = df[df["OP TYPE"] != "signpost"]
    df = df[df["DEVICE KERNEL DURATION [ns]"] != "-"]
    if df.empty:
        return []

    df = df.copy()
    df["DEVICE KERNEL DURATION [ns]"] = df["DEVICE KERNEL DURATION [ns]"].astype(float)

    # Group by row order: every num_devices consecutive rows is one op
    # This handles multi-device profiling where each device logs separately
    df["op_index"] = np.arange(len(df)) // num_devices
    per_op = df.groupby("op_index", sort=True)["DEVICE KERNEL DURATION [ns]"].max()

    return per_op.values.tolist()


def create_global_semaphores(mesh_device, num_devices, core_range_set, initial_value=0):
    """Create global semaphores for CCL operations."""
    return [ttnn.create_global_semaphore(mesh_device, core_range_set, initial_value) for _ in range(num_devices)]


def _mm_sweep_worker_param_list():
    """Stable pytest ids for orchestrator: m{m}-{shape_id}-device_config={dc}."""
    out = []
    for m_block in range(SWEEP_MIN_M_BLOCK_TILES, MAX_BLOCK + 1):
        for device_config, shape in SWEEP_PAIRS:
            shape_id = _shape_tuple_id(shape)
            pid = f"m{m_block}-{shape_id}-device_config={device_config}"
            out.append(pytest.param(device_config, shape, m_block, id=pid))
    return out


def _setup_agmm_tensors(mesh_device, cfg, M, K, N, core_grid):
    """Create input, weight, bias, persistent_output_buffer, and semaphores for AGMM.

    For Llama 70B FF2 on 8x4 Galaxy mesh with cluster_axis=1:
    - The all-gather ring is along mesh axis 1 (4 devices per ring)
    - M is the FULL per-device sequence length (8192) - NOT sharded across rows
    - K_per_device = K / ring_size = 3584 / 4 = 896 (sharded across TP ring)
    - All-gather combines K: (M, K_per_device) -> (M, K) = (8192, 896) -> (8192, 3584)
    - Matmul: (M, K) @ (K, N) = (8192, 3584) @ (3584, 2048) -> (8192, 2048)

    Per-device shapes matching Llama profiler:
    - Input: [8192, 896]
    - Weight: [3584, 2048]
    - Output: [8192, 2048]
    """
    dtype = ttnn.bfloat8_b
    ring_size = cfg["ring_size"]
    cluster_axis = cfg["cluster_axis"]

    # Input tensor: create FULL shape (M, K) = (8192, 3584)
    # ShardTensor2dMesh with dims=[None, cluster_axis] will:
    # - dim 0 (M=8192): None = not sharded, replicated across rows
    # - dim 1 (K=3584): sharded across cluster_axis (axis 1, 4 devices) -> 3584/4 = 896 per device
    # Result per device: (8192, 896) - matches Llama profiler
    tt_input = ttnn.from_torch(
        torch.randn((M, K), dtype=torch.float32),  # Full shape (8192, 3584)
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, cluster_axis]),
    )

    # Weight and bias are replicated to all devices (no mesh_mapper needed)
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

    # Persistent output buffer for all-gather result: (M, K) per device
    persistent_output_buffer = ttnn.from_torch(
        torch.zeros((M, K), dtype=torch.float32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
    )

    ccl_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )
    ccl_semaphore_handles = create_global_semaphores(mesh_device, mesh_device.get_num_devices(), ccl_cores, 0)

    return tt_input, tt_weight, tt_bias, persistent_output_buffer, ccl_semaphore_handles


def _run_agmm_op(
    tt_input,
    tt_weight,
    tt_bias,
    persistent_output_buffer,
    ccl_semaphore_handles,
    compute_config,
    matmul_config,
    cfg,
    mesh_device,
):
    """Execute one all_gather_minimal_matmul_async call."""
    ttnn.experimental.all_gather_minimal_matmul_async(
        tt_input,
        tt_weight,
        bias_tensor=tt_bias,
        fused_activation=None,
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
        num_buffers_per_channel=8,
        scalar=None,
        addcmul_input_tensor1=None,
        addcmul_input_tensor2=None,
        chunks=1,
    )
    ttnn.synchronize_device(mesh_device)


# ============================================================================
# WORKER TEST — profiled in subprocess by device profiler
# ============================================================================


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_config,shape,m_block", _mm_sweep_worker_param_list())
def test_mm_sweep_worker(device_config, shape, m_block):
    """Run all (K_block, N_block) combos for a given device config, shape, and M_block.

    Designed to be invoked via run_device_profiler as a subprocess.
    Emits start/stop signposts around the measured region.
    """
    cfg = resolve_config(device_config)
    M, K, N, cgx, cgy = shape

    M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)

    if M_tiles % m_block != 0:
        pytest.skip(f"m_block={m_block} doesn't divide M_tiles={M_tiles}")

    kn_combos = generate_kn_combos(K_tiles, N_tiles, cfg["ring_size"])
    if not kn_combos:
        pytest.skip("No valid (K_block, N_block) combos")

    logger.info(
        f"Worker [{device_config}] agmm: M={M} K={K} N={N} grid={cgx}x{cgy} "
        f"m_block={m_block}, {len(kn_combos)} K/N combos"
    )

    mesh_device = open_mesh(cfg)
    try:
        core_grid = ttnn.CoreCoord(cgx, cgy)

        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        tt_input, tt_weight, tt_bias, persistent_output_buffer, ccl_semaphore_handles = _setup_agmm_tensors(
            mesh_device, cfg, M, K, N, core_grid
        )

        def run_op(k_blk, n_blk):
            sb_h, sb_w = pick_subblock(m_block, n_blk)
            matmul_config = ttnn.MinimalMatmulConfig(
                M_block_size=m_block,
                K_block_size=k_blk,
                N_block_size=n_blk,
                subblock_h=sb_h,
                subblock_w=sb_w,
                compute_with_storage_grid_size=core_grid,
            )
            _run_agmm_op(
                tt_input,
                tt_weight,
                tt_bias,
                persistent_output_buffer,
                ccl_semaphore_handles,
                compute_config,
                matmul_config,
                cfg,
                mesh_device,
            )

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

        # Measured run -- only valid combos
        from tracy import signpost

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


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_config,shape", SWEEP_PAIRS, ids=SWEEP_PAIR_IDS)
def test_mm_subblock_sweep_worker(device_config, shape):
    """Sweep subblock sizes for fixed (M_block, K_block, N_block) read from env vars.

    Designed to be invoked by the orchestrator's pass 2 via run_device_profiler.
    Block sizes are passed via MM_SWEEP_M_BLOCK, MM_SWEEP_K_BLOCK, MM_SWEEP_N_BLOCK env vars.
    """
    m_block = int(os.environ["MM_SWEEP_M_BLOCK"])
    k_block = int(os.environ["MM_SWEEP_K_BLOCK"])
    n_block = int(os.environ["MM_SWEEP_N_BLOCK"])

    cfg = resolve_config(device_config)
    M, K, N, cgx, cgy = shape

    sb_combos = generate_subblock_combos(m_block, n_block)
    if len(sb_combos) <= 1:
        pytest.skip("Only one valid subblock combo")

    logger.info(
        f"Subblock worker [{device_config}]: M={M} K={K} N={N} "
        f"blocks=({m_block},{k_block},{n_block}), {len(sb_combos)} subblock combos"
    )

    mesh_device = open_mesh(cfg)
    try:
        core_grid = ttnn.CoreCoord(cgx, cgy)

        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        tt_input, tt_weight, tt_bias, persistent_output_buffer, ccl_semaphore_handles = _setup_agmm_tensors(
            mesh_device, cfg, M, K, N, core_grid
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
            _run_agmm_op(
                tt_input,
                tt_weight,
                tt_bias,
                persistent_output_buffer,
                ccl_semaphore_handles,
                compute_config,
                matmul_config,
                cfg,
                mesh_device,
            )

        # Warmup -- skip combos that OOM
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


@pytest.mark.timeout(21600)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance sweep - skip on CI")
@pytest.mark.parametrize("device_config,shape", SWEEP_PAIRS, ids=SWEEP_PAIR_IDS)
def test_mm_sweep(device_config, shape):
    """Orchestrate the block size sweep for one (device_config, shape).

    For each valid M_block, invokes test_mm_sweep_worker via the device profiler
    as a subprocess, then parses the ops log to extract device kernel durations.
    After Pass 1, runs Pass 2 subblock sweep on the best block config.
    """
    from tracy.process_model_log import run_device_profiler

    cfg = resolve_config(device_config)
    M, K, N, cgx, cgy = shape
    M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
    shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_agmm"
    core_grid_str = f"{cgx}x{cgy}"

    m_blocks = [b for b in get_divisors(M_tiles) if b >= SWEEP_MIN_M_BLOCK_TILES]
    kn_combos = generate_kn_combos(K_tiles, N_tiles, cfg["ring_size"])

    if not kn_combos:
        pytest.skip(f"No valid (K_block, N_block) combos for K_tiles={K_tiles}, N_tiles={N_tiles}")
    if not m_blocks:
        pytest.skip(f"No M_block >= {SWEEP_MIN_M_BLOCK_TILES} divides M_tiles={M_tiles}")

    logger.info(
        f"Sweep [{device_config}] agmm {shape_id}: M_tiles={M_tiles}, K_tiles={K_tiles}, N_tiles={N_tiles}, "
        f"{len(m_blocks)} M_blocks (>={SWEEP_MIN_M_BLOCK_TILES}), {len(kn_combos)} K/N combos each"
    )

    write_csv_header(CSV_FILE)
    all_results = []

    for m_block in m_blocks:
        subdir = f"mm_sweep_{device_config}_{shape_id}_m{m_block}"
        combos_file = f"valid_combos_{device_config}_{shape_id}_m{m_block}.json"
        os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
        command = (
            f"pytest tests/ttnn/unit_tests/operations/ccl/sweep_llama70b_agmm_block_sizes.py"
            f"::test_mm_sweep_worker[m{m_block}-{shape_id}-device_config={device_config}] -x"
        )

        logger.info(f"  M_block={m_block}: profiling {len(kn_combos)} combos...")

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            durations = parse_ops_log(subdir)

            # Read valid combos from worker (may be subset of kn_combos due to L1 OOM)
            if os.path.exists(combos_file):
                with open(combos_file) as f:
                    valid_combos = [tuple(c) for c in json.load(f)]
                os.remove(combos_file)
            else:
                valid_combos = kn_combos

            if len(durations) != len(valid_combos):
                logger.warning(
                    f"  M_block={m_block}: expected {len(valid_combos)} ops, " f"got {len(durations)} in profiler log"
                )

            skipped = len(kn_combos) - len(valid_combos)
            if skipped:
                logger.info(f"  M_block={m_block}: {skipped} combos skipped (L1 OOM)")

            for i, (k_blk, n_blk) in enumerate(valid_combos):
                sb_h, sb_w = pick_subblock(m_block, n_blk)
                if i < len(durations):
                    duration_ns = durations[i]
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
                        "agmm",
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

            logger.info(f"  M_block={m_block}: done, " f"{min(len(durations), len(valid_combos))} results recorded")

        except Exception as e:
            err_msg = str(e)
            if len(err_msg) > 200:
                err_msg = err_msg[:200] + "..."
            logger.warning(f"  M_block={m_block}: FAILED - {err_msg}")
            for k_blk, n_blk in kn_combos:
                sb_h, sb_w = pick_subblock(m_block, n_blk)
                append_csv_row(
                    CSV_FILE,
                    [
                        device_config,
                        "agmm",
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
    best_m, best_k, best_n = best["M_block"], best["K_block"], best["N_block"]
    sb_combos = generate_subblock_combos(best_m, best_n)

    if len(sb_combos) > 1:
        logger.info(f"Pass 2: sweeping {len(sb_combos)} subblock combos for " f"blocks=({best_m},{best_k},{best_n})...")

        os.environ["MM_SWEEP_M_BLOCK"] = str(best_m)
        os.environ["MM_SWEEP_K_BLOCK"] = str(best_k)
        os.environ["MM_SWEEP_N_BLOCK"] = str(best_n)

        subdir = f"mm_sweep_{device_config}_{shape_id}_subblock"
        combos_file = f"valid_combos_{device_config}_{shape_id}_subblock.json"
        os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
        command = (
            f"pytest tests/ttnn/unit_tests/operations/ccl/sweep_llama70b_agmm_block_sizes.py"
            f"::test_mm_subblock_sweep_worker[{shape_id}-device_config={device_config}] -x"
        )

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            durations = parse_ops_log(subdir)

            if os.path.exists(combos_file):
                with open(combos_file) as f:
                    valid_sb_combos = [tuple(c) for c in json.load(f)]
                os.remove(combos_file)
            else:
                valid_sb_combos = sb_combos

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
                            "agmm",
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

    parser = argparse.ArgumentParser(
        description="Sweep AGMM block sizes (4k–128k ISL, 6x8, wh_galaxy / 3 links; see SWEEP_PAIRS)"
    )
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
        help="Filter to a single shape as M,K,N (e.g. 8192,3584,2048)",
    )
    parser.add_argument("--csv", type=str, default=CSV_FILE)
    parser.add_argument("--max-block", type=int, default=MAX_BLOCK, help="Max block size in tiles (default: 64)")
    args = parser.parse_args()

    device_config = args.device_config
    cfg = resolve_config(device_config)

    # Pairs matching this device config (sweep uses wh_galaxy for all listed shapes)
    pairs = [(dc, s) for dc, s in SWEEP_PAIRS if dc == device_config]
    if args.shape:
        m, k, n = [int(x) for x in args.shape.split(",")]
        pairs = [(dc, s) for dc, s in pairs if s[0] == m and s[1] == k and s[2] == n]
        if not pairs:
            print(f"Shape {args.shape} not found for device_config={device_config} in SWEEP_PAIRS")
            return

    write_csv_header(args.csv)
    all_best = {}

    print(
        f"Device config: {device_config} (mesh={cfg['mesh_shape']}, "
        f"links={cfg['num_links']}, ring_size={cfg['ring_size']})"
    )

    for dc_loop, (M, K, N, cgx, cgy) in pairs:
        assert dc_loop == device_config
        M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
        shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_agmm"
        core_grid_str = f"{cgx}x{cgy}"

        m_blocks = [b for b in get_divisors(M_tiles, args.max_block) if b >= SWEEP_MIN_M_BLOCK_TILES]
        kn_combos = generate_kn_combos(K_tiles, N_tiles, cfg["ring_size"])

        if not kn_combos:
            print(f"Skipping {shape_id}: no valid K/N combos")
            continue
        if not m_blocks:
            print(f"Skipping {shape_id}: no M_block >= {SWEEP_MIN_M_BLOCK_TILES}")
            continue

        print(f"\n{'='*80}")
        print(
            f"[{device_config}] agmm Shape {M}_{K}_{N} grid={core_grid_str}: "
            f"M_tiles={M_tiles} K_tiles={K_tiles} N_tiles={N_tiles}"
        )
        print(f"  {len(m_blocks)} M_blocks x {len(kn_combos)} K/N combos")
        print(f"{'='*80}")

        shape_results = []

        for m_block in m_blocks:
            subdir = f"mm_sweep_{device_config}_{shape_id}_m{m_block}"
            combos_file = f"valid_combos_{device_config}_{shape_id}_m{m_block}.json"
            os.environ["MM_SWEEP_VALID_COMBOS_FILE"] = combos_file
            command = (
                f"pytest tests/ttnn/unit_tests/operations/ccl/sweep_llama70b_agmm_block_sizes.py"
                f"::test_mm_sweep_worker[m{m_block}-{shape_id}-device_config={device_config}] -x"
            )

            print(f"  M_block={m_block}: profiling {len(kn_combos)} combos...", end=" ", flush=True)

            try:
                run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
                durations = parse_ops_log(subdir)

                # Read valid combos from worker (may be subset due to L1 OOM)
                if os.path.exists(combos_file):
                    with open(combos_file) as f:
                        valid_combos = [tuple(c) for c in json.load(f)]
                    os.remove(combos_file)
                else:
                    valid_combos = kn_combos

                skipped = len(kn_combos) - len(valid_combos)
                matched = min(len(durations), len(valid_combos))
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
                                "agmm",
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
                    else:
                        append_csv_row(
                            args.csv,
                            [
                                device_config,
                                "agmm",
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

                suffix = f" ({skipped} skipped L1 OOM)" if skipped else ""
                print(f"{matched} results OK{suffix}")

            except Exception as e:
                print(f"FAILED: {str(e)[:100]}")
                for k_blk, n_blk in kn_combos:
                    sb_h, sb_w = pick_subblock(m_block, n_blk)
                    append_csv_row(
                        args.csv,
                        [
                            device_config,
                            "agmm",
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
                    f"pytest tests/ttnn/unit_tests/operations/ccl/sweep_llama70b_agmm_block_sizes.py"
                    f"::test_mm_subblock_sweep_worker[{shape_id}-device_config={device_config}] -x"
                )

                try:
                    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
                    durations = parse_ops_log(subdir)

                    if os.path.exists(combos_file):
                        with open(combos_file) as f:
                            valid_sb_combos = [tuple(c) for c in json.load(f)]
                        os.remove(combos_file)
                    else:
                        valid_sb_combos = sb_combos

                    if len(durations) != len(valid_sb_combos):
                        print(f"    Subblock sweep: expected {len(valid_sb_combos)} ops, got {len(durations)}")

                    sb_results = []
                    for i, (sb_h, sb_w) in enumerate(valid_sb_combos):
                        if i < len(durations):
                            duration_ns = durations[i]
                            sb_results.append({"subblock_h": sb_h, "subblock_w": sb_w, "duration_ns": duration_ns})
                            append_csv_row(
                                args.csv,
                                [
                                    device_config,
                                    "agmm",
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

                        print("    Pass 2 subblock results:")
                        for rank, r in enumerate(sb_results, 1):
                            print(
                                f"      #{rank}: sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns"
                            )

                except Exception as e:
                    print(f"    Subblock sweep FAILED: {str(e)[:200]}")
            else:
                print("  Pass 2: skipped (only one valid subblock combo)")

            # Final best for this shape
            print(
                f"  FINAL BEST: M={best['M_block']} K={best['K_block']} N={best['N_block']} "
                f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns"
            )
            all_best[(M, K, N, cgx, cgy)] = best

    # Print summary
    if all_best:
        print(f"\n{'='*100}")
        print(f"SWEEP SUMMARY [{device_config}] - Best configs per shape")
        print(f"{'='*100}")
        print(
            f"{'M':>6} {'K':>6} {'N':>6} {'grid':>7} | "
            f"{'M_blk':>5} {'K_blk':>5} {'N_blk':>5} {'sb_h':>4} {'sb_w':>4} | "
            f"{'duration_ns':>12} {'duration_us':>12}"
        )
        print("-" * 100)
        for (M, K, N, cgx, cgy), best in sorted(all_best.items()):
            print(
                f"{M:>6} {K:>6} {N:>6} {cgx}x{cgy:>2} | "
                f"{best['M_block']:>5} {best['K_block']:>5} {best['N_block']:>5} "
                f"{best['subblock_h']:>4} {best['subblock_w']:>4} | "
                f"{best['duration_ns']:>12.0f} {best['duration_ns']/1000:>12.2f}"
            )
        print(f"{'='*100}")
        print(f"\nFull results written to: {args.csv}")


if __name__ == "__main__":
    main()

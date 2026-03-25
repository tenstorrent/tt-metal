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
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import create_global_semaphores

# ============================================================================
# DEVICE CONFIGURATIONS
# ============================================================================
# To add a new config: add an entry here with the required fields.
# The worker and orchestrator will pick it up automatically.

DEVICE_CONFIGS = {
    "bh_4x8": {
        "mesh_shape": (4, 8),
        "fabric_config": "FABRIC_1D_RING",
        "fabric_router_config_payload": 4096,  # max_packet_payload_size_bytes
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

# (M, K, N, core_grid_x, core_grid_y, is_agmm)
# M, K, N are the matmul dimensions as seen by the kernel (per-device).
# Core grid matches what the model passes to get_matmul_config at runtime.
# For is_agmm=False: calls ttnn.experimental.minimal_matmul
# For is_agmm=True: calls ttnn.experimental.all_gather_minimal_matmul_async
SHAPES = [
    (96, 96, 192, 11, 10, False),
    (64, 192, 384, 11, 10, False),
    (64, 96, 192, 11, 10, False),
    (32, 96, 192, 11, 10, False),
    (32, 192, 384, 11, 10, False),
    (32, 256, 5120, 11, 10, False),
    (32, 32, 32, 11, 10, False),
    (32, 1280, 30720, 11, 10, False),
    (32, 3072, 10240, 11, 10, False),
    (32, 5120, 1280, 11, 10, False),
    (32, 10240, 10240, 11, 10, False),
    (128, 5120, 2560, 11, 10, False),
    (512, 4096, 5120, 11, 10, False),
    (512, 5120, 5120, 11, 10, False),
    (6144, 384, 384, 11, 10, False),
    (6144, 384, 1152, 11, 10, False),
    (6144, 3456, 5120, 11, 10, False),
    (6144, 5120, 64, 11, 10, False),
    (6144, 5120, 1280, 12, 9, False),
    (6144, 5120, 3456, 11, 10, False),
    (6144, 5120, 3840, 12, 9, False),
    (6240, 384, 384, 11, 10, False),
    (6240, 384, 1152, 11, 10, False),
    (6240, 3456, 5120, 11, 10, False),
    (6240, 5120, 64, 11, 10, False),
    (6240, 5120, 1280, 12, 9, False),
    (6240, 5120, 3456, 11, 10, False),
    (6240, 5120, 3840, 12, 9, False),
    (14400, 384, 384, 11, 10, False),
    (14400, 384, 1152, 11, 10, False),
    (14400, 3456, 5120, 11, 10, False),
    (14400, 5120, 64, 11, 10, False),
    (14400, 5120, 1280, 12, 9, False),
    (14400, 5120, 3456, 11, 10, False),
    (14400, 5120, 3840, 12, 9, False),
]

SHAPE_IDS = [f"{M}_{K}_{N}_{cgx}x{cgy}_{'agmm' if agmm else 'mm'}" for M, K, N, cgx, cgy, agmm in SHAPES]

# Block sweep range
MAX_BLOCK = 16

CSV_FILE = "sweep_results_mm.csv"
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


def generate_kn_combos(K_tiles, N_tiles):
    """Generate all valid (K_block, N_block) divisor combos in [1, MAX_BLOCK]."""
    k_divs = get_divisors(K_tiles)
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


def parse_ops_log(subdir):
    """Parse device profiler ops log between start/stop signposts.

    Returns list of per-op max device kernel durations (ns), one per op dispatch.
    Groups by GLOBAL CALL COUNT to handle multi-device rows.
    """
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
        per_op = df.groupby("GLOBAL CALL COUNT", sort=False)["DEVICE KERNEL DURATION [ns]"].max()
        return per_op.values.tolist()
    else:
        return df["DEVICE KERNEL DURATION [ns]"].values.tolist()


# ============================================================================
# WORKER TEST — profiled in subprocess by device profiler
# ============================================================================


@pytest.mark.parametrize("device_config", list(DEVICE_CONFIGS.keys()))
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("m_block", range(1, MAX_BLOCK + 1), ids=[f"m{i}" for i in range(1, MAX_BLOCK + 1)])
def test_mm_sweep_worker(device_config, shape, m_block):
    """Run all (K_block, N_block) combos for a given device config, shape, and M_block.

    Designed to be invoked via run_device_profiler as a subprocess.
    Emits start/stop signposts around the measured region.
    """
    cfg = resolve_config(device_config)
    M, K, N, cgx, cgy, is_agmm = shape

    M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)

    if M_tiles % m_block != 0:
        pytest.skip(f"m_block={m_block} doesn't divide M_tiles={M_tiles}")

    kn_combos = generate_kn_combos(K_tiles, N_tiles)
    if not kn_combos:
        pytest.skip("No valid (K_block, N_block) combos")

    op_type = "agmm" if is_agmm else "mm"
    logger.info(
        f"Worker [{device_config}] {op_type}: M={M} K={K} N={N} grid={cgx}x{cgy} "
        f"m_block={m_block}, {len(kn_combos)} K/N combos"
    )

    mesh_device = open_mesh(cfg)
    try:
        core_grid = ttnn.CoreCoord(cgx, cgy)
        dtype = ttnn.bfloat16

        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

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

            ccl_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
            )
            ccl_semaphore_handles = create_global_semaphores(mesh_device, mesh_device.get_num_devices(), ccl_cores, 0)

            persistent_output_buffer = ttnn.from_torch(
                torch.zeros((M, K), dtype=torch.float32),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
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
                    num_buffers_per_channel=48,
                    scalar=None,
                    addcmul_input_tensor1=None,
                    addcmul_input_tensor2=None,
                    chunks=1,
                )
                ttnn.synchronize_device(mesh_device)

        else:
            # ----- Non-AGMM path: replicated tensors + minimal_matmul -----
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
                ttnn.experimental.minimal_matmul(
                    input_tensor=tt_input,
                    weight_tensor=tt_weight,
                    bias_tensor=tt_bias,
                    config=matmul_config,
                    fused_activation=None,
                    compute_kernel_config=compute_config,
                )
                ttnn.synchronize_device(mesh_device)

        # Warmup: compile all programs (outside signpost region)
        for k_blk, n_blk in kn_combos:
            run_op(k_blk, n_blk)
        logger.info("Warmup done, starting measured run")

        # Measured run
        from tracy import signpost

        signpost("start")
        for k_blk, n_blk in kn_combos:
            run_op(k_blk, n_blk)
        signpost("stop")

        logger.info(f"Worker done: {len(kn_combos)} combos measured")

    finally:
        close_mesh(mesh_device)


# ============================================================================
# ORCHESTRATOR TEST — iterates M_blocks, invokes worker via device profiler
# ============================================================================


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
    M, K, N, cgx, cgy, is_agmm = shape
    M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
    op_type = "agmm" if is_agmm else "mm"
    shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_{op_type}"
    core_grid_str = f"{cgx}x{cgy}"

    m_blocks = get_divisors(M_tiles)
    kn_combos = generate_kn_combos(K_tiles, N_tiles)

    if not kn_combos:
        pytest.skip(f"No valid (K_block, N_block) combos for K_tiles={K_tiles}, N_tiles={N_tiles}")

    logger.info(
        f"Sweep [{device_config}] {op_type} {shape_id}: M_tiles={M_tiles}, K_tiles={K_tiles}, N_tiles={N_tiles}, "
        f"{len(m_blocks)} M_blocks, {len(kn_combos)} K/N combos each"
    )

    write_csv_header(CSV_FILE)
    all_results = []

    for m_block in m_blocks:
        subdir = f"mm_sweep_{device_config}_{shape_id}_m{m_block}"
        command = (
            f"pytest models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py"
            f"::test_mm_sweep_worker[m{m_block}-{shape_id}-{device_config}] -x"
        )

        logger.info(f"  M_block={m_block}: profiling {len(kn_combos)} combos...")

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            durations = parse_ops_log(subdir)

            if len(durations) != len(kn_combos):
                logger.warning(
                    f"  M_block={m_block}: expected {len(kn_combos)} ops, " f"got {len(durations)} in profiler log"
                )

            for i, (k_blk, n_blk) in enumerate(kn_combos):
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
                        op_type,
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

            logger.info(f"  M_block={m_block}: done, " f"{min(len(durations), len(kn_combos))} results recorded")

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
                        op_type,
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

    # Print summary of best configs
    if all_results:
        all_results.sort(key=lambda r: r["duration_ns"])
        best = all_results[0]
        logger.info(
            f"BEST for [{device_config}] {shape_id}: "
            f"M={best['M_block']} K={best['K_block']} N={best['N_block']} "
            f"sb_h={best['subblock_h']} sb_w={best['subblock_w']} "
            f"-> {best['duration_ns']:.0f} ns"
        )
        logger.info("Top 5 configs:")
        for rank, r in enumerate(all_results[:5], 1):
            logger.info(
                f"  #{rank}: M={r['M_block']} K={r['K_block']} N={r['N_block']} "
                f"sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns"
            )
    else:
        logger.warning(f"No valid results for [{device_config}] {shape_id}")


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
    parser.add_argument("--max-block", type=int, default=MAX_BLOCK, help="Max block size in tiles (default: 16)")
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

    for M, K, N, cgx, cgy, is_agmm in shapes:
        M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
        op_type = "agmm" if is_agmm else "mm"
        shape_id = f"{M}_{K}_{N}_{cgx}x{cgy}_{op_type}"
        core_grid_str = f"{cgx}x{cgy}"

        m_blocks = get_divisors(M_tiles, args.max_block)
        kn_combos = generate_kn_combos(K_tiles, N_tiles)

        if not kn_combos:
            print(f"Skipping {shape_id}: no valid K/N combos")
            continue

        print(f"\n{'='*80}")
        print(
            f"[{device_config}] {op_type} Shape {M}_{K}_{N} grid={core_grid_str}: "
            f"M_tiles={M_tiles} K_tiles={K_tiles} N_tiles={N_tiles}"
        )
        print(f"  {len(m_blocks)} M_blocks x {len(kn_combos)} K/N combos")
        print(f"{'='*80}")

        shape_results = []

        for m_block in m_blocks:
            subdir = f"mm_sweep_{device_config}_{shape_id}_m{m_block}"
            command = (
                f"pytest models/tt_dit/tests/models/wan2_2/sweep_mm_block_sizes.py"
                f"::test_mm_sweep_worker[m{m_block}-{shape_id}-{device_config}] -x"
            )

            print(f"  M_block={m_block}: profiling {len(kn_combos)} combos...", end=" ", flush=True)

            try:
                run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
                durations = parse_ops_log(subdir)

                matched = min(len(durations), len(kn_combos))
                for i, (k_blk, n_blk) in enumerate(kn_combos):
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
                                op_type,
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

                print(f"{matched} results OK")

            except Exception as e:
                print(f"FAILED: {str(e)[:100]}")
                for k_blk, n_blk in kn_combos:
                    sb_h, sb_w = pick_subblock(m_block, n_blk)
                    append_csv_row(
                        args.csv,
                        [
                            device_config,
                            op_type,
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
            all_best[(M, K, N, cgx, cgy, op_type)] = shape_results[0]
            best = shape_results[0]
            print(
                f"  BEST: M={best['M_block']} K={best['K_block']} N={best['N_block']} "
                f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns"
            )

    # Print summary
    if all_best:
        print(f"\n{'='*100}")
        print(f"SWEEP SUMMARY [{device_config}] - Best configs per shape")
        print(f"{'='*100}")
        print(
            f"{'type':>4} {'M':>6} {'K':>6} {'N':>6} {'grid':>7} | "
            f"{'M_blk':>5} {'K_blk':>5} {'N_blk':>5} {'sb_h':>4} {'sb_w':>4} | "
            f"{'duration_ns':>12}"
        )
        print("-" * 100)
        for (M, K, N, cgx, cgy, op_type), best in sorted(all_best.items()):
            print(
                f"{op_type:>4} {M:>6} {K:>6} {N:>6} {cgx}x{cgy:>2} | "
                f"{best['M_block']:>5} {best['K_block']:>5} {best['N_block']:>5} "
                f"{best['subblock_h']:>4} {best['subblock_w']:>4} | "
                f"{best['duration_ns']:>12.0f}"
            )
        print(f"{'='*100}")
        print(f"\nFull results written to: {args.csv}")


if __name__ == "__main__":
    main()

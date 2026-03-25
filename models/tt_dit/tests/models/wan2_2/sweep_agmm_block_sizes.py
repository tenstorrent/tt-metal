# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Sweep block sizes for ttnn.experimental.all_gather_minimal_matmul_async
using device profiler for accurate kernel timing.

Architecture:
  - Worker test (test_agmm_sweep_worker): Self-contained profiled test.
    For a given (shape, M_block), runs all valid (K_block, N_block) combos,
    calling the fused op directly. Invoked as a subprocess by the device profiler.
  - Orchestrator test (test_agmm_sweep): Iterates M_blocks, spawns worker via
    run_device_profiler, parses ops log to extract per-op device kernel durations.

Usage:
    # Orchestrator: sweep one shape (invokes device profiler per M_block)
    pytest models/tt_dit/tests/models/wan2_2/sweep_agmm_block_sizes.py::test_agmm_sweep -k "6144_5120_3456" -x -s

    # Worker: run directly (useful for debugging, no profiling)
    pytest models/tt_dit/tests/models/wan2_2/sweep_agmm_block_sizes.py::test_agmm_sweep_worker -k "6144_5120_3456-m4" -x -s

    # Standalone script
    python models/tt_dit/tests/models/wan2_2/sweep_agmm_block_sizes.py --shape 6144,5120,3456
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
# CONFIGURATION
# ============================================================================

# (M, K, N, core_grid_x, core_grid_y)
SHAPES = [
    (96, 96, 192, 11, 10),
    (64, 192, 384, 11, 10),
    (64, 96, 192, 11, 10),
    (32, 96, 192, 11, 10),
    (32, 192, 384, 11, 10),
    (32, 256, 5120, 11, 10),
    (32, 32, 32, 11, 10),
    (32, 1280, 30720, 11, 10),
    (32, 3072, 10240, 11, 10),
    (32, 5120, 1280, 11, 10),
    (32, 10240, 10240, 11, 10),
    (128, 5120, 2560, 11, 10),
    (512, 4096, 5120, 11, 10),
    (512, 5120, 5120, 11, 10),
    (6144, 384, 384, 11, 10),
    (6144, 384, 1152, 11, 10),
    (6144, 3456, 5120, 11, 10),
    (6144, 5120, 64, 11, 10),
    (6144, 5120, 1280, 12, 9),
    (6144, 5120, 3456, 11, 10),
    (6144, 5120, 3840, 12, 9),
    (6240, 384, 384, 11, 10),
    (6240, 384, 1152, 11, 10),
    (6240, 3456, 5120, 11, 10),
    (6240, 5120, 64, 11, 10),
    (6240, 5120, 1280, 12, 9),
    (6240, 5120, 3456, 11, 10),
    (6240, 5120, 3840, 12, 9),
    (14400, 384, 384, 11, 10),
    (14400, 384, 1152, 11, 10),
    (14400, 3456, 5120, 11, 10),
    (14400, 5120, 64, 11, 10),
    (14400, 5120, 1280, 12, 9),
    (14400, 5120, 3456, 11, 10),
    (14400, 5120, 3840, 12, 9),
]

SHAPE_IDS = [f"{M}_{K}_{N}" for M, K, N, _, _ in SHAPES]

# Mesh / fabric config
SP_AXIS = 0
TP_AXIS = 1
CLUSTER_AXIS = 1
NUM_LINKS = 1
NUM_WORKERS_PER_LINK = 4

# Block sweep range
MAX_BLOCK = 16

CSV_FILE = "sweep_results_agmm.csv"
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


def pick_subblock(m_block, n_block):
    """Pick best valid (sb_h, sb_w) where sb_h|m_block, sb_w|n_block, sb_h*sb_w <= 8."""
    best = (1, 1)
    best_product = 1
    for h in range(1, min(m_block, 8) + 1):
        if m_block % h != 0:
            continue
        for w in range(1, min(n_block, 8) + 1):
            if n_block % w != 0:
                continue
            if h * w <= 8 and h * w > best_product:
                best = (h, w)
                best_product = h * w
    return best


def generate_kn_combos(K_tiles, N_tiles):
    """Generate all valid (K_block, N_block) divisor combos in [1, MAX_BLOCK]."""
    k_divs = get_divisors(K_tiles)
    n_divs = get_divisors(N_tiles)
    return [(k, n) for k in k_divs for n in n_divs]


def compute_tile_counts(M, K, N, sp_size=2):
    """Compute tile counts for the per-device matmul."""
    per_device_M = M // sp_size
    M_tiles = max(1, -(-per_device_M // 32))  # ceiling division
    K_tiles = K // 32
    N_tiles = N // 32
    return per_device_M, M_tiles, K_tiles, N_tiles


def open_mesh():
    """Open a (2,4) mesh device with FABRIC_1D."""
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))


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


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("m_block", range(1, MAX_BLOCK + 1), ids=[f"m{i}" for i in range(1, MAX_BLOCK + 1)])
def test_agmm_sweep_worker(shape, m_block):
    """Run all (K_block, N_block) combos for a given shape and M_block.

    Designed to be invoked via run_device_profiler as a subprocess.
    Emits start/stop signposts around the measured region.
    """
    M, K, N, cgx, cgy = shape
    per_device_M, M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)

    if M_tiles % m_block != 0:
        pytest.skip(f"m_block={m_block} doesn't divide M_tiles={M_tiles}")

    kn_combos = generate_kn_combos(K_tiles, N_tiles)
    if not kn_combos:
        pytest.skip("No valid (K_block, N_block) combos")

    logger.info(f"Worker: M={M} K={K} N={N} m_block={m_block}, {len(kn_combos)} K/N combos")

    mesh_device = open_mesh()
    try:
        core_grid = ttnn.CoreCoord(cgx, cgy)
        dtype = ttnn.bfloat16

        # Create input tensors (same layout as run_test_linear)
        torch_input = torch.randn((M, K), dtype=torch.float32)
        weight_input = torch.randn((K, N), dtype=torch.float32)
        bias_input = torch.randn((1, N), dtype=torch.float32)

        shard_dims = [TP_AXIS, SP_AXIS]
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        tt_weight = ttnn.from_torch(weight_input, dtype=dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_bias = ttnn.from_torch(bias_input, dtype=dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT)

        # Persistent output buffer for all_gather intermediate
        ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
        )
        ccl_semaphore_handles = create_global_semaphores(mesh_device, mesh_device.get_num_devices(), ccl_cores, 0)

        persistent_output_buffer = ttnn.from_torch(
            torch.zeros((per_device_M, K), dtype=torch.float32),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
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
                num_links=NUM_LINKS,
                topology=ttnn.Topology.Ring,
                cluster_axis=CLUSTER_AXIS,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=NUM_WORKERS_PER_LINK,
                num_buffers_per_channel=48,
                scalar=None,
                addcmul_input_tensor1=None,
                addcmul_input_tensor2=None,
                chunks=1,
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
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_agmm_sweep(shape):
    """Orchestrate the block size sweep for one shape.

    For each valid M_block, invokes test_agmm_sweep_worker via the device profiler
    as a subprocess, then parses the ops log to extract device kernel durations.
    """
    from tracy.process_model_log import run_device_profiler

    M, K, N, cgx, cgy = shape
    per_device_M, M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
    shape_id = f"{M}_{K}_{N}"
    core_grid_str = f"{cgx}x{cgy}"

    m_blocks = get_divisors(M_tiles)
    kn_combos = generate_kn_combos(K_tiles, N_tiles)

    if not kn_combos:
        pytest.skip(f"No valid (K_block, N_block) combos for K_tiles={K_tiles}, N_tiles={N_tiles}")

    logger.info(
        f"Sweep {shape_id}: M_tiles={M_tiles}, K_tiles={K_tiles}, N_tiles={N_tiles}, "
        f"{len(m_blocks)} M_blocks, {len(kn_combos)} K/N combos each"
    )

    write_csv_header(CSV_FILE)
    all_results = []

    for m_block in m_blocks:
        subdir = f"agmm_sweep_{shape_id}_m{m_block}"
        command = (
            f"pytest models/tt_dit/tests/models/wan2_2/sweep_agmm_block_sizes.py"
            f"::test_agmm_sweep_worker[{shape_id}-m{m_block}] -x"
        )

        logger.info(f"  M_block={m_block}: profiling {len(kn_combos)} combos...")

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            durations = parse_ops_log(subdir)

            if len(durations) != len(kn_combos):
                logger.warning(
                    f"  M_block={m_block}: expected {len(kn_combos)} ops, got {len(durations)} in profiler log"
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

            logger.info(f"  M_block={m_block}: done, {min(len(durations), len(kn_combos))} results recorded")

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
            f"BEST for {shape_id}: "
            f"M={best['M_block']} K={best['K_block']} N={best['N_block']} "
            f"sb_h={best['subblock_h']} sb_w={best['subblock_w']} "
            f"-> {best['duration_ns']:.0f} ns"
        )
        # Print top 5
        logger.info("Top 5 configs:")
        for rank, r in enumerate(all_results[:5], 1):
            logger.info(
                f"  #{rank}: M={r['M_block']} K={r['K_block']} N={r['N_block']} "
                f"sb=({r['subblock_h']},{r['subblock_w']}) -> {r['duration_ns']:.0f} ns"
            )
    else:
        logger.warning(f"No valid results for {shape_id}")


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================


def main():
    from tracy.process_model_log import run_device_profiler

    parser = argparse.ArgumentParser(description="Sweep AGMM block sizes with device profiler")
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Filter to a single shape as M,K,N (e.g. 6144,5120,3456)",
    )
    parser.add_argument("--csv", type=str, default=CSV_FILE)
    parser.add_argument("--max-block", type=int, default=MAX_BLOCK, help="Max block size in tiles (default: 16)")
    args = parser.parse_args()

    # Filter shapes
    if args.shape:
        m, k, n = [int(x) for x in args.shape.split(",")]
        shapes = [(M, K, N, cgx, cgy) for M, K, N, cgx, cgy in SHAPES if M == m and K == k and N == n]
        if not shapes:
            print(f"Shape {args.shape} not found in SHAPES table")
            return
    else:
        shapes = SHAPES

    write_csv_header(args.csv)
    all_best = {}

    for M, K, N, cgx, cgy in shapes:
        per_device_M, M_tiles, K_tiles, N_tiles = compute_tile_counts(M, K, N)
        shape_id = f"{M}_{K}_{N}"
        core_grid_str = f"{cgx}x{cgy}"

        m_blocks = get_divisors(M_tiles, args.max_block)
        kn_combos = generate_kn_combos(K_tiles, N_tiles)

        if not kn_combos:
            print(f"Skipping {shape_id}: no valid K/N combos")
            continue

        print(f"\n{'='*80}")
        print(f"Shape {shape_id}: M_tiles={M_tiles} K_tiles={K_tiles} N_tiles={N_tiles}")
        print(f"  {len(m_blocks)} M_blocks x {len(kn_combos)} K/N combos")
        print(f"{'='*80}")

        shape_results = []

        for m_block in m_blocks:
            subdir = f"agmm_sweep_{shape_id}_m{m_block}"
            command = (
                f"pytest models/tt_dit/tests/models/wan2_2/sweep_agmm_block_sizes.py"
                f"::test_agmm_sweep_worker[{shape_id}-m{m_block}] -x"
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
            all_best[(M, K, N)] = shape_results[0]
            best = shape_results[0]
            print(
                f"  BEST: M={best['M_block']} K={best['K_block']} N={best['N_block']} "
                f"sb=({best['subblock_h']},{best['subblock_w']}) -> {best['duration_ns']:.0f} ns"
            )

    # Print summary
    if all_best:
        print(f"\n{'='*100}")
        print("SWEEP SUMMARY - Best configs per shape")
        print(f"{'='*100}")
        print(
            f"{'M':>6} {'K':>6} {'N':>6} | {'M_blk':>5} {'K_blk':>5} {'N_blk':>5} "
            f"{'sb_h':>4} {'sb_w':>4} | {'duration_ns':>12}"
        )
        print("-" * 100)
        for (M, K, N), best in sorted(all_best.items()):
            print(
                f"{M:>6} {K:>6} {N:>6} | "
                f"{best['M_block']:>5} {best['K_block']:>5} {best['N_block']:>5} "
                f"{best['subblock_h']:>4} {best['subblock_w']:>4} | "
                f"{best['duration_ns']:>12.0f}"
            )
        print(f"{'='*100}")
        print(f"\nFull results written to: {args.csv}")


if __name__ == "__main__":
    main()

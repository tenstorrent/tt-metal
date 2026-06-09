# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone matmul perf benchmark for [ISL, 7168] @ [7168, 2048].
  Activations: bfloat8_b
  Weights:     bfloat4_b

Run:
  python models/demos/deepseek_v3_d_p/tests/perf/matmul_perf.py

The driver spawns one tracy-profiled subprocess that runs a matmul per ISL
(each delimited by a signpost). It then parses the resulting
ops_perf_results_*.csv, pulls the "DEVICE KERNEL DURATION [ns]" column for
each matmul, and writes matmul_perf.png next to this script.
"""

import argparse
from pathlib import Path

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.perf.matmul_perf_bh_program_configs import (
    BHProgramConfig1DFirstTwoMatmuls,
    BHProgramConfig1DThirdMatmul,
    BHProgramConfig2DFirstTwoMatmuls,
    BHProgramConfig2DThirdMatmul,
)
from models.demos.deepseek_v3_d_p.tests.perf.matmul_perf_wh_program_configs import (
    WHProgramConfig1DFirstTwoMatmuls,
    WHProgramConfig1DThirdMatmul,
    WHProgramConfig2DFirstTwoMatmuls,
    WHProgramConfig2DThirdMatmul,
)

K, N = 7168, 2048
ISL_VALUES = [32, 64, 128, 192, 256, 384, 512, 768]
NUM_ITERS = 10
SUBDIR = "matmul_perf"

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def run_worker(program_configs):
    """Profiled subprocess: open device, signpost+matmul per ISL."""
    import torch
    from loguru import logger
    from tracy import signpost

    # Use a 1x1 mesh — direct ttnn.open_device(device_id=0) fails on this box
    # because fabric auto-discovery sets up a 2x1 mesh and chip 0 isn't in the
    # control-plane chip mapping (TT_FATAL @ control_plane.cpp:1264).
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        torch.manual_seed(42)

        w_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        tt_w = ttnn.from_torch(w_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b)

        for isl in ISL_VALUES:
            print(f"\n=== ISL={isl} ===")
            x_torch = torch.randn(1, 1, isl, K, dtype=torch.bfloat16)
            tt_x = ttnn.from_torch(x_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

            program_config = program_configs[isl]

            # Per-ISL warmup
            _ = ttnn.matmul(tt_x, tt_w, program_config=program_config, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
            ttnn.synchronize_device(device)

            signpost(f"isl_{isl}")
            for _ in range(NUM_ITERS):
                _ = ttnn.matmul(
                    tt_x, tt_w, program_config=program_config, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI
                )
            ttnn.synchronize_device(device)
            logger.info(f"ran {NUM_ITERS}x matmul ISL={isl}")
    finally:
        ttnn.close_mesh_device(device)


def run_driver(third_matmul: bool = False, dim: str = "1d"):
    """Spawn worker under tracy, parse CSV, plot ISL vs kernel duration."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from loguru import logger
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    # Tracy's report mode reassembles argv with " ".join, which drops the
    # quoting around the inner command. So we pass a module path (no
    # `python` prefix, no embedded spaces) — runpy will find it as a
    # namespace package.
    cmd = "models.demos.deepseek_v3_d_p.tests.perf.matmul_perf --worker"
    if third_matmul:
        cmd += " --third_matmul"
    if dim != "1d":
        cmd += f" --dim {dim}"
    run_device_profiler(cmd, SUBDIR, device_analysis_types=["device_kernel_duration"])

    csv_path = get_latest_ops_log_filename(SUBDIR)
    logger.info(f"profiler CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    duration_col = "DEVICE KERNEL DURATION [ns]"
    isl_to_durations = {}
    current_isl = None
    for _, row in df.iterrows():
        op_type = str(row.get("OP TYPE", ""))
        op_code = str(row.get("OP CODE", ""))
        if op_type == "signpost" and op_code.startswith("isl_"):
            current_isl = int(op_code.split("_", 1)[1])
            isl_to_durations[current_isl] = []
        elif current_isl is not None and "matmul" in op_code.lower():
            # Cap at NUM_ITERS — the next ISL's warmup syncs before its
            # signpost is placed, so without this cap it leaks into the
            # previous ISL's bucket.
            if len(isl_to_durations[current_isl]) < NUM_ITERS:
                val = row[duration_col]
                if val != "-":
                    isl_to_durations[current_isl].append(float(val))

    if not any(isl_to_durations.values()):
        raise RuntimeError(
            f"no matmul rows matched signposts in {csv_path}. "
            f"OP CODEs present: {sorted(df['OP CODE'].astype(str).unique())[:20]}"
        )

    isls = sorted(isl_to_durations)
    durations_us = [sum(isl_to_durations[isl]) / len(isl_to_durations[isl]) / 1e3 for isl in isls]
    for isl, us in zip(isls, durations_us):
        n = len(isl_to_durations[isl])
        logger.info(f"ISL={isl:>5}: mean {us:8.2f} us  (n={n})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(isls, durations_us, marker="o")
    ax.set_xlabel("ISL")
    ax.set_ylabel("Kernel duration (μs)")
    ax.set_title(f"matmul [ISL, {K}] @ [{K}, {N}], bf8 × bf4")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    out_path = Path(__file__).resolve().parent / "matmul_perf.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="internal: profiled matmul runner")
    parser.add_argument(
        "--third_matmul",
        action="store_true",
        help="use (K,N)=(2048, 7168) instead of the default (7168, 2048)",
    )
    parser.add_argument(
        "--dim",
        choices=["1d", "2d"],
        default="1d",
        help="which program-config family to use: 1D mcast (default) or 2D mcast",
    )
    args = parser.parse_args()

    if args.third_matmul:
        K, N = 2048, 7168

    if args.dim == "1d" and not args.third_matmul:
        program_configs = BHProgramConfig1DFirstTwoMatmuls if is_blackhole() else WHProgramConfig1DFirstTwoMatmuls
    elif args.dim == "1d" and args.third_matmul:
        program_configs = BHProgramConfig1DThirdMatmul if is_blackhole() else WHProgramConfig1DThirdMatmul
    elif args.dim == "2d" and not args.third_matmul:
        program_configs = BHProgramConfig2DFirstTwoMatmuls if is_blackhole() else WHProgramConfig2DFirstTwoMatmuls
    else:
        program_configs = BHProgramConfig2DThirdMatmul if is_blackhole() else WHProgramConfig2DThirdMatmul

    if args.worker:
        run_worker(program_configs)
    else:
        run_driver(third_matmul=args.third_matmul, dim=args.dim)

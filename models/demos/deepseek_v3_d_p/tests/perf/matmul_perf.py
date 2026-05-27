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

K, N = 7168, 2048
# ISL_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 5120]
ISL_VALUES = [32]
NUM_ITERS = 10
SUBDIR = "matmul_perf"

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Placeholder default — replace per-ISL entries below with tuned values.
DEFAULT_PROGRAM_CONFIG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 7),
    in0_block_w=1,
    out_subblock_h=1,
    out_subblock_w=1,
    per_core_M=1,
    per_core_N=1,
    fuse_batch=False,
    mcast_in0=False,
)

PROGRAM_CONFIGS = {
    32: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        per_core_M=1,
        per_core_N=2,
        out_subblock_h=1,
        out_subblock_w=2,
        mcast_in0=True,
        fuse_batch=False,
        fused_activation=None,
    ),
    64: DEFAULT_PROGRAM_CONFIG,
    128: DEFAULT_PROGRAM_CONFIG,
    256: DEFAULT_PROGRAM_CONFIG,
    512: DEFAULT_PROGRAM_CONFIG,
    1024: DEFAULT_PROGRAM_CONFIG,
    2048: DEFAULT_PROGRAM_CONFIG,
    4096: DEFAULT_PROGRAM_CONFIG,
    5120: DEFAULT_PROGRAM_CONFIG,
}


def run_worker():
    """Profiled subprocess: open device, signpost+matmul per ISL."""
    import torch
    from loguru import logger
    from tracy import signpost

    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(42)

        w_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        tt_w = ttnn.from_torch(w_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b)

        for isl in ISL_VALUES:
            x_torch = torch.randn(1, 1, isl, K, dtype=torch.bfloat16)
            tt_x = ttnn.from_torch(x_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

            program_config = PROGRAM_CONFIGS[isl]

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
        ttnn.close_device(device)


def run_driver():
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
    args = parser.parse_args()

    if args.worker:
        run_worker()
    else:
        run_driver()

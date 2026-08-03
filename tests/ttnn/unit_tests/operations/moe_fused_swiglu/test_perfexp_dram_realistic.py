# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Realistic DM recreation. See perf_experiments/dram_download/dl_realistic.py."""
import os

os.environ.setdefault("TT_METAL_PROFILER_DIR", "/tmp/moe_dl_profiler")
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_bench as base
from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_realistic as dr


def test_realistic(device):
    tensors = base.make_tensors(device, 7168, 5120, "bf16_rm", "nd_shard", 8)
    out = {}
    for stage in (0, 1):
        logger.info(f"[real] TRYING stage {stage} ...")
        r = dr.measure(device, stage=stage, tensors=tensors, reps=7)
        out[stage] = r
        logger.info(
            f"[real] stage {stage}  med {r['ns_median']/1e3:7.2f}  min {r['ns_min']/1e3:7.2f}"
            f"  max {r['ns_max']/1e3:7.2f} us   cores={r['cores']}"
        )
    a, b = out[0]["ns_median"], out[1]["ns_median"]
    logger.info(f"[real] ORDERING COST: {a/1e3:.2f} -> {b/1e3:.2f} us  {100*(b-a)/a:+.1f}%")
    logger.info(f"[real] vs op skip_compute 104.35 us (has the semaphore rendezvous too)")

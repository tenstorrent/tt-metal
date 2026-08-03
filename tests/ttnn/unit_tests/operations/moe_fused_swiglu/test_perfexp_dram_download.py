# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Thin pytest entry point for the dram_download floor bench.

All logic lives under ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/dram_download/ — a
test_*.py cannot live inside the ttnn package tree (pytest's importlib mode re-executes
ttnn/ttnn/__init__.py under a second qualified name and crashes on duplicate op registration).

    scripts/run_safe_pytest.sh --run-all --no-precompile <this file>
"""
import os

# Must be set before the device opens; a private dir so a concurrent profiled run cannot race us.
os.environ.setdefault("TT_METAL_PROFILER_DIR", "/tmp/moe_dl_profiler")
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_bench as dl

CASES = [
    (fmt, wp, mode)
    for fmt in ("bf16_rm", "bfp8_tile")
    for wp in ("nd_shard", "interleaved")
    for mode in ("weights", "x", "all")
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}_{c[1]}_{c[2]}")
def test_download(device, case):
    fmt, wplace, mode = case
    r = dl.measure(device, input_format=fmt, wplace=wplace, mode=mode, count=256)
    logger.info(
        f"[dl] {mode:<7} {fmt:<9} {wplace:<11} {r['bytes']/1e6:6.2f} MB  "
        f"{r['ns_median']/1e3:8.2f} us  {r['gbps']:6.1f} GB/s  {r['pct_peak']:5.1f}% of 512  "
        f"(min {r['ns_min']/1e3:.2f} max {r['ns_max']/1e3:.2f}) cores={r['cores']}"
    )

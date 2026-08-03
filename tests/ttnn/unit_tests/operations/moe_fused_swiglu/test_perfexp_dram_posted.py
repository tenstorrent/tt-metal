# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""POSTED vs non-posted multicast. MEASUREMENT ONLY — posted gives no landing guarantee."""
import os

os.environ.setdefault("TT_METAL_PROFILER_DIR", "/tmp/moe_dl_profiler")
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_collective as dc

CASES = [
    ("h 1 root  (clean)", ("h",), True),
    ("h 11 roots", ("h",), False),
    ("all phases", ("x", "reduce", "h"), False),
]


def test_posted(device):
    for name, phases, single in CASES:
        res = {}
        for posted in (0, 1):
            r = dc.measure(device, phases=phases, posted=posted, single_root=single, reps=15)
            res[posted] = r
            logger.info(
                f"[p2] {name:<20} posted={posted}  {r['bytes']/1e6:6.2f} MB  med {r['ns_median']/1e3:7.2f}"
                f"  min {r['ns_min']/1e3:7.2f}  max {r['ns_max']/1e3:7.2f} us"
            )
        for k in ("ns_median", "ns_min"):
            a, b = res[0][k], res[1][k]
            logger.info(f"[p2] {name:<20} {k:<10} {a/1e3:7.2f} -> {b/1e3:7.2f} us  {100*(b-a)/a:+6.1f}%")

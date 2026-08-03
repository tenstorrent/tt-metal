# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collective-traffic floor with the op's real per-RISC assignment.

See perf_experiments/dram_download/dl_collective.py. The op sends h and x on the READER (NOC_0) and
splits the reduce-scatter across both RISCs.
"""
import os

os.environ.setdefault("TT_METAL_PROFILER_DIR", "/tmp/moe_dl_profiler")
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_collective as dc

OP = dc.OP_ASSIGN
H_ON_WRITER = {"x": (1, 0), "reduce": (1, 1), "h": (0, 1)}
H_ON_BOTH = {"x": (1, 0), "reduce": (1, 1), "h": (1, 1)}

CASES = [
    ("h  only  NOC_0 (shipped)", ("h",), OP),
    ("h  only  NOC_1", ("h",), H_ON_WRITER),
    ("h  only  BOTH (2x bytes)", ("h",), H_ON_BOTH),
    ("x  only  NOC_0", ("x",), OP),
    ("reduce   split", ("reduce",), OP),
    ("all      op assignment", ("x", "reduce", "h"), OP),
]


def test_collective(device):
    for name, phases, assign in CASES:
        logger.info(f"[coll2] TRYING {name} ...")
        r = dc.measure(device, phases=phases, assign=assign, reps=5)
        logger.info(
            f"[coll2] {name:<26} {r['bytes']/1e6:6.2f} MB  {r['ns_median']/1e3:7.2f} us  "
            f"{r['gbps']:6.1f} GB/s delivered  (min {r['ns_min']/1e3:.2f} max {r['ns_max']/1e3:.2f})"
        )

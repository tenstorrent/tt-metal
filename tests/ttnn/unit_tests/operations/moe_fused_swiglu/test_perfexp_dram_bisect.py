# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bisect the dl_split hang: simplest config first, so the last logged line names the culprit."""
import os

os.environ.setdefault("TT_METAL_PROFILER_DIR", "/tmp/moe_dl_profiler")
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_bench as base
from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_split as ds

STEPS = [
    ("wonly seq  f=1", dict(noc0_frac=1.0, interleave=0, with_x=False)),
    ("wonly seq  f=0", dict(noc0_frac=0.0, interleave=0, with_x=False)),
    ("wonly seq  f=.5", dict(noc0_frac=0.5, interleave=0, with_x=False)),
    ("wonly ilv  f=.5", dict(noc0_frac=0.5, interleave=1, with_x=False)),
    ("wonly ilv  f=1", dict(noc0_frac=1.0, interleave=1, with_x=False)),
    ("withx seq  f=.5", dict(noc0_frac=0.5, interleave=0, with_x=True)),
    ("withx ilv  f=1", dict(noc0_frac=1.0, interleave=1, with_x=True)),
]


def test_bisect(device):
    tensors = base.make_tensors(device, 7168, 5120, "bf16_rm", "nd_shard", 8)
    for name, kw in STEPS:
        logger.info(f"[bisect] TRYING {name} ...")
        r = ds.measure(device, tensors=tensors, reps=1, **kw)
        logger.info(f"[bisect] OK {name}: {r['ns_median']/1e3:7.2f} us  {r['gbps']:6.1f} GB/s")

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""NoC-split sweep for the DRAM weight download. See perf_experiments/dram_download/dl_split.py.

WEIGHTS ONLY (98.2 % of the op's read bytes). The x stream is measured separately by
test_perfexp_dram_download.py (0.46 MB, 1.8 us); combining it with the split currently hangs and is
not needed to answer the split question.
"""
import os

os.environ.setdefault("TT_METAL_PROFILER_DIR", "/tmp/moe_dl_profiler")
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_bench as base
from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_split as ds

FRACS = [1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.2, 0.0]


@pytest.mark.parametrize("interleave", [0, 1], ids=["sequential", "interleaved"])
@pytest.mark.parametrize("wplace", ["nd_shard", "interleaved"])
def test_split_sweep(device, interleave, wplace):
    tensors = base.make_tensors(device, 7168, 5120, "bf16_rm", wplace, 8)
    rows = []
    for f in FRACS:
        r = ds.measure(device, noc0_frac=f, interleave=interleave, with_x=False, tensors=tensors, reps=5)
        rows.append(r)
        logger.info(
            f"[sw {wplace[:5]}/{interleave}] f={f:4.2f} NOC0 {r['bytes_noc0']/1e6:5.2f}MB NOC1 {r['bytes_noc1']/1e6:5.2f}MB"
            f"  {r['ns_median']/1e3:7.2f}us {r['gbps']:6.1f}GB/s {r['pct_peak']:5.1f}%peak"
        )
    solo0 = next(r for r in rows if r["noc0_frac"] == 1.0)
    solo1 = next(r for r in rows if r["noc0_frac"] == 0.0)
    r0, r1 = solo0["gbps"], solo1["gbps"]
    fstar = r0 / (r0 + r1)
    best = min(rows, key=lambda r: r["ns_median"])
    logger.info(
        f"[sw {wplace[:5]}/{interleave}] r0(NOC_0)={r0:.1f} r1(NOC_1)={r1:.1f} GB/s  ratio={r0/r1:.2f}x"
        f" | f*=r0/(r0+r1)={fstar:.3f} -> if rates ADDED: {solo0['bytes']/((r0+r1)*1e9)*1e6:.2f}us ({r0+r1:.0f}GB/s)"
        f" | BEST f={best['noc0_frac']:.2f}: {best['ns_median']/1e3:.2f}us {best['gbps']:.1f}GB/s"
        f" ({100*best['gbps']/(r0+r1):.0f}% of the additive bound)"
    )

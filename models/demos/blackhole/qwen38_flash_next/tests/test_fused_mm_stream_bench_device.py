# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x52 step 1: per-tile cost of the decode matmuls' compute on one core (HiFi2 x BF8, HiFi4 x BF16, fp32 DEST), traced slope of
reps 0 vs R.  Run through x52_mm_stream/run_dev.sh."""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

RESULTS = Path(os.environ.get("X52_RESULTS", "/tmp/x52_bench.jsonl"))
KERNEL = fp.kernel_source("mm_stream", "bench_mm_compute.cpp")
FID = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4, "LoFi": ttnn.MathFidelity.LoFi}
WT = {"bf8": (ttnn.bfloat8_b, 1088), "bf16": (ttnn.bfloat16, 2048)}


def program(core, fid, wt, ct, kt, kblk, reps):
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    dtype, tile_bytes = WT[wt]
    cbs = [
        fp.cb_descriptor(0, ttnn.bfloat16, 2048, 80, cores),
        fp.cb_descriptor(1, dtype, tile_bytes, 80 * ct, cores),
        fp.cb_descriptor(2, ttnn.bfloat16, 2048, 4, cores),
    ]
    compute = fp.compute_kernel(
        KERNEL,
        cores,
        [],
        [(core, [reps])],
        defines=[("CT", str(ct)), ("KT", str(kt)), ("KBLK", str(kblk))],
        fidelity=FID[fid],
        fp32_dest=True,
        approx=False,
        dst_full_sync=False,
    )
    return fp.program_descriptor([compute], cbs=cbs)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_bench_mm_compute(mesh_device):
    mesh = mesh_device
    io = [fp.allocate((1, 1, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh) for _ in range(2)]
    core = ttnn.CoreCoord(1, 1)
    for fid, wt in (("HiFi2", "bf8"), ("HiFi4", "bf16"), ("HiFi2", "bf16"), ("LoFi", "bf8")):
        for ct in (1, 2, 4):
            for kt, kblk in ((80, 10), (80, 1)):
                res = {}
                for reps in (0, 40):
                    prog = program(core, fid, wt, ct, kt, kblk, reps)
                    res[reps] = traced_wall_us(mesh, lambda: fp.run_program(io, prog), replays=60)[0]
                per_rep = (res[40] - res[0]) / 40
                ns_tile = per_rep * 1000 / (kt * ct)
                row = dict(
                    fid=fid,
                    wt=wt,
                    ct=ct,
                    kt=kt,
                    kblk=kblk,
                    us_per_subblock=round(per_rep, 3),
                    ns_per_tile=round(ns_tile, 2),
                    cycles_per_tile=round(ns_tile * 1.35, 1),
                )
                logger.info(json.dumps(row))
                with RESULTS.open("a") as f:
                    f.write(json.dumps(row) + "\n")

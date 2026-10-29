# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x37 step 0: microbench of the routed-expert program on ONE die (a 1x1 mesh): (a) the cost of one BF4 x bf16 LoFi
matmul tile of moe_compute's compute configuration, (b) the DRAM rate of ``n`` reader cores per bank on today's
cached BF4 layout.  Wall time per op = the slope between a 4-op and a 12-op trace; the per-rep cost is the slope
between ``reps = 0`` and ``reps = R`` of the same program.

Run: MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0 flock ... pytest this_file -s -p conftest   (MESH_DEVICE only for tp.py's import)
Results: appended as json lines to $X37_RESULTS (default /tmp/x37_bench.jsonl).
"""

import json
import os
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import load_layer_shard, traced_wall_us
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_experts as me
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

RESULTS = Path(os.environ.get("X37_RESULTS", "/tmp/x37_bench.jsonl"))
BF16, BF4 = ttnn.bfloat16, ttnn.bfloat4_b
DEVICE_PARAMS = [{"trace_region_size": 32 * 1024 * 1024}]


def record(**row):
    row["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row["load"] = os.getloadavg()
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info(json.dumps(row))


def compute_program(unit: int, reps: int, core):
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    cbs = [
        fp.cb_descriptor(0, BF16, 2048, me.HT, cores),
        fp.cb_descriptor(1, BF16, 2048, 4, cores),
        fp.cb_descriptor(3, BF4, 576, 3 * me.BLOCK_TILES, cores),
        fp.cb_descriptor(6, BF16, 2048, 24, cores),
    ]
    compute = fp.compute_kernel(
        fp.kernel_source(me.NAME, "bench_compute_moe_experts.cpp"),
        cores,
        [],
        [(core, [reps])],
        defines=[("BENCH_UNIT", str(unit))],
        named=[("activation_limit_bits", 0)],
        fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest=False,
        approx=True,
        dst_full_sync=False,
        bfp8_pack_precise=False,
    )
    return fp.program_descriptor([compute], cbs=cbs)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("unit", [1, 0, 2, 3])
def test_bench_compute(mesh_device, unit):
    mesh = mesh_device
    dummy = [fp.allocate((1, 1, 32, 32), BF16, ttnn.TILE_LAYOUT, mesh) for _ in range(2)]
    core = ttnn.CoreCoord(1, 1)
    reps_hi = 40
    tiles = {0: 320, 1: 320, 2: 80, 3: 160}[unit]
    res = {}
    for reps in (0, reps_hi):
        prog = compute_program(unit, reps, core)
        res[reps] = traced_wall_us(mesh, lambda: fp.run_program(dummy, prog))
    per_rep = (res[reps_hi][0] - res[0][0]) / reps_hi
    ns_tile = per_rep * 1000 / tiles
    record(
        kind="compute",
        unit=unit,
        tiles_per_rep=tiles,
        us_reps0=res[0],
        us_reps_hi=res[reps_hi],
        us_per_rep=per_rep,
        ns_per_tile=ns_tile,
        cycles_per_tile_1p35GHz=ns_tile * 1.35,
    )


def reader_program(w01, per_bank: int, slots: int, inflight: int, blocks_per_bank: int, reps: int):
    mesh = w01.device()
    placement = me.row_placement(mesh, per_bank)
    placed = [(c, b, r) for b in range(me.RING) for r, c in enumerate(placement[b])]
    vcs = me.reader_vcs(placed)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in placed])
    args = []
    for (core, bank, r), vc in zip(placed, vcs):
        first = blocks_per_bank * r // per_bank
        last = blocks_per_bank * (r + 1) // per_bank
        args.append(
            (
                core,
                [
                    bank,
                    vc,
                    w01.buffer_address() + first * me.BLOCK_BYTES,
                    last - first,
                    reps,
                    me.W01_TILES_PER_BANK_EXPERT * me.BF4_TILE_BYTES,
                ],
            )
        )
    named = [
        ("slots", slots),
        ("inflight", inflight),
        ("cb_w", 3),
        ("txn_bytes", me.TILES_PER_TXN * me.BF4_TILE_BYTES),
        ("txns_per_block", 2),
    ]
    reader = fp.reader_kernel(fp.kernel_source(me.NAME, "bench_reader_moe_experts.cpp"), cores, [], args, named=named)
    cbs = [fp.cb_descriptor(3, BF4, me.BF4_TILE_BYTES, slots * me.BLOCK_TILES, cores)]
    return fp.program_descriptor([reader], cbs=cbs), [(c.x, c.y) for c, _, _ in placed]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_bench_dram(mesh_device):
    mesh = mesh_device
    w01, w2 = load_layer_shard(mesh, 0, 0)
    dummy = [fp.allocate((1, 1, 32, 32), BF16, ttnn.TILE_LAYOUT, mesh) for _ in range(2)]
    reps_hi = 20
    blocks_per_bank = int(
        os.environ.get("X37_BLOCKS", "15")
    )  # 15 = one expert's bank bytes (10 gate/up + 5 down blocks)
    for per_bank, slots, inflight in [
        (1, 3, 2),
        (1, 6, 5),
        (2, 3, 2),
        (2, 6, 5),
        (3, 3, 2),
        (3, 4, 3),
        (3, 6, 5),
        (4, 3, 2),
        (4, 6, 5),
    ]:
        res = {}
        for reps in (0, reps_hi):
            prog, cores = reader_program(w01, per_bank, slots, inflight, blocks_per_bank, reps)
            res[reps] = traced_wall_us(mesh, lambda: fp.run_program([w01, dummy[0]], prog))
        per_rep = (res[reps_hi][0] - res[0][0]) / reps_hi
        bank_bytes = blocks_per_bank * me.BLOCK_BYTES
        record(
            kind="dram",
            per_bank=per_bank,
            slots=slots,
            inflight=inflight,
            blocks_per_bank=blocks_per_bank,
            us_per_rep=per_rep,
            GBps_per_bank=bank_bytes / per_rep / 1000,
            GBps_total=8 * bank_bytes / per_rep / 1000,
            cores=cores if per_bank == 3 and slots == 3 else None,
        )

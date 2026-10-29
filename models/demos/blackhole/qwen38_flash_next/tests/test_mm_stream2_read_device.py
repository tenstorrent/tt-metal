# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x60 step 1: pure-read micro-benchmark of the re-laid weight layout: R reader cores per DRAM bank, each reading ONE
contiguous region of its bank's shard in <= 16 KB trid-pipelined packets (no compute).  Traced slope (4 vs 12 calls), 1 die.
Run through x60_mm_stream2/run_dev.sh."""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mm_stream2 as m2
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

RESULTS = Path(os.environ.get("X60_RESULTS", "/tmp/x60_read.jsonl"))
READER = fp.kernel_source("mm_stream2", "reader_mm_stream2.cpp")
WT = {"bf8": (ttnn.bfloat8_b, 1088), "bf16": (ttnn.bfloat16, 2048)}
# (name, weight type, K tiles, tile columns per bank): the dense decode shapes at TP8 (padded to 8 banks)
SHAPES = {
    "qsa": ("bf8", 80, 9),
    "gdn_in": ("bf8", 80, 11),
    "router": ("bf16", 80, 2),
    "gate_up": ("bf8", 80, 2),
    "out_proj": ("bf8", 24, 10),
}


def read_program(mesh, w, wt, tiles_per_bank, readers, block_tiles, slots, per_core_tiles=None):
    dtype, tile_bytes = WT[wt]
    region = tiles_per_bank // readers
    blocks = region // block_tiles
    pt = m2.page_tiles(block_tiles, tile_bytes)
    if os.environ.get("X60_PLACE", "row") == "rect":  # an 8 x R rectangle: one kernel range per launch
        x0, y0 = int(os.environ.get("X60_X0", "1")), int(os.environ.get("X60_Y0", "0"))
        placed_by_bank = {b: [ttnn.CoreCoord(x0 + b, y0 + r) for r in range(readers)] for b in range(8)}
    else:
        placed_by_bank = m2.row_placement(mesh, readers)
    placed = [(c, b, r) for b in range(8) for r, c in enumerate(placed_by_bank[b])]
    vcs = m2.assign_vcs(placed)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(placed[0][0], placed[0][0])])
    for c, _, _ in placed[1:]:
        cores = cores.merge(
            ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])
        )  # adjacent cores of a row become one range (one binary multicast)
    logger.info(f"cores {len(placed)} ranges {len(cores.ranges())}")
    named = [
        ("cb_w", 0),
        ("blocks", blocks),
        ("block_tiles", block_tiles),
        ("page_tiles", pt),
        ("slots", slots),
        ("w_tile_bytes", tile_bytes),
    ]
    rt = [(c, [w.buffer_address(), b, r * region * tile_bytes, vc]) for (c, b, r), vc in zip(placed, vcs)]
    reader = fp.reader_kernel(READER, cores, [], rt, defines=[("BENCH", "1")], named=named)
    cbs = [fp.cb_descriptor(0, dtype, tile_bytes, slots * block_tiles, cores)]
    return fp.program_descriptor([reader], cbs=cbs)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_read_rate(mesh_device):
    mesh = mesh_device
    names = os.environ.get("X60_SHAPES", "qsa,gdn_in,router,gate_up,out_proj").split(",")
    readers_list = [int(v) for v in os.environ.get("X60_READERS", "1,2,3,4,6").split(",")]
    for name in names:
        wt, kt, nb = SHAPES[name]
        dtype, tile_bytes = WT[wt]
        tiles = kt * nb
        w = fp.allocate((1, 1, tiles * 32, 8 * 32), dtype, ttnn.TILE_LAYOUT, mesh, m2.weight_memory_config(8, tiles))
        io = [w, fp.allocate((1, 1, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh)]
        for readers in readers_list:
            if tiles % readers:
                continue
            region = tiles // readers
            for block_tiles in sorted({d for d in range(4, region + 1) if region % d == 0 and d <= 80}):
                for slots in (3, 6):
                    prog = None
                    try:
                        prog = read_program(mesh, w, wt, tiles, readers, block_tiles, slots)
                        us = traced_wall_us(mesh, lambda: fp.run_program(io, prog), replays=40)[0]
                    except Exception as e:  # a placement / sizing that does not fit is not a result
                        logger.warning(f"{name} R={readers} block={block_tiles}: {str(e)[:100]}")
                        continue
                    total = 8 * tiles * tile_bytes
                    row = dict(
                        shape=name,
                        wt=wt,
                        readers=readers,
                        block_tiles=block_tiles,
                        slots=slots,
                        page_tiles=m2.page_tiles(block_tiles, tile_bytes),
                        bytes=total,
                        us=round(us, 2),
                        gbps=round(total / us / 1e3, 1),
                    )
                    logger.info(json.dumps(row))
                    with RESULTS.open("a") as f:
                        f.write(json.dumps(row) + "\n")

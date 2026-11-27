# Refinement 4 probe: device-ns of one kernel/knob variant on the flagged perf case 308 and its R7
# twin 286. Run under scripts/run_safe_pytest.sh --profile (one op per test, CSV rows in order).
#   G6_KDIR   kernel directory to build from (an ablated copy of kernels/); default: the op's own.
#   G6_KNOBS  JSON {program-descriptor knob: value} applied for the run.
#   G6_CHECK  1: bit-exact check against the input (off for ablations, which move wrong data).
#   G6_SWAP_NOC  1: reader on NoC1, writer on NoC0.
#   G6_ROW_WEIGHTS  JSON list of 10 per-row work weights (logical row 0..9): a static weighted tile
#                   split instead of the even one, to measure what balancing per-core speed buys.
import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

import importlib
import sys

importlib.import_module("ttnn.operations.g6_to_layout")
op_mod = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout"]
import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd
from ttnn.operations.g6_to_layout import g6_to_layout

SHAPES = [(1, 2048, 2048, 3), (1, 1024, 1024, 3)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_variant(device, shape, monkeypatch):
    if os.environ.get("G6_KDIR"):
        monkeypatch.setattr(pd, "KERNEL_DIR", Path(os.environ["G6_KDIR"]))
    for name, value in json.loads(os.environ.get("G6_KNOBS", "{}")).items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)
    if os.environ.get("G6_SWAP_NOC") == "1":
        reader = ttnn.DataMovementConfigDescriptor(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        writer = ttnn.DataMovementConfigDescriptor(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        monkeypatch.setattr(ttnn, "ReaderConfigDescriptor", lambda: reader)
        monkeypatch.setattr(ttnn, "WriterConfigDescriptor", lambda: writer)
    if os.environ.get("G6_ROW_WEIGHTS"):
        weights = json.loads(os.environ["G6_ROW_WEIGHTS"])

        def weighted_split(grid_x, grid_y, geometry):
            cores = [(x, y) for y in range(grid_y) for x in range(grid_x)]
            total = sum(weights[y] for _, y in cores)
            core_args, tile = [], 0
            for n, (x, y) in enumerate(cores):
                end = (
                    geometry.tensor_tiles
                    if n == len(cores) - 1
                    else round(geometry.tensor_tiles * (sum(weights[c[1]] for c in cores[: n + 1]) / total))
                )
                core_args.append(((x, y), (tile, end - tile)))
                tile = end
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])
            return len(cores), grid, core_args, max(n for _, (_, n) in core_args)

        monkeypatch.setattr(pd, "_tile_split", weighted_split)
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    plan = op_mod.select_plan(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    print(f"\nPLAN {shape} {plan['regime']} rows={plan['block_rows_tiles']} l1={plan['l1_bytes_per_core']}")
    out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    if os.environ.get("G6_CHECK") == "1":
        assert torch.equal(ttnn.to_torch(out).reshape(shape).view(torch.int16), x.view(torch.int16))
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()

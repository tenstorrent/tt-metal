# Verifier probe: device-ns of PIPELINE_BLOCKS_PER_CORE in {1 (L1-cap coarsest block), 4, 16 (shipped), 64}
# on the flagged perf case 308 and the two R1 guard shapes 304 / 1978.
# Run under scripts/run_safe_pytest.sh --profile; each test dispatches REPEATS ops (CSV rows in order).
import importlib
import sys

import pytest
import torch
import ttnn

importlib.import_module("ttnn.operations.g6_to_layout")
op_mod = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout"]
import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd
from ttnn.operations.g6_to_layout import g6_to_layout

SHAPES = [(1, 2048, 2048, 3), (1, 1024, 2048, 128), (16, 4096, 2560)]
KNOBS = [1, 4, 16, 64]
REPEATS = 2


@pytest.mark.parametrize("knob", KNOBS, ids=lambda k: f"pbpc{k}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_knob(device, shape, knob, monkeypatch):
    monkeypatch.setattr(pd, "PIPELINE_BLOCKS_PER_CORE", knob)
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    plan = op_mod.select_plan(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    print(
        f"\nPLAN {shape} knob={knob} rows={plan['block_rows_tiles']} cols={plan['block_cols_tiles']} l1={plan['l1_bytes_per_core']}"
    )
    for _ in range(REPEATS):
        out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(out).reshape(shape).view(torch.int16), x.view(torch.int16))
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()

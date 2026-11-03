# R4 sweep: forced (num_waves, k_block_tiles) at t = 4096 layer 0 under the model's L1 occupancy. Run explicitly.
import os

import pytest
import torch
import ttnn

from ttnn.operations.moe_down_proj import moe_down_proj_program_descriptor as pd
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj import (
    LAYER0,
    check,
    make_operands,
    rows_for,
    run,
)
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj_perf import LAYERS, SCALED

TOKENS = int(os.environ.get("TOKENS", "4096"))
COUNTS = LAYERS[int(os.environ.get("LAYER", "0"))] if TOKENS == 4096 else SCALED[TOKENS]

FREE = 853568
# "<waves>x<w K block>[r<x step rounds>]"
CANDIDATES = [
    tuple(int(v) for v in c.replace("r", "x").split("x")) + ((1,) if "r" not in c else ())
    for c in os.environ.get("SWEEP", "2x6").split(",")
]


def occupy(device, free_bytes):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    live = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    pages = (min(live, default=view.total_bytes_per_bank) - free_bytes) // 2048
    if pages <= 0:
        return None
    filler = torch.zeros(pages * view.num_banks, 1024, dtype=torch.bfloat16)
    return ttnn.from_torch(
        filler, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )


@pytest.mark.parametrize("cand", CANDIDATES, ids=[f"w{w}k{k}r{r}" for w, k, r in CANDIDATES])
def test_sweep(device, cand, monkeypatch):
    waves, k, rounds = cand
    orig = pd.select_blocking

    def forced(row_tiles, layout, counts_page_bytes, budget, out_in_dram=False):
        rows = pd._div_up(pd._div_up(row_tiles, waves), layout.num_m_groups)
        print(
            "FORCED",
            waves,
            k,
            rounds,
            rows,
            pd.footprint_bytes(rows, k, layout.core_n_tiles, counts_page_bytes, waves, rounds),
            "budget",
            budget,
        )
        return pd.Blocking(row_tiles, waves, k, rows, rounds)

    monkeypatch.setattr(pd, "select_blocking", forced)
    for item in filter(None, os.environ.get("PD_SET", "").split(",")):
        name, value = item.split("=")
        monkeypatch.setattr(pd, name, int(value))
    import sys

    op = sys.modules["ttnn.operations.moe_down_proj.moe_down_proj"]
    monkeypatch.setattr(op, "select_blocking", lambda *a, **kw: None)
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, TOKENS, COUNTS)
    filler = (
        occupy(device, FREE + pd._div_up(rows_for(TOKENS), 110) * 1536) if os.environ.get("OCC", "1") == "1" else None
    )
    out = run(x_dev, w_dev, counts_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    if filler is not None:
        ttnn.deallocate(filler)
    check(out, xq, wq, COUNTS, ttnn.L1_MEMORY_CONFIG)

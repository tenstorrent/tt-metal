# Refinement 4: cap block_h (the x row-block, i.e. the fill transfer) at 8x512 (perf only, PCC checked).
import math
import os

import pytest
import torch
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

pytestmark = pytest.mark.skipif(not os.environ.get("LINEAR_GELU_PERF_PROBES"), reason="set LINEAR_GELU_PERF_PROBES=1")

SHAPE = tuple(int(v) for v in os.environ.get("R4_SHAPE", "8x512").split("x"))


@pytest.mark.parametrize("out_dram", [False, True])
@pytest.mark.parametrize("block_h_cap", [4, 3, 2, 1])
def test_block_h(device, block_h_cap, out_dram, monkeypatch):
    batch, seq = SHAPE
    if os.environ.get("R4_OUT_PAGES"):
        monkeypatch.setattr(pd, "OUT_RING_PAGES", int(os.environ["R4_OUT_PAGES"]))
    monkeypatch.setattr(pd, "BLOCK_H_MAX", block_h_cap)
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=ttnn.DRAM_MEMORY_CONFIG if out_dram else ttnn.L1_MEMORY_CONFIG)
    ref = torch.nn.functional.gelu(x.double().reshape(batch, seq, 768) @ w.double() + b.double(), approximate="tanh")
    act = ttnn.to_torch(out).reshape(batch, seq, 3072).double()
    pcc = torch.corrcoef(torch.stack([act.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc > 0.9999

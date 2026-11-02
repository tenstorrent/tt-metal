# Refinement 4: W_CHUNKS_BEFORE_X (fill order) across shapes / x placement (perf only, PCC checked).
import math
import os

import pytest
import torch
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

pytestmark = pytest.mark.skipif(not os.environ.get("LINEAR_GELU_PERF_PROBES"), reason="set LINEAR_GELU_PERF_PROBES=1")


@pytest.mark.parametrize("w_first", [0, 1])
@pytest.mark.parametrize("batch, seq, x_dram", [(8, 512, False), (8, 512, True), (2, 512, False), (2, 512, True)])
def test_fill_order(device, batch, seq, x_dram, w_first, monkeypatch):
    monkeypatch.setattr(pd, "W_CHUNKS_BEFORE_X", {ttnn.BufferType.L1: w_first, ttnn.BufferType.DRAM: w_first})
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.DRAM_MEMORY_CONFIG if x_dram else ttnn.L1_MEMORY_CONFIG, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=ttnn.L1_MEMORY_CONFIG)
    ref = torch.nn.functional.gelu(x.double().reshape(batch, seq, 768) @ w.double() + b.double(), approximate="tanh")
    act = ttnn.to_torch(out).reshape(batch, seq, 3072).double()
    pcc = torch.corrcoef(torch.stack([act.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc > 0.9999

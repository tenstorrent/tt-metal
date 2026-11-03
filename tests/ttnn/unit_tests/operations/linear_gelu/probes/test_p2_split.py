# Perf 2 row-split sweep (perf only): ROW_SPLIT policy x case. Rows land in the ops CSV in parametrize order.
import math
import os

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

pytestmark = pytest.mark.skipif(not os.environ.get("LINEAR_GELU_PERF_PROBES"), reason="set LINEAR_GELU_PERF_PROBES=1")

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
CASES = {
    "8x512": (8, 512, L1, L1),
    "8x512_dramx": (8, 512, DRAM, L1),
    "8x512_dramout": (8, 512, L1, DRAM),
    "8x500": (8, 500, L1, L1),
    "8x384": (8, 384, L1, L1),
    "8x384_dramx": (8, 384, DRAM, L1),
    "2x512_dramx": (2, 512, DRAM, L1),
}
POLICIES = os.environ.get("SPLIT_POLICIES", "tail,balanced").split(",")
SELECTED = os.environ.get("SPLIT_CASES", ",".join(CASES)).split(",")


@pytest.mark.parametrize("rep", range(int(os.environ.get("SPLIT_REPS", "1"))))
@pytest.mark.parametrize("case", SELECTED)
@pytest.mark.parametrize("policy", POLICIES)
def test_split(device, policy, case, rep, monkeypatch):
    import torch

    batch, seq, x_mem, out_mem = CASES[case]
    monkeypatch.setattr(pd, "ROW_SPLIT", {ttnn.BufferType.L1: policy, ttnn.BufferType.DRAM: policy})
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=x_mem, **tile)
    wd = ttnn.from_torch(w, memory_config=DRAM, **tile)
    bd = ttnn.from_torch(b, memory_config=DRAM, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=out_mem)
    ref = torch.nn.functional.gelu(x.float() @ w.float() + b.float(), approximate="tanh")
    got = ttnn.to_torch(out).float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc > 0.9999, pcc

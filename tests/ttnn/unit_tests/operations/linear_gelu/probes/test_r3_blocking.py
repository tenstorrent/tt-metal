# Refinement 3: force (block_w, transposed) to separate orientation from block_w effects (perf only).
import math
import os

import pytest
import torch
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

# Perf-only probes (run under run_safe_pytest.sh --profile); off in the regular suite.
pytestmark = pytest.mark.skipif(not os.environ.get("LINEAR_GELU_PERF_PROBES"), reason="set LINEAR_GELU_PERF_PROBES=1")

FORCED = [(1, False)]


@pytest.mark.parametrize("out_dram", [True, False])
@pytest.mark.parametrize("stagger", [True])
@pytest.mark.parametrize("out_ring_pages", [8, 24, 48])
@pytest.mark.parametrize("block_w, transposed", FORCED)
@pytest.mark.parametrize("batch, seq", [(8, 512)])
def test_forced(device, batch, seq, block_w, transposed, stagger, out_dram, out_ring_pages, monkeypatch):
    monkeypatch.setattr(pd, "OUT_RING_PAGES", out_ring_pages)
    monkeypatch.setattr(pd, "W_SENDER_STAGGER", stagger)
    monkeypatch.setattr(pd, "BLOCK_W", block_w)
    orig = pd._candidate
    grid = device.compute_with_storage_grid_size()

    def forced(tensor_m_tiles, bw, m_slots, n_slots, tile_bytes):
        want = (grid.x, grid.y) if transposed else (grid.y, grid.x)
        return orig(tensor_m_tiles, bw, m_slots, n_slots, tile_bytes) if (m_slots, n_slots) == want else None

    monkeypatch.setattr(pd, "_candidate", forced)
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
    print(
        f"CASE {batch}x{seq} bw={block_w} T={transposed} stagger={stagger} out_dram={out_dram} out_ring_pages={out_ring_pages}"
    )

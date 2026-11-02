# Refinement 4 ablation at the flagged 4096-token shape (perf only). Reuses the R3 edit table.
import math
import os

import pytest
import torch
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_r3_ablate import VARIANTS, _make  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("LINEAR_GELU_PERF_PROBES"), reason="set LINEAR_GELU_PERF_PROBES=1")

R4_VARIANTS = os.environ.get("R4_VARIANTS", "full,nogelu,nocompute,nocompute_now_nox,syncfloor,nowrite_only").split(",")
R4_SHAPE = tuple(int(v) for v in os.environ.get("R4_SHAPE", "8x512").split("x"))
R4_OUT = os.environ.get("R4_OUT", "L1")


@pytest.mark.parametrize("variant", R4_VARIANTS)
def test_ablate_r4(device, variant, monkeypatch):
    batch, seq = R4_SHAPE
    monkeypatch.setattr(pd, "KERNEL_DIR", _make(variant))
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out_mem = ttnn.L1_MEMORY_CONFIG if R4_OUT == "L1" else ttnn.DRAM_MEMORY_CONFIG
    cfg = None
    if os.environ.get("R4_FIDELITY"):
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, os.environ["R4_FIDELITY"]),
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        )
    linear_gelu(xd, wd, bd, memory_config=out_mem, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)

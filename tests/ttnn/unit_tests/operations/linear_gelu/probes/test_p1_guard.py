# Perf 1 guard set (perf only): whole-op DEVICE KERNEL DURATION per (mode, case) under --profile.
# GUARD_MODES: legacy (op kernels with the pre-Perf-1 GeluTanhActivation<1> body), new (op as is),
# bh4 (op as is, BLOCK_H_MAX = 4; obsolete after Perf 1, the op default). Rows land in the ops CSV in parametrize order.
import math
import os
from pathlib import Path

import pytest
import torch
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

pytestmark = pytest.mark.skipif(not os.environ.get("LINEAR_GELU_PERF_PROBES"), reason="set LINEAR_GELU_PERF_PROBES=1")

SRC = Path(pd.__file__).parent / "kernels"
LEGACY = Path(__file__).parent / "ablate_kernels" / "p1_legacy"
MODES = os.environ.get("GUARD_MODES", "legacy,new").split(",")
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
CASES = {
    "8x512": (8, 512, L1, L1),
    "16x256": (16, 256, L1, L1),
    "32x128": (32, 128, L1, L1),
    "8x384": (8, 384, L1, L1),
    "4x512": (4, 512, L1, L1),
    "8x256": (8, 256, L1, L1),
    "2x512": (2, 512, L1, L1),
    "4x256": (4, 256, L1, L1),
    "8x512_dramx": (8, 512, DRAM, L1),
    "8x512_dramout": (8, 512, L1, DRAM),
    "8x500": (8, 500, L1, L1),
    "1x32": (1, 32, L1, L1),
}
SELECTED = os.environ.get("GUARD_CASES", ",".join(CASES)).split(",")


def _legacy_dir():
    LEGACY.mkdir(exist_ok=True)
    for src in SRC.iterdir():
        text = src.read_text()
        if src.name == "linear_gelu_compute.cpp":
            old = "using Gelu = GeluTanhActivation<kGeluLoadMacro>;"
            assert old in text
            text = text.replace(old, "using Gelu = GeluTanhActivation<1>;")
        (LEGACY / src.name).write_text(text)
    return LEGACY


@pytest.mark.parametrize("case", SELECTED)
@pytest.mark.parametrize("mode", MODES)
def test_guard(device, mode, case, monkeypatch):
    batch, seq, x_mem, out_mem = CASES[case]
    if mode == "legacy":
        monkeypatch.setattr(pd, "KERNEL_DIR", _legacy_dir())
    if mode == "bh4":
        monkeypatch.setattr(pd, "BLOCK_H_MAX", 4)
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=x_mem, **tile)
    wd = ttnn.from_torch(w, memory_config=DRAM, **tile)
    bd = ttnn.from_torch(b, memory_config=DRAM, **tile)
    linear_gelu(xd, wd, bd, memory_config=out_mem)
    ttnn.synchronize_device(device)

# Perf 1 domain bench of regime nhwc_rm_transpose: 12 launches per case (median of launches 2..12 is the
# number, read with generated/meta/g6_permute/bench/arm_medians.py). PERF1_CASES picks case ids.
import importlib
import os

import pytest
import torch
import ttnn

op_mod = importlib.import_module("ttnn.operations.g6_permute.g6_permute")

CASES = {
    "p44_l1": ((1, 128, 256, 32), "l1"),
    "p44_dram": ((1, 128, 256, 32), "dram"),
    "c64_l1": ((1, 64, 256, 64), "l1"),
    "c64_dram": ((1, 64, 256, 64), "dram"),
    "c128_l1": ((1, 64, 256, 128), "l1"),
    "c24_l1": ((1, 128, 256, 24), "l1"),
    "c3_l1": ((1, 128, 256, 3), "l1"),
    "w1000_l1": ((1, 8, 1000, 32), "l1"),
    "w100_l1": ((1, 256, 100, 32), "l1"),
    "big_dram": ((1, 256, 512, 32), "dram"),
    "tiny_l1": ((2, 7, 5, 32), "l1"),
    "tiny_dram": ((1, 3, 100, 3), "dram"),
}
SELECTED = [c for c in os.environ.get("PERF1_CASES", ",".join(CASES)).split(",") if c]
MEM = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}


@pytest.mark.parametrize("case", SELECTED)
def test_case(case, device):
    shape, buffer = CASES[case]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=MEM[buffer])
    out = None
    for i in range(12):
        y = op_mod.g6_permute(xt, (0, 3, 1, 2))
        ttnn.synchronize_device(device)
        if i == 0:
            out = ttnn.to_torch(y)
        ttnn.deallocate(y)
    assert torch.equal(out.view(torch.int16), x.permute(0, 3, 1, 2).contiguous().view(torch.int16))

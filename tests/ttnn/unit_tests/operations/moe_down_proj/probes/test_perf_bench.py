# Perf 1 bench: moe_down_proj whole-op device time on the focus case and the guard set, with an optional kernel
# directory override for ablations / candidates. Run explicitly under run_safe_pytest.sh --profile --no-precompile.
#
#   MOE_CASES   comma list of case ids (default: focus). See CASES below.
#   MOE_KERNEL_DIR  directory holding a full copy of the op's kernels/ (edited); default = the op's own kernels.
#   MOE_DEFINES  comma list of NAME[=VALUE] preprocessor defines added to all three kernels (ablation switches).
#   MOE_CHECK=0  skip the correctness check (ablations produce wrong output by design).
import os
import sys

import pytest
import torch
import ttnn

from ttnn.operations.moe_down_proj import moe_down_proj_program_descriptor as pd
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj import check, make_operands, rows_for
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj_perf import LAYERS, SCALED
from ttnn.operations.moe_down_proj import moe_down_proj

# Nomic 8x512 per-bank L1 free below the lowest live buffer before the call (feature_spec MODEL_L1_FREE_BYTES_8x512).
MODEL_FREE = 853568
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

# id -> (tokens, counts, x memory, out memory, occupied)
CASES = {
    "focus": (4096, LAYERS[0], L1, L1, True),
    "t4096_l3_occ": (4096, LAYERS[3], L1, L1, True),
    "t4096_l2_occ": (4096, LAYERS[2], L1, L1, True),
    "t4096_l0": (4096, LAYERS[0], L1, L1, False),
    "t3072": (3072, SCALED[3072], L1, L1, False),
    "t2048": (2048, SCALED[2048], L1, L1, False),
    "t1024": (1024, SCALED[1024], L1, L1, False),
    "t4096_dram_x": (4096, LAYERS[0], DRAM, L1, False),
    "t4096_dram_out": (4096, LAYERS[0], L1, DRAM, False),
    "t1024_dram_both": (1024, SCALED[1024], DRAM, DRAM, False),
    "t37": (37, (20, 0, 9, 14, 1, 11, 7, 12), L1, L1, False),
    "t1": (1, (1, 0, 0, 0, 0, 0, 1, 0), L1, L1, False),
}
SELECTED = os.environ.get("MOE_CASES", "focus").split(",")


def occupy(device, free_bytes):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    live = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    pages = (min(live, default=view.total_bytes_per_bank) - free_bytes) // 2048
    if pages <= 0:
        return None
    filler = torch.zeros(pages * view.num_banks, 1024, dtype=torch.bfloat16)
    return ttnn.from_torch(filler, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=L1)


@pytest.mark.parametrize("case", SELECTED, ids=SELECTED)
def test_bench(device, case, monkeypatch):
    tokens, counts, x_mem, out_mem, occupied = CASES[case]
    kdir = os.environ.get("MOE_KERNEL_DIR")
    if kdir:
        from pathlib import Path

        monkeypatch.setattr(pd, "KERNEL_DIR", Path(kdir))
    defines = [d for d in os.environ.get("MOE_DEFINES", "").split(",") if d]
    if defines:
        orig = pd.create_program_descriptor

        def with_defines(*args, **kwargs):
            desc = orig(*args, **kwargs)
            pairs = [tuple(d.split("=", 1)) if "=" in d else (d, "1") for d in defines]
            for kernel in desc.kernels:
                kernel.defines = list(kernel.defines) + pairs
            return desc

        monkeypatch.setattr(pd, "create_program_descriptor", with_defines)
        op = sys.modules["ttnn.operations.moe_down_proj.moe_down_proj"]
        if hasattr(op, "create_program_descriptor"):
            monkeypatch.setattr(op, "create_program_descriptor", with_defines)
    device.disable_and_clear_program_cache()
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, tokens, counts, x_mem)
    filler = None
    if occupied:
        out_bank = 0 if out_mem.buffer_type != ttnn.BufferType.L1 else pd._div_up(rows_for(tokens), 110) * 1536
        filler = occupy(device, MODEL_FREE + out_bank)
    out = moe_down_proj(x_dev, w_dev, counts_dev, memory_config=out_mem)
    if filler is not None:
        ttnn.deallocate(filler)
    if os.environ.get("MOE_CHECK", "1") == "1":
        check(out, xq, wq, counts, out_mem)

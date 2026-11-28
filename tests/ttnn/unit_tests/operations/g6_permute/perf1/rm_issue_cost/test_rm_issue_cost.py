# perf experiment rm_issue_cost driver (kernels + variants in
# ttnn/ttnn/operations/g6_permute/perf_experiments/rm_issue_cost). Env:
#   RIC_VARIANTS  comma list of variants (variants.VARIANTS), default "base"
#   RIC_ARMS      comma list of arms (full, nogather), default "full,nogather"
#   RIC_SHAPE     p44 (L1) | p44_dram, default p44
#   RIC_LAUNCHES  launches per (variant, arm), default 12 (median of 2..N is the bench number)
# Combos run in order variant-major, arm-minor; read the CSV with ric_medians.py.
import importlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[7]
EXP = ROOT / "ttnn/ttnn/operations/g6_permute/perf_experiments/rm_issue_cost"
_spec = importlib.util.spec_from_file_location("ric_variants", EXP / "variants.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

op_mod = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
nhwc = importlib.import_module("ttnn.operations.g6_permute.g6_permute_nhwc_rm_transpose")

SHAPES = {
    "p44": ((1, 128, 256, 32), ttnn.L1_MEMORY_CONFIG),
    "p44_dram": ((1, 128, 256, 32), ttnn.DRAM_MEMORY_CONFIG),
    # Domain sweep around p44.
    "h64_l1": ((1, 64, 128, 32), ttnn.L1_MEMORY_CONFIG),  # 1-2 units per core
    "h256_l1": ((1, 256, 256, 32), ttnn.L1_MEMORY_CONFIG),  # 2x p44: 9-10 units per core
    "c64_l1": ((1, 64, 128, 64), ttnn.L1_MEMORY_CONFIG),  # 128 B input pages
    "c3_l1": ((1, 64, 200, 3), ttnn.L1_MEMORY_CONFIG),  # 6 B pages, W tail
    "c64_dram": ((1, 64, 128, 64), ttnn.DRAM_MEMORY_CONFIG),
    "h256_dram": ((1, 256, 256, 32), ttnn.DRAM_MEMORY_CONFIG),
}
VARIANT_LIST = [v for v in os.environ.get("RIC_VARIANTS", "base").split(",") if v]
ARM_LIST = [a for a in os.environ.get("RIC_ARMS", "full,nogather").split(",") if a]
SHAPE_LIST = [s for s in os.environ.get("RIC_SHAPE", "p44").split(",") if s]
LAUNCHES = int(os.environ.get("RIC_LAUNCHES", "12"))
COMBOS = [(sh, v, a) for sh in SHAPE_LIST for v in VARIANT_LIST for a in ARM_LIST]


@pytest.mark.parametrize("SHAPE,variant,arm", COMBOS, ids=[f"{sh}/{v}-{a}" for sh, v, a in COMBOS])
def test_ric(SHAPE, variant, arm, device, monkeypatch):
    shape, mem = SHAPES[SHAPE]
    op_mod._PROGRAM_CACHE.clear()
    V.install(monkeypatch, nhwc, variant, arm)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    out = None
    for i in range(LAUNCHES):
        y = op_mod.g6_permute(xt, (0, 3, 1, 2))
        ttnn.synchronize_device(device)
        if i == 0:
            out = ttnn.to_torch(y)
        ttnn.deallocate(y)
    plan = op_mod.g6_permute_plan(xt, (0, 3, 1, 2))
    exact = bool(torch.equal(out.view(torch.int16), x.permute(0, 3, 1, 2).contiguous().view(torch.int16)))
    print(
        f"\nRIC_COMBO {SHAPE}/{variant}-{arm} shape={SHAPE} block_w={plan.block_w} block_units={plan.block_units} "
        f"depth={plan.input_depth} exact={exact}"
    )
    op_mod._PROGRAM_CACHE.clear()
    # Empty the device profiler's DRAM buffers between combos (they hold ~130 launches of this op).
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception as err:  # not a profiled run
        print(f"RIC_NOTE ReadDeviceProfiler: {err}")
    if arm == "full":
        assert exact, f"{variant}: not bit-exact"

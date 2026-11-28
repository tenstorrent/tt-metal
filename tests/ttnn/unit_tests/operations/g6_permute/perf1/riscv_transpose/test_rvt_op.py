# riscv_transpose bake-off driver: nhwc_rm_transpose variants (rvt_host.py) on the op's own plan.
# Each (case, variant) runs LAUNCHES dispatches of one descriptor on the same input; launch 1 is checked
# bit-exact against torch (random bf16 bit patterns: NaN, inf, -0, denormals included), launches 2..N
# are the timing (median, rvt_report.py op LOG OPSCSV DEVCSV).
# RVT_CASES / RVT_VARIANTS (comma lists) select subsets.
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[7]
_spec = importlib.util.spec_from_file_location(
    "rvt_host", ROOT / "ttnn/ttnn/operations/g6_permute/perf_experiments/riscv_transpose/rvt_host.py"
)
rvt = importlib.util.module_from_spec(_spec)
sys.modules["rvt_host"] = rvt
_spec.loader.exec_module(rvt)
V = rvt.Variant

LAUNCHES = int(os.environ.get("RVT_LAUNCHES", "12"))
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG

CASES = {
    "p44_l1": ([1, 128, 256, 32], L1),
    "p44_dram": ([1, 128, 256, 32], DRAM),
    "c64_l1": ([1, 32, 128, 64], L1),
    "c3_l1": ([1, 64, 120, 3], L1),
    "c5_dram": ([2, 17, 37, 5], DRAM),
    "c1_l1": ([1, 16, 200, 1], L1),
    "w1000_l1": ([1, 8, 1000, 16], L1),
    "w5_dram": ([1, 30, 5, 8], DRAM),
    "n2_l1": ([2, 64, 96, 32], L1),
    # forced unit width (the op's BLOCK_W knob): W tails of 1, 4 and 16 pages
    "tail1_l1": ([1, 8, 65, 32], L1, 64),
    "tail4_dram": ([1, 8, 68, 16], DRAM, 64),
    "tail16_l1": ([1, 4, 80, 8], L1, 64),
}

TRI = {"T0": 6, "T1": 5, "T2": 5}
VARIANTS = {
    "op": V("op"),
    "solo_base1": V("solo", rvt.IMPL_BASE, 1),
    "solo_base8": V("solo", rvt.IMPL_BASE, 8),
    "solo_blkv16": V("solo", rvt.IMPL_BLKV, 16),
    "tri_b4_r1": V("shared", block_pages=256, input_depth=2, rows_depth=1, shares=TRI),
    "tri_b4_r2": V("shared", block_pages=256, input_depth=2, rows_depth=2, shares=TRI),
    "tri_b1_d4_r2": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI),
    "tri_b1_d4_r3": V("shared", block_pages=64, input_depth=4, rows_depth=3, shares=TRI),
    "tri_b4_d3_r2": V("shared", block_pages=256, input_depth=3, rows_depth=2, shares=TRI),
    "tri_b2_d4_r2": V("shared", block_pages=128, input_depth=4, rows_depth=2, shares=TRI),
    "tri_b1_d5_r2": V("shared", block_pages=64, input_depth=5, rows_depth=2, shares=TRI),
    "tri_b2_d3_r2": V("shared", block_pages=128, input_depth=3, rows_depth=2, shares=TRI),
    "s1q_b1_d4_r2": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI, brisc_read=(1, 4)),
    "s1f_b1_d4_r2": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI, brisc_read=(5, 16)),
    "s1e_b1_d4_r2": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI, brisc_read=(3, 8)),
    "s1q_b1_d6_r2": V("shared", block_pages=64, input_depth=6, rows_depth=2, shares=TRI, brisc_read=(1, 4)),
    "st_b1_d4_r2": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI, stream=True),
    "s1q_st_r2": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI, brisc_read=(1, 4), stream=True),
    "s1q_st_r3": V("shared", block_pages=64, input_depth=4, rows_depth=3, shares=TRI, brisc_read=(1, 4), stream=True),
    "s1q_h0": V(
        "shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI, brisc_read=(1, 4), first_read=(1, 2)
    ),
    "s1q_h0_st": V(
        "shared",
        block_pages=64,
        input_depth=4,
        rows_depth=2,
        shares=TRI,
        brisc_read=(1, 4),
        first_read=(1, 2),
        stream=True,
    ),
    "s1f_h0_st": V(
        "shared",
        block_pages=64,
        input_depth=4,
        rows_depth=2,
        shares=TRI,
        brisc_read=(5, 16),
        first_read=(1, 2),
        stream=True,
    ),
    # DM RISCs only (no TRISCs): NCRISC reads and transposes a share while the next block's reads fly,
    # BRISC transposes the rest and writes.
    "duo_b4_r2": V("shared", block_pages=256, input_depth=2, rows_depth=2, shares={"N": 1, "B": 1}),
    "duo_b1_d4_r2": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares={"N": 1, "B": 1}),
    "duo_b1_d4_r2_n3b5": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares={"N": 3, "B": 5}),
    # all five RISCs transpose
    "penta_b1_d4_r2": V(
        "shared", block_pages=64, input_depth=4, rows_depth=2, shares={"T0": 4, "T1": 4, "T2": 4, "N": 2, "B": 2}
    ),
    "tn_b1_d4_r2": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares={"T0": 5, "T1": 4, "T2": 4, "N": 3}),
    "s1q_O3": V(
        "shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI, brisc_read=(1, 4), dm_opt=rvt.OptLevel.O3
    ),
    "tri_O3": V("shared", block_pages=64, input_depth=4, rows_depth=2, shares=TRI, dm_opt=rvt.OptLevel.O3),
    "solo_blkv16_O3": V("solo", rvt.IMPL_BLKV, 16, dm_opt=rvt.OptLevel.O3),
    "quad_b1_d4_r2": V(
        "shared", block_pages=64, input_depth=4, rows_depth=2, shares={"T0": 4, "T1": 4, "T2": 4, "B": 4}
    ),
}


def _select(names, env):
    sel = os.environ.get(env)
    return [n for n in names if not sel or n in sel.split(",")]


def random_bf16(shape):
    bits = torch.randint(-32768, 32768, shape, dtype=torch.int32).to(torch.int16)
    return bits.view(torch.bfloat16)


@pytest.mark.parametrize("variant", _select(list(VARIANTS), "RVT_VARIANTS"))
@pytest.mark.parametrize("case", _select(list(CASES), "RVT_CASES"))
def test_op(case, variant, device):
    shape, memcfg, *forced = CASES[case]
    var = VARIANTS[variant]
    torch.manual_seed(0)
    x = random_bf16(shape)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memcfg)
    plan = rvt.build_plan(var, xt, forced[0] if forced else None)
    out_shape = [shape[d] for d in rvt.DIMS]
    ref = x.permute(*rvt.DIMS).contiguous()
    desc = None
    first = ttnn._ttnn.get_device_operation_id()
    for i in range(LAUNCHES):
        y = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, memcfg)
        if desc is None:
            desc = rvt.create(var, plan, xt, y)
        else:
            rvt.patch(desc, var, xt, y)
        out = ttnn.generic_op([xt, y], desc)
        ttnn.synchronize_device(device)
        if i == 0:
            got = ttnn.to_torch(out)
            mism = (got.view(torch.int16) != ref.view(torch.int16)).sum().item()
            print(
                f"\nRVT_PLAN {case}/{variant} cores={plan.num_cores} block_w={plan.block_w} tail_w={plan.tail_w}"
                f" block_units={plan.block_units}"
                f" depth={plan.input_depth} units_g1={plan.units_per_core_g1} mismatches={mism}"
            )
            assert mism == 0, f"{case}/{variant}: {mism} values differ from torch (bit-exact gate)"
        ttnn.deallocate(y)
    print(f"\nRVT_OP {case}/{variant} first_op={first} last_op={ttnn._ttnn.get_device_operation_id() - 1}")

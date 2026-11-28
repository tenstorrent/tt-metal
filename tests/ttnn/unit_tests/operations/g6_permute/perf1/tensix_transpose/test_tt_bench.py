# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time bench of the tensix_transpose experiment vs the op's current nhwc_rm_transpose.
One arm per test id; 12 launches per arm (launch 1 is checked bit-exact, launches 2..12 are timed).
Run one arm per process under the profiler:
    scripts/run_safe_pytest.sh --profile --no-py-zones <this file> -k "<arm> and <shape>"
then read DEVICE KERNEL DURATION [ns] of op ids [first+1, last] (bench_parse.py)."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[7]
_spec = importlib.util.spec_from_file_location(
    "tensix_transpose_exp",
    ROOT / "ttnn/ttnn/operations/g6_permute/perf_experiments/tensix_transpose/tensix_transpose.py",
)
tt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tt
_spec.loader.exec_module(tt)
op_mod = importlib.import_module("ttnn.operations.g6_permute.g6_permute")

NHWC = (0, 3, 1, 2)
LAUNCHES = int(os.environ.get("TTX_LAUNCHES", "12"))
ZONES = os.environ.get("TTX_ZONES", "0") == "1"
ABLATE = tuple(d for d in os.environ.get("TTX_ABLATE", "").split(",") if d)

ARMS = {
    "baseline": None,
    "stock": None,
    "tp": dict(variant="two_pass"),
    "tp_kw1": dict(variant="two_pass", kw=1),
    "tp_kw4": dict(variant="two_pass", kw=4),
    "tp_push2": dict(variant="two_pass", push_tiles=2),
    "tp_in3": dict(variant="two_pass", in_units=3),
    "tp_in1": dict(variant="two_pass", in_units=1),
    "tp_rows4": dict(variant="two_pass", rows_pages=4),
    "tpt": dict(variant="two_pass", use_trid=True),
    "tpt_in3": dict(variant="two_pass", use_trid=True, in_units=3),
    "tpt_push2": dict(variant="two_pass", use_trid=True, push_tiles=2),
    "tpt_kw1": dict(variant="two_pass", use_trid=True, kw=1),
    "tpt_kw4": dict(variant="two_pass", use_trid=True, kw=4),
    "fr": dict(variant="fused_raw"),
    "fd": dict(variant="fused_d2b"),
    "fdo": dict(variant="fused_d2b", reader="op"),
    "rec": dict(variant="fused_d2b", use_trid=True, reader="auto"),
    "rec_tp": dict(variant="two_pass", use_trid=True, reader="auto"),
    "fdo_in4": dict(variant="fused_d2b", reader="op", in_units=4),
    "fdt": dict(variant="fused_d2b", use_trid=True),
    "fdt_kw1": dict(variant="fused_d2b", use_trid=True, kw=1),
    "fdt_kw4": dict(variant="fused_d2b", use_trid=True, kw=4),
    "fdt_in3": dict(variant="fused_d2b", use_trid=True, in_units=3),
    "fdt_rows4": dict(variant="fused_d2b", use_trid=True, rows_pages=4),
}

SHAPES = {
    "p44": ((1, 128, 256, 32), "l1"),
    "p44dram": ((1, 128, 256, 32), "dram"),
    "c64": ((1, 64, 256, 64), "l1"),
    "w1000": ((1, 8, 1000, 32), "l1"),
    "c24": ((1, 128, 256, 24), "l1"),
    "w100": ((1, 256, 100, 32), "l1"),
    "c64dram": ((1, 64, 256, 64), "dram"),
    "big": ((1, 256, 512, 32), "dram"),
    "c128": ((1, 64, 256, 128), "l1"),
    "c3": ((1, 128, 256, 3), "l1"),
    "n2": ((2, 64, 256, 32), "l1"),
    "tiny_dram": ((1, 3, 100, 3), "dram"),
    "w5": ((2, 7, 5, 32), "l1"),
}
MEM = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}


@pytest.mark.parametrize("shape_id", list(SHAPES))
@pytest.mark.parametrize("arm", list(ARMS))
def test_bench(arm, shape_id, device):
    shape, buffer = SHAPES[shape_id]
    torch.manual_seed(44)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=MEM[buffer])
    if arm == "baseline":
        op_mod._PROGRAM_CACHE.clear()
        fn = lambda t: op_mod.g6_permute(t, NHWC)
    elif arm == "stock":
        fn = lambda t: ttnn.permute(t, list(NHWC))
    else:
        tt._CACHE.clear()
        cfg = tt.Cfg(**ARMS[arm], zones=ZONES, ablate=ABLATE)
        fn = lambda t: tt.permute(t, cfg)
        plan = tt.build_plan(xt, cfg)
        print(
            f"\nTTX_PLAN {arm} {shape_id} cores={plan.num_cores} units={plan.num_units} g1={plan.units_g1} "
            f"g2={plan.units_g2} kw={plan.kw} block_w={plan.block_w} tail_w={plan.tail_w} ct={plan.ct}"
        )
    first = ttnn._ttnn.get_device_operation_id()
    out = None
    for i in range(LAUNCHES):
        y = fn(xt)
        ttnn.synchronize_device(device)
        if i == 0:
            out = ttnn.to_torch(y)
        ttnn.deallocate(y)
    last = ttnn._ttnn.get_device_operation_id() - 1
    print(f"\nTTX_ARM {arm} {shape_id} first_op={first} last_op={last}")
    expected = x.permute(*NHWC).contiguous()
    if ABLATE:
        return
    assert torch.equal(out.contiguous().view(torch.int16), expected.view(torch.int16))

# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
# M2 unit check: w_all (local, 2304 x 8608 BFP8) as attention's BankLinear (kb 2, 2 cores/bank, bf16 out).
#  1. split 1/3/5 (14/42/70 cores) and staged (prefix from L1) outputs bit-equal to split 1;
#  2. traced times: linear alone, fill alone, staged linear alone;
#  3. overlap: [busy 1-core program; linear] vs [busy program + fill kernels in ONE program; staged linear].
# usage: python models/demos/inkling2_mini/tests/m2_stage_check.py <out.jsonl> [busy_us]
import json
import os
import sys
import time

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear
from models.demos.inkling2_mini.tt.mk.stage import MKDIR, StagedBankLinear, program_cores

OUT = sys.argv[1]
BUSY_US = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)


def traced_us(fn, reps=10, iters=10):
    fn()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.release_trace(dev, tid)
    return (time.perf_counter() - t0) / iters / reps * 1e6


def busy_prog(core):
    k = ttnn.KernelDescriptor(
        kernel_source=f"{MKDIR}/spin.cpp", core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))]),
        runtime_args=[(ttnn.CoreCoord(*core), [int(BUSY_US * 1350)])], config=ttnn.WriterConfigDescriptor(),
    )  # fmt: skip
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])


try:
    torch.manual_seed(0)
    K, N = 2304, 8608
    nb = dev.dram_grid_size().x
    bank_w = -(-N // (32 * nb)) * 32
    w = torch.randn(K, N) * 0.02
    x = ttnn.from_torch(torch.randn(1, 1, 1, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)  # fmt: skip
    mk = lambda s: BankLinear(
        dev,
        w,
        ttnn.bfloat8_b,
        bank_w * nb,
        kb=2,
        cores_per_bank=2,
        compute_kernel_config=ckc,
        out_cols=bank_w * nb,
        out_dtype=ttnn.bfloat16,
        split=s,
    )  # noqa: E731
    ref_bl = mk(1)
    ref = ttnn.to_torch(ref_bl(x)).float()
    print("STAGE ref done", flush=True)
    for s, staged in ((1, 384 << 10), (3, None), (5, None)):
        bl = ref_bl if s == 1 else mk(s)
        st = StagedBankLinear(bl, staged)
        y = ttnn.to_torch(bl(x)).float()
        st.fill_alone()
        ys = ttnn.to_torch(st(x)).float()
        grid = dev.compute_with_storage_grid_size()
        free = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x) if (cx, cy) not in st.cores]
        bcore = free[0]
        res = {"split": s, "cores": len(st.cores), "staged_kb_per_core": st.staged // 1024,
               "range_kb_max": max(r[3] for r in st.ranges) // 1024,
               "bit_equal_split": bool(torch.equal(y, ref)), "bit_equal_staged": bool(torch.equal(ys, ref))}  # fmt: skip
        bp = busy_prog(bcore)
        assert not (program_cores(bp) & st.cores)

        def merged():
            ttnn.generic_op([x] + st.fill_io, st.attach(busy_prog(bcore)))
            return st(x)

        ttnn.generic_op([x, x], busy_prog(bcore))
        ym = ttnn.to_torch(merged()).float()
        res["bit_equal_merged"] = bool(torch.equal(ym, ref))
        if os.environ.get("M2_EAGER_ONLY"):  # mockwarm: compile every program, no traces
            print("M2STAGE", json.dumps(res), flush=True)
            ttnn.deallocate(st.stage)
            continue
        res["linear_us"] = round(traced_us(lambda: bl(x)), 1)
        res["fill_alone_us"] = round(traced_us(st.fill_alone), 1)
        res["staged_linear_us"] = round(traced_us(lambda: st(x)), 1)
        res["busy_us"] = round(traced_us(lambda: ttnn.generic_op([x, x], busy_prog(bcore))), 1)
        res["serial_busy_then_linear_us"] = round(
            traced_us(lambda: (ttnn.generic_op([x, x], busy_prog(bcore)), bl(x))), 1
        )
        res["merged_busy_fill_then_staged_us"] = round(traced_us(merged), 1)
        res["saved_us"] = round(res["serial_busy_then_linear_us"] - res["merged_busy_fill_then_staged_us"], 1)
        print("M2STAGE", json.dumps(res), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
        ttnn.deallocate(st.stage)
finally:
    ttnn.close_device(dev)

# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
# Unit check of tt/mk/pipe_stage.py on a w_all-sized BankLinear (2304 x 8608 BFP8, kb 2, 2 cores/bank, split 3):
#  1. outputs from the prefetcher ring bit-equal to the BankLinear, two layers interleaved, eager and traced;
#  2. traced us per call: [busy 1-core program; linear from DRAM] vs [busy; linear from the ring], per busy time;
#  3. the model's request pattern (one slab ahead, each step queues [1.., 0]), eager and traced, then a drain.
# usage: python models/demos/inkling2_mini/tests/pipe_stage_check.py <out.jsonl> [busy_us,...]
# env: PIPECHK_KN=K,N  PIPECHK_PC=<pass_cols>  PIPECHK_SPLIT=<cores per group>  PIPECHK_EXROWS=<rows kept free>
import json
import sys
import time

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear
from models.demos.inkling2_mini.tt.mk.pipe_stage import PipeStage
from models.demos.inkling2_mini.tt.mk.stage import MKDIR, resplit

OUT = sys.argv[1]
BUSY = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [60.0]
dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
res = {}


def log(**kw):
    print("PIPECHK", json.dumps(kw), flush=True)
    res.update(kw)


def traced_us(fn, reps=10, iters=10, pre=None):
    if pre:
        pre(False, 1)
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    if pre:
        pre(True, reps)
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


def busy(us):
    core = ttnn.CoreCoord(0, 0)
    k = ttnn.KernelDescriptor(kernel_source=f"{MKDIR}/spin.cpp", core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                              runtime_args=[(core, [int(us * 1350)])], config=ttnn.WriterConfigDescriptor())  # fmt: skip
    dummy = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
    return lambda: ttnn.generic_op([dummy, dummy], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))


try:
    log(supported=ttnn.experimental.is_tensor_prefetcher_supported(dev))
    torch.manual_seed(0)
    import os

    K, N = [int(v) for v in os.environ.get("PIPECHK_KN", "2304,8608").split(",")]
    pc = int(os.environ.get("PIPECHK_PC", "0")) or None
    sp = int(os.environ.get("PIPECHK_SPLIT", "3"))
    nb = dev.dram_grid_size().x
    bank_w = -(-N // (32 * nb)) * 32
    x = ttnn.from_torch(torch.randn(1, 1, 1, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)  # fmt: skip
    gx = dev.compute_with_storage_grid_size().x
    ex = [(cx, cy) for cy in range(int(os.environ.get("PIPECHK_EXROWS", "6"))) for cx in range(gx)]
    bls, refs = {}, {}
    for L in range(2):
        w = torch.randn(K, N) * 0.02
        bl = BankLinear(dev, w, ttnn.bfloat8_b, bank_w * nb, kb=2, cores_per_bank=2, compute_kernel_config=ckc,
                        out_cols=N, out_dtype=ttnn.bfloat16)  # fmt: skip
        refs[L] = ttnn.to_torch(bl(x))
        if pc:
            bl = BankLinear(dev, w, ttnn.bfloat8_b, bank_w * nb, kb=2, cores_per_bank=2, compute_kernel_config=ckc,
                            out_cols=N, out_dtype=ttnn.bfloat16, pass_cols=pc)  # fmt: skip
        bls[L] = resplit(bl, sp, ex)
        if L == 0:
            bls[L] = bls[L]
        else:
            from models.demos.inkling2_mini.tt.mk.pipe_stage import place_like

            bls[L] = place_like(bls[L], bls[0])
    ps = PipeStage(
        dev, bls, int(os.environ.get("PIPECHK_ENTRY", "32")), dual=bool(int(os.environ.get("PIPECHK_DUAL", "0")))
    )
    log(
        KN=[K, N],
        pc=pc,
        split=sp,
        dual=ps.pipes is not None and len(ps.pipes),
        cores=len(bls[0].cores),
        entries=ps.n_entries,
        ring_kb=ps.ring_bytes // 1024,
    )
    ps.start()
    order = [0, 1, 0, 1, 1]
    ps.request(order)
    eq = [bool(torch.equal(ttnn.to_torch(ps(L, x)), refs[L])) for L in order]
    ttnn.synchronize_device(dev)
    log(eager_bit_equal=eq)
    for us in BUSY:
        spin = busy(us)
        t_dram = traced_us(lambda: (spin(), bls[0](x)))
        t_spin = traced_us(spin)
        k = {"i": 0}

        def ring_call():
            spin()
            ps(k["i"] % 2, x)
            k["i"] += 1

        t_ring = traced_us(
            ring_call, pre=lambda cap, n: (k.update(i=0), ps.request([j % 2 for j in range(n)], capture=cap))
        )
        log(
            KN=[K, N],
            split=sp,
            busy=us,
            us_spin=round(t_spin, 1),
            us_lin_dram=round(t_dram - t_spin, 1),
            us_lin_ring=round(t_ring - t_spin, 1),
        )
    # traced replay bit-equality: one more traced run whose outputs are read back
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    ps.request([0, 1], capture=True)
    ya, yb = ps(0, x), ps(1, x)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    good = []
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        good.append(bool(torch.equal(ttnn.to_torch(ya), refs[0])) and bool(torch.equal(ttnn.to_torch(yb), refs[1])))
    ttnn.release_trace(dev, tid)
    log(traced_bit_equal=good)
    # the model's pattern (PipeOverlap): one app-0 slab ahead, each step queues [1.., 0] before its first reader
    order = [0, 1, 1, 0]
    ps.request(order[:1])
    print("PIPECHK rot start", flush=True)
    for st in range(2):  # eager steps
        ys = []
        for i, L in enumerate(order):
            if i == 0:
                ps.request(order[1:] + order[:1], capture=True)
            ys.append(ps(L, x))
        log(**{f"rot_eager{st}": [bool(torch.equal(ttnn.to_torch(y), refs[L])) for y, L in zip(ys, order)]})
    tids, outs = [], []
    for ph in range(2):
        tids.append(ttnn.begin_trace_capture(dev, cq_id=0))
        ys = []
        for i, L in enumerate(order):
            if i == 0:
                ps.request(order[1:] + order[:1], capture=True)
            ys.append(ps(L, x))
        outs.append(ys)
        ttnn.end_trace_capture(dev, tids[-1], cq_id=0)
    good = []
    for r in range(4):
        ttnn.execute_trace(dev, tids[r % 2], cq_id=0, blocking=True)
        good.append(all(bool(torch.equal(ttnn.to_torch(y), refs[L])) for y, L in zip(outs[r % 2], order)))
    log(rot_traced=good)
    for t in tids:
        ttnn.release_trace(dev, t)
    ps.drain(order[0])
    ps.stop()
finally:
    json.dump(res, open(OUT, "a"))
    open(OUT, "a").write("\n")
    ttnn.close_device(dev)

# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
# Unit check of tt/mk/expert_pipe.py: PF-layout BankMoeExperts reading guessed experts from the prefetcher ring
# vs the same op from DRAM (bit-equal, several masks; the guess-out mask), and traced us per step with the requests
# free or paced by the op signal. usage: python models/demos/inkling2_mini/tests/expert_ring_check.py
# env: ER_DT=bfp4,mixed  ER_NG=0,3  ER_BUSY=40,100  ER_RING_KB=300
import json
import math
import os
import time

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.moe_bank import BankMoeExperts, pf_weights
from models.demos.inkling2_mini.tt.mk.expert_pipe import ExpertPipe
from models.demos.inkling2_mini.tt.mk.stage import MKDIR

H, I, K, n = 2304, 1152, 8, 128
DT = {
    "bfp4": (ttnn.bfloat4_b, ttnn.bfloat4_b),
    "mixed": (ttnn.bfloat4_b, ttnn.bfloat8_b),
    "bfp8": (ttnn.bfloat8_b, ttnn.bfloat8_b),
}
RING_KB = int(os.environ.get("ER_RING_KB", "300"))
dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)


def log(**kw):
    print("ERING", json.dumps(kw), flush=True)


def traced_us(body, reps=10, iters=10):
    body(False)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(reps):
        body(True)
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
    k = ttnn.KernelDescriptor(
        kernel_source=f"{MKDIR}/spin.cpp",
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        runtime_args=[(core, [int(us * 1350)])],
        config=ttnn.WriterConfigDescriptor(),
    )
    dummy = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
    return lambda: ttnn.generic_op([dummy, dummy], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))


ep = None
try:
    g = torch.Generator().manual_seed(1)
    gu = torch.cat(
        [
            torch.randn(n, H, I, generator=g, dtype=torch.bfloat16) * (3 / math.sqrt(H)),
            torch.randn(n, H, I, generator=g, dtype=torch.bfloat16) * (8 / math.sqrt(H)),
        ],
        2,
    )
    dn = torch.randn(n, I, H, generator=g, dtype=torch.bfloat16) / math.sqrt(I)
    l1 = ttnn.L1_MEMORY_CONFIG
    put = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1)
    names = os.environ.get("ER_DT", "bfp4,mixed").split(",")
    ops = {}
    for name in names:
        dg, dd = DT[name]
        ops[name] = BankMoeExperts(
            dev, pf_weights(dev, gu, dg, "experts.w_gate_up"), pf_weights(dev, dn, dd, "experts.w_down")
        )
    del gu, dn
    ref_ops = {k: BankMoeExperts(dev, (v.w_gate_up, v.gu_x), (v.w_down, v.dn_x)) for k, v in ops.items()}
    ep = ExpertPipe(dev, list(ops.values()), RING_KB << 10)
    ep.start()
    log(ring_kb=ep.ring_bytes >> 10, signal=ep.signal)

    def mask_t(sel):
        m = torch.zeros(1, 1, 1, n, dtype=torch.int32)
        m[..., list(sel)] = 1
        return ttnn.from_torch(
            m, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    mout = mask_t([])
    for name, op in ops.items():
        ids = torch.randperm(n, generator=g)[:K]
        others = [e for e in range(n) if e not in ids.tolist()]
        w = torch.rand(K, generator=g) + 0.5
        x = torch.randn(H, generator=g).bfloat16()
        xt, wt = put(x.reshape(1, 1, 1, H), ttnn.bfloat16), put(w.reshape(1, 1, 1, K).bfloat16(), ttnn.bfloat16)
        it = put(ids.reshape(1, 1, 1, K).to(torch.int32), ttnn.uint16)
        ref = ttnn.to_torch(ref_ops[name](xt, wt, it))
        wb = w.bfloat16()
        want = sorted(ids[torch.argsort(wb.float(), descending=True)[:3]].tolist())
        cases = {"3hit": ids[:3].tolist(), "2hit1miss": ids[3:5].tolist() + others[:1], "1miss": others[1:2]}
        eq = {}
        for cn, sel in cases.items():
            mt = mask_t(sel)
            ttnn.synchronize_device(dev)
            ep.request(op, mt)
            y = op(xt, wt, it, guess=mt, guess_out=mout, signal=ep.signal)
            ep.stepped()
            eq[cn] = bool(torch.equal(ttnn.to_torch(y).view(torch.int16), ref.view(torch.int16)))
            got = sorted(torch.nonzero(ttnn.to_torch(mout).reshape(-1).int()).reshape(-1).tolist())
            eq[cn + "_mask"] = got == want
        log(dtype=name, eager_bit_equal=eq)
        for ng in [int(v) for v in os.environ.get("ER_NG", "0,3").split(",")]:
            mt = mask_t(ids[:ng].tolist() if ng > 0 else others[:1])
            ttnn.synchronize_device(dev)
            for us in [float(v) for v in os.environ.get("ER_BUSY", "40,100").split(",")]:
                spin = busy(us)
                t_spin = traced_us(lambda cap: spin())
                t_dram = traced_us(lambda cap: (spin(), ref_ops[name](xt, wt, it)))

                def body(cap, paced):
                    ep.request(op, mt, capture=cap) if paced else ttnn.experimental.queue_tensor_prefetcher_request(
                        dev,
                        [(op.w_gate_up, 2, [], mt), (op.w_down, 1, [], mt)],
                        prefetcher_pipes=ep.pipes,
                        capture_into_trace=cap,
                    )
                    spin()
                    op(xt, wt, it, guess=mt, signal=ep.signal if paced else None)
                    if paced:
                        ep.stepped()

                t_free = traced_us(lambda cap: body(cap, False))
                ep.close()
                t_paced = traced_us(lambda cap: body(cap, True))
                ep.close()
                ttnn.synchronize_device(dev)
                log(
                    dtype=name,
                    busy=us,
                    guesses=ng,
                    us_spin=round(t_spin, 1),
                    us_dram=round(t_dram - t_spin, 1),
                    us_ring_free=round(t_free - t_spin, 1),
                    us_ring_paced=round(t_paced - t_spin, 1),
                )
    ep.close()
    ep.stop()
finally:
    ttnn.close_device(dev)

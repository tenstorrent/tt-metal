"""B47c unit test (1x8 slot, MESH_DEVICE=1x8): fused MTP rms-norm program (hy4_rms_fused) vs the stock fp32 op chain of Hy4MTP._rms6 (+typecast/add). RES lines:
count of bf16 elements that differ, max abs diff in bf16 ulps, fp32 sum (x_new) bit-identity, time per call (host-side wall, trace-free)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt import hy4_rms_fused as RF
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

HID = 6144
EPS = 1e-5


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def d0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def rms6(x32, wsc):
    sq = ttnn.multiply(x32, x32)
    ss = ttnn.sum(sq, dim=3, keepdim=True)
    r = ttnn.rsqrt(ttnn.add(ss, EPS * HID))
    y = ttnn.multiply(x32, r)
    return ttnn.multiply(y, wsc, dtype=ttnn.bfloat16)


def cmp(name, a, b, S):
    a = a[..., :S, :].float()
    b = b[..., :S, :].float()
    ne = (a != b).sum().item()
    ulp = (a - b).abs() / (b.abs().clamp_min(1e-30) * 2.0**-8)
    res(case=name, n=a.numel(), ne=ne, max_ulp=float(ulp.max()), mean_abs=float((a - b).abs().mean()))


def main():
    S = int(os.environ.get("S", "4"))
    full, md = open_row()
    try:
        g = torch.Generator().manual_seed(5)
        w = torch.randn(1, 1, 1, HID, generator=g) * 0.5 + 1.0
        wsc = upload(md, (w * HID**0.5).expand(1, 1, S, HID).contiguous(), ttnn.float32)
        wsc1 = upload(md, (w * HID**0.5).contiguous(), ttnn.float32)
        for trial in range(int(os.environ.get("N", "3"))):
            xh = torch.randn(1, 1, S, HID, generator=g) * (3.0 if trial % 2 else 0.2)
            rh = torch.randn(1, 1, S, HID, generator=g) * 0.7
            xb = upload(md, xh.bfloat16(), ttnn.bfloat16)
            x32 = upload(md, xh.float(), ttnn.float32)
            rb = upload(md, rh.bfloat16(), ttnn.bfloat16)
            # case A: bf16 x, no residual (pre)
            ref = rms6(ttnn.typecast(xb, ttnn.float32), wsc1)
            out = RF.rms_fused(xb, wsc, EPS)
            cmp(f"A_t{trial}", d0(out), d0(ref), S)
            # case B: fp32 x + bf16 residual, x_new out
            sref = ttnn.add(x32, ttnn.typecast(rb, ttnn.float32))
            ref = rms6(sref, wsc1)
            out, xn = RF.rms_fused(x32, wsc, EPS, res=rb, out_x=True)
            cmp(f"B_t{trial}", d0(out), d0(ref), S)
            res(case=f"B_xnew_t{trial}", xn_ne=int((d0(xn)[..., :S, :] != d0(sref)[..., :S, :]).sum()))
            # case C: bf16 x + bf16 residual
            sref = ttnn.add(ttnn.typecast(xb, ttnn.float32), ttnn.typecast(rb, ttnn.float32))
            ref = rms6(sref, wsc1)
            out = RF.rms_fused(xb, wsc, EPS, res=rb)
            cmp(f"C_t{trial}", d0(out), d0(ref), S)
        # cat: two writes into one [1,1,S,2D] tensor
        cat = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, S, 2 * HID]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        RF.rms_fused(xb, wsc, EPS, out=cat, out_off=0)
        RF.rms_fused(rb, wsc, EPS, out=cat, out_off=HID // 32)
        ref = ttnn.concat(
            [rms6(ttnn.typecast(xb, ttnn.float32), wsc1), rms6(ttnn.typecast(rb, ttnn.float32), wsc1)], dim=3
        )
        cmp("cat", d0(cat), d0(ref), S)

        # device time per call: trace of 20 calls, executed 5 times
        def timed(fn, name):
            fn()
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(20):
                fn()
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.synchronize_device(md)
            t0 = time.time()
            for _ in range(5):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res(**{name: (time.time() - t0) / 100 * 1e6})
            ttnn.release_trace(md, tid)

        def stockB():
            sr = ttnn.add(x32, ttnn.typecast(rb, ttnn.float32))
            rms6(sr, wsc1)

        timed(lambda: RF.rms_fused(x32, wsc, EPS, res=rb, out_x=True), "fused_B_us")
        timed(stockB, "stock_B_us")
        timed(lambda: RF.rms_fused(xb, wsc, EPS), "fused_A_us")
        timed(lambda: rms6(ttnn.typecast(xb, ttnn.float32), wsc1), "stock_A_us")
    finally:
        close_row(full, md)


main()

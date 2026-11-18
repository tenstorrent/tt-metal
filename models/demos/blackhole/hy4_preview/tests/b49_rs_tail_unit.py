"""B49 unit (1x8 slot): rs_tail (hy4_rs_tail) vs the stock rs-exact tail ops of Hy4MoeTp32PrefillBlock (slice x4, add x3 in the order 0..3, typecast bf16, to_layout, reshape, embedding of the own rows,
reshape). yg fp32 [1,1,4 Tt,Dc] random (n=512, Tt=2048, Dc=768), own row index R in 0..3 (the rowbase word). Bit equality / ulp counts, 200-iteration soak, traced us. RES lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_rs_tail import rs_tail

N, DC = 512, 768
TT = 4 * N


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def tr(md, fn, reps=3, n=5):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return round(min(ts), 1)


def snap(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def stock(md, yg, sel_idx):
    acc = ttnn.slice(yg, [0, 0, 0, 0], [1, 1, TT, DC])
    for k in range(1, 4):
        blk = ttnn.slice(yg, [0, 0, k * TT, 0], [1, 1, (k + 1) * TT, DC])
        acc = ttnn.add(acc, blk)
    sb = ttnn.typecast(acc, ttnn.bfloat16)
    sb = ttnn.to_layout(sb, ttnn.ROW_MAJOR_LAYOUT)
    sb = ttnn.reshape(sb, [1, 1, TT, DC])
    y3 = ttnn.embedding(sel_idx, sb, layout=ttnn.TILE_LAYOUT)
    return ttnn.reshape(y3, [1, 1, N, DC])


def main():
    full, md = open_row()
    try:
        g = torch.Generator().manual_seed(int(os.environ.get("SEED", "1")))
        for R in (0, 1, 3):
            yh = torch.randn(1, 1, 4 * TT, DC, generator=g) * float(os.environ.get("AMP", "3.0"))
            yg = upload(md, yh, ttnn.float32)
            rb = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            rb[0, 0, 0, 0] = R * (N // 32)
            rowbase = upload(md, rb, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            ids = (R * N + torch.arange(N, dtype=torch.int32)).reshape(1, N)
            sel_idx = upload(md, ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            ref = snap(stock(md, yg, sel_idx))
            got = snap(rs_tail(md, yg, rowbase, N, DC))
            ulp = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
            res(
                event="exact",
                R=R,
                equal=bool(torch.equal(got, ref)),
                frac_ne=float((ulp != 0).float().mean()),
                max_ulp=int(ulp.max()),
                n_elems=int(got.numel()),
            )
            bad = 0
            for it in range(int(os.environ.get("ITER", "200"))):
                o = rs_tail(md, yg, rowbase, N, DC)
                if it % 20 == 0 and not torch.equal(snap(o), got):
                    bad += 1
                ttnn.deallocate(o)
            ttnn.synchronize_device(md)
            res(event="soak", R=R, mismatches=bad)
            if R == 1:
                res(
                    event="time_us",
                    stock=tr(md, lambda: stock(md, yg, sel_idx), reps=1),
                    fused=tr(md, lambda: rs_tail(md, yg, rowbase, N, DC), reps=1),
                )
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()

"""B49 iHC prefill micro-bench / unit (1x8 slot): stock Hy4IHCGen.collapse / pre_post / expand vs candidates; x4 [1,4,512,6144] fp32. Prints RES lines (traced us per call, PCC / max rel err vs stock)."""
import json
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4IHCGen
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

N, D = 512, 6144


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
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def main():
    full, md = open_row()
    try:
        g = torch.Generator().manual_seed(1)
        fn8 = torch.randn(8, 4 * D, generator=g) * 0.01
        ihc = Hy4IHCGen(md, fn8, torch.randn(8, generator=g) * 0.1, torch.tensor([0.5, 0.5]), 1e-6, 1e-6)
        x = upload(md, torch.randn(1, 4, N, D, generator=g) * 2.0, ttnn.float32)
        pre, post = ihc.pre_post(x)
        yb32 = upload(md, torch.randn(1, 1, N, D, generator=g), ttnn.float32)
        res(
            event="stock",
            pre_post_us=tr(md, lambda: ihc.pre_post(x)),
            collapse_us=tr(md, lambda: ihc.collapse(x, pre)),
            expand_us=tr(md, lambda: ttnn.add(x, ttnn.multiply(post, yb32))),
        )
        ref = snap(ihc.collapse(x, pre))

        # B: per-stream slices, bcast multiply, add chain
        def coll_b(x4, pre):
            acc = None
            for i in range(4):
                xi = ttnn.slice(x4, [0, i, 0, 0], [1, i + 1, N, D])
                pi = ttnn.slice(pre, [0, i, 0, 0], [1, i + 1, N, 1])
                t = ttnn.multiply(xi, pi)
                acc = t if acc is None else ttnn.add(acc, t)
            return acc

        out = coll_b(x, pre)
        d = (snap(out) - ref).abs().max().item()
        res(event="collapse_B", us=tr(md, lambda: coll_b(x, pre)), maxdiff=d, refmax=ref.abs().max().item())

        # C: multiply then sum via matmul-free reshape: [4,n,D] -> sum dim 0 as ttnn.sum(dim=0)
        def coll_c(x4, pre):
            w = ttnn.multiply(x4, pre)
            return ttnn.sum(ttnn.reshape(w, [4, 1, N, D]), dim=0, keepdim=True)

        out = coll_c(x, pre)
        res(
            event="collapse_C",
            us=tr(md, lambda: coll_c(x, pre)),
            maxdiff=(snap(out).reshape(-1) - ref.reshape(-1)).abs().max().item(),
        )
        # sections of the stock collapse
        res(
            event="collapse_parts",
            mul_us=tr(md, lambda: ttnn.multiply(x, pre)),
            sum_us=tr(md, lambda: ttnn.sum(ttnn.multiply(x, pre), dim=1, keepdim=True)),
        )
        # pre_post parts
        sq = lambda: ttnn.sum(ttnn.multiply(x, x), dim=3, keepdim=True)
        res(
            event="pre_parts",
            sq_sum_us=tr(md, sq),
            mm_us=tr(md, lambda: ttnn.matmul(x, ihc.w, compute_kernel_config=ihc.ckc, dtype=ttnn.float32)),
        )
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()

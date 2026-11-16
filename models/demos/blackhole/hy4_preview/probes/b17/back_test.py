"""B17 stage 1: fused_back (w_uv matmul + concat heads + gate) vs the stock 3-op chain, PCC/maxdiff over varying inputs + traced us/call. 1x8 slot, no fabric needed."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA

NCALL = int(os.environ.get("NCALL", "6"))
S = int(os.environ.get("S", "1"))


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def traced_us(md, fn, n=7, reps=5):
    def body():
        for _ in range(reps):
            o = fn()
            ttnn.deallocate(o)

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        g_ = torch.Generator().manual_seed(3)
        sh = lambda t, d: ttnn.from_torch(
            t,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(md, dim=d),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w = sh((torch.randn(1, 64, 512, 256, generator=g_) * 0.05), 1)
        ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        for gfold in (True, False):
            worst = 1.0
            worst_md = 0.0
            for it in range(NCALL):
                o = sh(torch.randn(1, 64, S, 512, generator=g_), 1)
                gr = sh(torch.randn(1, 1, S, 64 * 256, generator=g_) * 2, 3)
                gg = gr if gfold else ttnn.sigmoid(gr)
                x = ttnn.matmul(o, w, compute_kernel_config=ckc)
                x = ttnn.experimental.nlp_concat_heads(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if gfold:
                    ref = ttnn.multiply(
                        x, gg, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)]
                    )
                else:
                    ref = ttnn.multiply(x, gg)
                got = FA.fused_back(o, w, gg, gfold)
                ttnn.synchronize_device(md)
                rt = ttnn.to_torch(ref, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[:, 0, :S, :2048].float()
                gt = ttnn.to_torch(got, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[:, 0, :S, :2048].float()
                p = pcc(rt, gt)
                mdiff = float((rt - gt).abs().max())
                eq = bool(torch.equal(rt, gt))
                worst = min(worst, p)
                worst_md = max(worst_md, mdiff)
                R(event="call", gfold=gfold, it=it, pcc=p, maxdiff=mdiff, bit_exact=eq, refmax=float(rt.abs().max()))
            R(event="summary", gfold=gfold, worst_pcc=worst, worst_maxdiff=worst_md)
        o = sh(torch.randn(1, 64, S, 512, generator=g_), 1)
        gr = sh(torch.randn(1, 1, S, 64 * 256, generator=g_), 3)

        def stock():
            x = ttnn.matmul(o, w, compute_kernel_config=ckc)
            x = ttnn.experimental.nlp_concat_heads(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.multiply(x, gr, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)])

        if os.environ.get("PROF", "0") == "1":
            for nm, fn in (("stock", stock), ("fused", lambda: FA.fused_back(o, w, gr, True))):
                for _ in range(2):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                for _ in range(4):
                    ttnn.deallocate(fn())
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(md)
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(md)
                R(event="prof", which=nm)
        R(
            event="timing",
            stock_us=round(traced_us(md, stock), 1),
            fused_us=round(traced_us(md, lambda: FA.fused_back(o, w, gr, True)), 1),
        )
    finally:
        ttnn.close_mesh_device(md)


main()

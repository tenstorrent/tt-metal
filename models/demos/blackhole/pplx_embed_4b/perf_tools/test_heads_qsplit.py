# fused heads+norm+rotary: QWEN_HEADSPLIT_Q_SPLIT=2 vs 1 must be bit-identical; traced timing of both (model dtypes: bfp8 Q/K/V out)
import os
import time

import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import make_norm_constants
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.op import (
    nlp_create_qkv_heads_norm_headsplit,
)
from models.tt_transformers.tt.common import get_rot_transformation_mat

NH, NKV, DH, EPS, S = 32, 8, 128, 1e-6, 512


def traced(dev, fn, n=6):
    for _ in range(2):
        [ttnn.deallocate(t) for t in fn()]
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    outs = [fn() for _ in range(n)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(8):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    us = (time.perf_counter() - t0) / 8 / n * 1e6
    ttnn.release_trace(dev, tid)
    [ttnn.deallocate(t) for o in outs for t in o]
    return us


D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
try:
    gq = torch.rand(DH) + 0.5
    gk = torch.rand(DH) + 0.5
    GQ, GK, SC, EP = make_norm_constants(gq, gk, EPS, D)
    ang = torch.rand(1, 1, S, DH) * 6.28
    L1 = ttnn.L1_MEMORY_CONFIG
    cos = ttnn.from_torch(torch.cos(ang), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    sin = ttnn.from_torch(torch.sin(ang), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    T = ttnn.from_torch(
        get_rot_transformation_mat(32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1
    )
    for B, dt, mc in ((1, ttnn.bfloat8_b, L1), (8, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG)):
        t = ttnn.from_torch(
            torch.randn(B, 1, S, (NH + 2 * NKV) * DH), dtype=dt, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc
        )

        def run(split):
            os.environ["QWEN_HEADSPLIT_Q_SPLIT"] = str(split)
            return nlp_create_qkv_heads_norm_headsplit(
                t,
                GQ,
                GK,
                SC,
                EP,
                num_heads=NH,
                num_kv_heads=NKV,
                memory_config=mc,
                rot_cos=cos,
                rot_sin=sin,
                trans_mat=T,
                q_dtype=ttnn.bfloat8_b,
                kv_dtype=ttnn.bfloat8_b,
            )

        o1 = run(1)
        o2 = run(2)
        for name, a, b in zip("qkv", o1, o2):
            ta, tb = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
            print(
                f"[eq] B={B} {name}: max|d|={(ta-tb).abs().max().item():.6f} identical={bool(torch.equal(ta,tb))}",
                flush=True,
            )
        [ttnn.deallocate(z) for z in o1 + o2]
        us1 = traced(D, lambda: run(1))
        us2 = traced(D, lambda: run(2))
        print(f"[perf] B={B}: q_split=1 {us1:8.1f} us   q_split=2 {us2:8.1f} us  ({100*(us2/us1-1):+.1f}%)", flush=True)
        ttnn.deallocate(t)
    print("[done]")
finally:
    ttnn.close_device(D)

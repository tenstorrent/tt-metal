"""Device helpers for the Hy4 decode split kernel tests (run inside the DSV4 image, see program/decode-kernel/run_dk.sh). One chip, 1x1 mesh."""
import importlib.util
import json
import os
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.path.join(os.path.dirname(HERE), "tt")
SCALE = 256**-0.5
SENT = 0xFFFFFFFF


def load_split():
    spec = importlib.util.spec_from_file_location("hy4_split", os.path.join(TT, "hy4_split.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def to_dev(md, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, device=md, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def ref_sparse(q, kv, idx, sink_logit, v_dim=512, with_stats=False):
    """q [H,S,D] bf16, kv [T,D], idx [S,K] int64 (-1 masked, any position), sink [H] (logit domain) or None. fp32. -> out [H,S,v] (+ m [H,S] scaled max, l [H,S] sum exp(s-m) when with_stats; sink excluded there)"""
    H, S, _ = q.shape
    out = torch.zeros(H, S, v_dim)
    M = torch.full((H, S), float("-inf"))
    L = torch.zeros(H, S)
    kvf = kv.float()
    for s in range(S):
        ix = idx[s]
        ix = ix[ix >= 0]
        if ix.numel() == 0:
            continue
        k = kvf[ix]
        sc = (q[:, s].float() @ k.T) * SCALE
        m = sc.max(dim=1).values
        mm = torch.maximum(m, sink_logit) if (sink_logit is not None and not with_stats) else m
        p = torch.exp(sc - mm[:, None])
        den = p.sum(dim=1)
        if sink_logit is not None and not with_stats:
            den = den + torch.exp(sink_logit - mm)
        out[:, s] = (p @ k[:, :v_dim]) / den[:, None]
        M[:, s] = m
        L[:, s] = p.sum(dim=1) if with_stats else den
    return (out, M, L) if with_stats else out


def make_idx(S, topk, regions, valid):
    idx = torch.full((S, topk), -1, dtype=torch.int64)
    for s in range(S):
        lo, hi = regions[s]
        n = min(valid, hi - lo)
        idx[s, :n] = lo + torch.randperm(hi - lo)[:n]
    return idx


def idx_dev(md, idx):
    S, K = idx.shape
    return to_dev(md, idx.to(torch.int32).reshape(1, 1, S, K), ttnn.uint32)  # -1 -> 0xFFFFFFFF


def ckc_of(md, fid="hifi4"):
    return ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity={"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2}[fid],
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def timed_traced(md, call, n=10, reps=3):
    for _ in range(2):
        call()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(n):
        call()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    us = (time.perf_counter() - t0) / (n * reps) * 1e6
    ttnn.release_trace(md, tid)
    return round(us, 1)


def emit(res):
    print("RES " + json.dumps(res), flush=True)

# SPDX-License-Identifier: Apache-2.0
"""tp32-prefill-build E2': stock-op floor of the TP32 prefill grouped GEMM on ONE chip: batched (over 256 experts) matmuls with capacity Mc rows per expert.
gate/up: [1,E,Mc,6144] x [1,E,6144,128] (64 gate + 64 up cols of this chip's slice, bfp4); down (layout v2 G=4): [1,E,Mc,256] x [1,E,256,1536] bfp4.
Env: MCS=32,64,128 E=256 OUT=json. Traced us per call (median)."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc

E = int(os.environ.get("E", "256"))
MCS = [int(x) for x in os.environ.get("MCS", "32,64,128").split(",")]
FID = os.environ.get("FID", "lofi")


def traced_us(md, fn, n=10, rep_=5):
    def body():
        for _ in range(rep_):
            ttnn.deallocate(fn())

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
        ts.append((time.perf_counter() - t0) / rep_ * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=16384, trace_region_size=100_000_000)
    cfg = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi if FID == "lofi" else ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    res = {}
    torch.manual_seed(0)
    for name, K, N in (("gateup", 6144, 128), ("down", 256, 1536)):
        w = torch.randn(1, E, K, N) * 0.02
        wt = ttnn.from_torch(
            w.bfloat16(),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for Mc in MCS:
            try:
                x = torch.randn(1, E, Mc, K)
                xt = ttnn.from_torch(
                    x.bfloat16(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                f = lambda: ttnn.matmul(
                    xt, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                o = f()
                if Mc == MCS[0]:
                    ref = torch.matmul(x.bfloat16().float(), ttnn.to_torch(wt).float())
                    p(name, Mc, "pcc", pcc(ref, ttnn.to_torch(o).float()))
                ttnn.deallocate(o)
                us = traced_us(md, f)
                fl = 2 * E * Mc * K * N
                res[f"{name}_Mc{Mc}"] = us
                p(
                    name,
                    "Mc",
                    Mc,
                    "us",
                    round(us, 1),
                    "TFLOP/s",
                    round(fl / us / 1e6, 1),
                    "wbytes_MB",
                    round(E * K * N * 0.5625 / 1e6, 1),
                )
                ttnn.deallocate(xt)
            except Exception as e:
                p(name, Mc, "FAIL", str(e)[:300])
                res[f"{name}_Mc{Mc}"] = "fail"
        ttnn.deallocate(wt)
    if os.environ.get("OUT"):
        json.dump(res, open(os.environ["OUT"], "w"), indent=1)
    ttnn.close_mesh_device(md)


main()

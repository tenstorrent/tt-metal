# SPDX-License-Identifier: Apache-2.0
"""Does sequence-parallel reduce the comm cost for our decode? Compare all_reduce vs
reduce_scatter+all_gather on [1,1,B,H] (SP shards the token/batch dim). Small latency-bound
tensors may make RS+AG (2 ops) slower than AR (1 op)."""
import time
import torch, ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
TP = 8


def bench(name, fn, md, iters=200, warmup=20):
    for _ in range(warmup): fn()
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    ttnn.synchronize_device(md)
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:38s} {ms:7.4f} ms", flush=True)
    return ms


def main():
    cfg = Qwen36ModelConfig(); H = cfg.hidden_size
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    REP = ttnn.ReplicateTensorToMesh(md)
    f = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=REP)
    RING, LIN = ttnn.Topology.Ring, ttnn.Topology.Linear
    try:
        for B in (8, 32):
            x = f(torch.randn(1, 1, B, H) * 0.1)
            print(f"[B={B}] comm on [1,1,{B},{H}]", flush=True)
            ar_r = bench("all_reduce Ring", lambda: ttnn.all_reduce(x, cluster_axis=1, topology=RING), md)
            ar_l = bench("all_reduce Linear", lambda: ttnn.all_reduce(x, cluster_axis=1, topology=LIN), md)
            def rs_ag():
                # SP: reduce-scatter the token dim (2), then all-gather back
                s = ttnn.reduce_scatter(x, dim=2, cluster_axis=1, topology=LIN)
                return ttnn.all_gather(s, dim=2, cluster_axis=1, topology=LIN)
            try:
                sp = bench("reduce_scatter+all_gather (SP)", rs_ag, md)
                print(f"  -> SP/AR(ring) ratio = {sp/ar_r:.2f}  ({'SP faster' if sp<ar_r else 'AR faster'})", flush=True)
            except Exception as e:
                print(f"  RS+AG FAIL: {type(e).__name__}: {str(e)[:140]}", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()

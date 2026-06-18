# SPDX-License-Identifier: Apache-2.0
"""Attribute the 8-chip TP decode floor: device-bound per-call time of each TP component
+ the all_reduce CCL cost, on a 1x8 line. Tells us what to actually optimize."""
import time, statistics
import torch
import ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig

TP = 8
DT_W, DT_A = ttnn.bfloat8_b, ttnn.bfloat16


def bench(name, fn, md, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(md)
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:34s} {ms:7.3f} ms/call", flush=True)
    return ms


def main():
    cfg = Qwen36ModelConfig()
    H, I = cfg.hidden_size, cfg.intermediate_size
    nv, nk, dk, dv = cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    Kc = cfg.linear_conv_kernel_dim
    key_dim, val_dim = dk * nk, dv * nv
    nvp, nkp = nv // TP, nk // TP
    ratio = nv // nk
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    sh = (1, TP)
    SH = lambda d: ttnn.ShardTensor2dMesh(md, dims=(None, d), mesh_shape=sh)
    REP = ttnn.ReplicateTensorToMesh(md)
    rnd = lambda *s: torch.randn(*s) * 0.02
    fW = lambda t, d: ttnn.from_torch(t.unsqueeze(0).unsqueeze(0), dtype=DT_W, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(d))
    fR = lambda t: ttnn.from_torch(t, dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=REP)
    fS = lambda t, d: ttnn.from_torch(t, dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(d))
    try:
        x = fR(torch.randn(1, 1, 1, H) * 0.1)
        AR = lambda t: ttnn.all_reduce(t, cluster_axis=1, topology=ttnn.Topology.Linear)

        print("[device-bound per-call on 1x8 line]", flush=True)
        # 1. all_reduce of a [1,1,1,H] tensor (the per-layer reduce)
        bench("all_reduce [1,1,1,5120]", lambda: AR(x), md)

        # 2. TP MLP matmuls only (no all_reduce)
        gate = fW(rnd(I, H).T, 3); up = fW(rnd(I, H).T, 3); down = fW(rnd(H, I).T, 2)
        bench("MLP matmuls (no AR)", lambda: ttnn.linear(ttnn.mul(ttnn.silu(ttnn.linear(x, gate)), ttnn.linear(x, up)), down), md)
        bench("MLP + all_reduce", lambda: AR(ttnn.linear(ttnn.mul(ttnn.silu(ttnn.linear(x, gate)), ttnn.linear(x, up)), down)), md)

        # 3. DeltaNet: projections + kernel + out_proj
        Wqkv = rnd(key_dim * 2 + val_dim, H)
        Wq = fW(Wqkv[:key_dim].T, 3); Wk = fW(Wqkv[key_dim:2*key_dim].T, 3); Wv = fW(Wqkv[2*key_dim:].T, 3)
        Wz = fW(rnd(val_dim, H).T, 3); Wb = fW(rnd(nv, H).T, 3); Wa = fW(rnd(nv, H).T, 3)
        Wo = fW(rnd(H, val_dim).T, 2)
        def cblk(c): return ttnn.from_torch(c.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(2))
        conv = ttnn.concat([cblk(rnd(key_dim, Kc)), cblk(rnd(key_dim, Kc)), cblk(rnd(val_dim, Kc))], dim=2)
        A = fS(rnd(nv).view(1,1,1,nv).to(torch.bfloat16), 3); dt = fS(rnd(nv).view(1,1,1,nv).to(torch.bfloat16), 3)
        nw = fR(torch.ones(1,1,1,dv)); rs = fS(torch.zeros(1,nv,dk,dv).to(torch.bfloat16), 1)
        cs = fS(torch.zeros(1,1,key_dim*2+val_dim,32).to(torch.bfloat16), 2)
        cdp = (key_dim*2+val_dim)//TP

        def dn_kernel():
            qkv = ttnn.concat([ttnn.linear(x, Wq), ttnn.linear(x, Wk), ttnn.linear(x, Wv)], dim=3)
            z = ttnn.linear(x, Wz); b = ttnn.linear(x, Wb); a = ttnn.linear(x, Wa)
            o, _, _ = ttnn.experimental.deltanet_decode_full(qkv, z, b, a, cs, rs, conv, A, dt, nw,
                num_heads=nvp, num_k_heads=nkp, k_head_dim=dk, v_head_dim=dv, conv_dim=cdp, conv_kernel_size=Kc, head_expand_ratio=ratio)
            return o
        bench("DeltaNet proj+kernel (no out/AR)", dn_kernel, md)
        bench("DeltaNet full + all_reduce", lambda: AR(ttnn.linear(dn_kernel(), Wo)), md)

        # isolate: projections only vs kernel only
        def dn_proj_only():
            qkv = ttnn.concat([ttnn.linear(x, Wq), ttnn.linear(x, Wk), ttnn.linear(x, Wv)], dim=3)
            return ttnn.linear(x, Wz), ttnn.linear(x, Wb), ttnn.linear(x, Wa), qkv
        bench("  - projections only (qkv/z/b/a)", dn_proj_only, md)
        z0, b0, a0, qkv0 = dn_proj_only()
        ttnn.synchronize_device(md)
        bench("  - deltanet_decode_full KERNEL only", lambda: ttnn.experimental.deltanet_decode_full(
            qkv0, z0, b0, a0, cs, rs, conv, A, dt, nw, num_heads=nvp, num_k_heads=nkp,
            k_head_dim=dk, v_head_dim=dv, conv_dim=cdp, conv_kernel_size=Kc, head_expand_ratio=ratio), md)

        # estimate: 64 MLP+AR, 48 deltanet+AR, 16 attn (~MLP-ish) ...
        print("\n[estimate per full step] 64*(MLP+AR) + 48*(DeltaNet+AR) + 16*(attn+AR) dominates", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()

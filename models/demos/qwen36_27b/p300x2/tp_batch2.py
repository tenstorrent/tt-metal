# SPDX-License-Identifier: Apache-2.0
"""Batch via FLATTENING B into the head dim: present B*nv "heads" to deltanet_decode_full
in one call (B=8 → 96 cores ≤ 110), so the kernel parallelizes batch across cores instead
of looping → ~FLAT kernel time. No kernel C++ change. Confirms 8-concurrent feasibility."""
import time
import torch, ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
TP = 4; DT_W, DT_A = ttnn.bfloat8_b, ttnn.bfloat16


def bench(fn, md, iters=50, warmup=8):
    for _ in range(warmup): fn()
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    ttnn.synchronize_device(md)
    return (time.perf_counter() - t0) / iters * 1000


def main():
    cfg = Qwen36ModelConfig()
    H, I = cfg.hidden_size, cfg.intermediate_size
    nv, nk, dk, dv = cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    Kc = cfg.linear_conv_kernel_dim
    key_dim, val_dim = dk*nk, dv*nv
    nvp, nkp = nv//TP, nk//TP; ratio = nv//nk; cdf = key_dim*2+val_dim
    kdp, vdp, cdp = key_dim//TP, val_dim//TP, cdf//TP
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP)); sh=(1,TP)
    SH=lambda d: ttnn.ShardTensor2dMesh(md,dims=(None,d),mesh_shape=sh); REP=ttnn.ReplicateTensorToMesh(md)
    rnd=lambda *s: torch.randn(*s)*0.02
    fW=lambda t,d: ttnn.from_torch(t.unsqueeze(0).unsqueeze(0),dtype=DT_W,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    fR=lambda t: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
    fS=lambda t,d: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    AR=lambda t: ttnn.all_reduce(t,cluster_axis=1,topology=ttnn.Topology.Ring)
    try:
        gu=fW(rnd(2*I,H).T,3); down=fW(rnd(H,I).T,2); Ip=I//TP
        Wqkv=fW(rnd(cdf,H).T,3); Wzba=fW(rnd(val_dim+2*nv,H).T,3); Wo=fW(rnd(H,val_dim).T,2)
        taps={nm: fS(rnd(1,1,1,cdf).to(torch.bfloat16),3) for nm in ("w0","w1","w2","w3","sA","sB","sC")}
        nw=fR(torch.ones(1,1,1,dv))
        print(f"{'B':>3} {'cores':>6} {'MLP_ms':>8} {'DNlayer_ms':>11} {'step_ms':>9} {'TPOT':>7} {'thru':>7}", flush=True)
        for B in (1, 4, 8):
            cores = B*nvp
            if cores > 110:
                print(f"{B:3d}  >110 cores — needs multi-pass", flush=True); continue
            xb = fR(torch.randn(1,1,B,H)*0.1)
            # flattened-head deltanet state/weights (dummy, B*nv heads)
            rsF=fS(torch.zeros(1,B*nv,dk,dv).to(torch.bfloat16),1)
            csF=fS(torch.zeros(1,1,B*cdf,32).to(torch.bfloat16),2)
            convF=fS(rnd(1,1,B*cdf,32).to(torch.bfloat16),2) if False else ttnn.from_torch(rnd(1,1,B*cdf,32).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(2))
            AF=fS(rnd(1,1,1,B*nv).to(torch.bfloat16),3); dtF=fS(rnd(1,1,1,B*nv).to(torch.bfloat16),3)
            def mlp():
                g=ttnn.linear(xb,gu); gg=ttnn.silu(ttnn.slice(g,[0,0,0,0],[1,1,B,Ip])); u=ttnn.slice(g,[0,0,0,Ip],[1,1,B,2*Ip])
                return AR(ttnn.linear(ttnn.mul(gg,u),down))
            def dn_layer():
                qkv=ttnn.linear(xb,Wqkv)  # [1,1,B,cdp]
                dot=ttnn.add(ttnn.add(ttnn.mul(taps["w0"],taps["sA"]),ttnn.mul(taps["w1"],taps["sB"])),ttnn.add(ttnn.mul(taps["w2"],taps["sC"]),ttnn.mul(taps["w3"],qkv)))
                qkv=ttnn.silu(dot)  # [1,1,B,cdp]
                zba=ttnn.linear(xb,Wzba)  # [1,1,B, vdp+2*nv/TP]
                # split q/k/v per batch row, flatten B into heads -> [all_q | all_k | all_v]
                q=ttnn.reshape(ttnn.slice(qkv,[0,0,0,0],[1,1,B,kdp]),[1,1,1,B*kdp])
                k=ttnn.reshape(ttnn.slice(qkv,[0,0,0,kdp],[1,1,B,2*kdp]),[1,1,1,B*kdp])
                v=ttnn.reshape(ttnn.slice(qkv,[0,0,0,2*kdp],[1,1,B,cdp]),[1,1,1,B*vdp])
                qkvF=ttnn.concat([q,k,v],dim=3)  # [1,1,1, B*cdp] = all_q|all_k|all_v
                nvpc=nv//TP
                z=ttnn.reshape(ttnn.slice(zba,[0,0,0,0],[1,1,B,vdp]),[1,1,1,B*vdp])
                b=ttnn.reshape(ttnn.slice(zba,[0,0,0,vdp],[1,1,B,vdp+nvpc]),[1,1,1,B*nvpc])
                a=ttnn.reshape(ttnn.slice(zba,[0,0,0,vdp+nvpc],[1,1,B,vdp+2*nvpc]),[1,1,1,B*nvpc])
                o,_,_=ttnn.experimental.deltanet_decode_full(qkvF,z,b,a,csF,rsF,convF,AF,dtF,nw,
                    num_heads=B*nvp, num_k_heads=B*nkp, k_head_dim=dk, v_head_dim=dv,
                    conv_dim=B*cdp, conv_kernel_size=Kc, head_expand_ratio=ratio)
                o=ttnn.reshape(o,[1,1,B,vdp])  # [1,1,1,B*vdp] -> per-batch rows for o_proj
                return AR(ttnn.linear(o,Wo))
            mlp_ms=bench(mlp,md); dn_ms=bench(dn_layer,md)
            step=48*dn_ms+64*mlp_ms+16*0.40+2.0
            print(f"{B:3d} {cores:6d} {mlp_ms:8.3f} {dn_ms:11.3f} {step:9.1f} {1000/step:7.2f} {B*1000/step:7.1f}", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()

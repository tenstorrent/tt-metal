# SPDX-License-Identifier: Apache-2.0
"""Per-component device-bound profile of the CURRENT optimized TP step (fused projections,
host conv, Linear all_reduce) to find where the remaining ~52ms goes. 1x8 mesh, 1 layer each."""
import time, statistics
import torch, ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
TP = 8; DT_W, DT_A = ttnn.bfloat8_b, ttnn.bfloat16


def bench(name, fn, md, iters=100, warmup=10):
    for _ in range(warmup): fn()
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    ttnn.synchronize_device(md)
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:30s} {ms:7.3f} ms", flush=True)
    return ms


def main():
    cfg = Qwen36ModelConfig()
    H, I = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd, rdim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    nv, nk, dk, dv = cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    Kc = cfg.linear_conv_kernel_dim
    key_dim, val_dim = dk*nk, dv*nv
    nvp, nkp = nv//TP, nk//TP
    scale = hd**-0.5; ratio = nv//nk; ctx = 64
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP)); sh=(1,TP)
    SH=lambda d: ttnn.ShardTensor2dMesh(md,dims=(None,d),mesh_shape=sh); REP=ttnn.ReplicateTensorToMesh(md)
    rnd=lambda *s: torch.randn(*s)*0.02
    fW=lambda t,d: ttnn.from_torch(t.unsqueeze(0).unsqueeze(0),dtype=DT_W,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    fR=lambda t: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
    fS=lambda t,d: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    try:
        x=fR(torch.randn(1,1,1,H)*0.1)
        AR=lambda t,topo: ttnn.all_reduce(t,cluster_axis=1,topology=topo)
        print("[per-component device-bound, 1x8 mesh]", flush=True)
        bench("all_reduce LINEAR [H]", lambda: AR(x, ttnn.Topology.Linear), md)
        bench("all_reduce RING [H]", lambda: AR(x, ttnn.Topology.Ring), md)

        # MLP fused
        gu=fW(rnd(2*I,H).T,3); down=fW(rnd(H,I).T,2); Ip=I//TP
        def mlp():
            g=ttnn.linear(x,gu); gg=ttnn.silu(ttnn.slice(g,[0,0,0,0],[1,1,1,Ip])); u=ttnn.slice(g,[0,0,0,Ip],[1,1,1,2*Ip])
            return AR(ttnn.linear(ttnn.mul(gg,u),down), ttnn.Topology.Linear)
        bench("MLP (fused gu + down + LinearAR)", mlp, md)

        # DeltaNet fused
        Wqkv=fW(rnd(key_dim*2+val_dim,H).T,3); Wzba=fW(rnd(val_dim+2*nv,H).T,3); Wo=fW(rnd(H,val_dim).T,2)
        cdf=key_dim*2+val_dim
        taps={nm: fS(rnd(1,1,1,cdf).to(torch.bfloat16),3) for nm in ("w0","w1","w2","w3","sA","sB","sC")}
        def cblk(c): return ttnn.from_torch(c.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(2))
        conv=ttnn.concat([cblk(rnd(key_dim,Kc)),cblk(rnd(key_dim,Kc)),cblk(rnd(val_dim,Kc))],dim=2)
        A=fS(rnd(nv).view(1,1,1,nv).to(torch.bfloat16),3); dt=fS(rnd(nv).view(1,1,1,nv).to(torch.bfloat16),3)
        nw=fR(torch.ones(1,1,1,dv)); rs=fS(torch.zeros(1,nv,dk,dv).to(torch.bfloat16),1)
        cs=fS(torch.zeros(1,1,cdf,32).to(torch.bfloat16),2); vd_p,nv_p=val_dim//TP,nv//TP
        def dn_conv():
            qkv=ttnn.linear(x,Wqkv)
            dot=ttnn.add(ttnn.add(ttnn.mul(taps["w0"],taps["sA"]),ttnn.mul(taps["w1"],taps["sB"])),ttnn.add(ttnn.mul(taps["w2"],taps["sC"]),ttnn.mul(taps["w3"],qkv)))
            return ttnn.silu(dot)
        def dn_full():
            qkv=dn_conv(); zba=ttnn.linear(x,Wzba)
            z=ttnn.slice(zba,[0,0,0,0],[1,1,1,vd_p]); b=ttnn.slice(zba,[0,0,0,vd_p],[1,1,1,vd_p+nv_p]); a=ttnn.slice(zba,[0,0,0,vd_p+nv_p],[1,1,1,vd_p+2*nv_p])
            o,_,_=ttnn.experimental.deltanet_decode_full(qkv,z,b,a,cs,rs,conv,A,dt,nw,num_heads=nvp,num_k_heads=nkp,k_head_dim=dk,v_head_dim=dv,conv_dim=cdf//TP,conv_kernel_size=Kc,head_expand_ratio=ratio)
            return AR(ttnn.linear(o,Wo), ttnn.Topology.Linear)
        bench("  DeltaNet Wqkv matmul", lambda: ttnn.linear(x,Wqkv), md)
        bench("  DeltaNet conv (8 ops)", dn_conv, md)
        bench("  DeltaNet kernel only", lambda: ttnn.experimental.deltanet_decode_full(ttnn.linear(x,Wqkv),ttnn.slice(ttnn.linear(x,Wzba),[0,0,0,0],[1,1,1,vd_p]),ttnn.slice(ttnn.linear(x,Wzba),[0,0,0,vd_p],[1,1,1,vd_p+nv_p]),ttnn.slice(ttnn.linear(x,Wzba),[0,0,0,vd_p+nv_p],[1,1,1,vd_p+2*nv_p]),cs,rs,conv,A,dt,nw,num_heads=nvp,num_k_heads=nkp,k_head_dim=dk,v_head_dim=dv,conv_dim=cdf//TP,conv_kernel_size=Kc,head_expand_ratio=ratio), md)
        bench("DeltaNet FULL (fused+conv+LinearAR)", dn_full, md)

        print("\n[per-step estimate] 48*DeltaNet + 64*MLP + 16*attn + lmhead", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Batch scaling of the TP step components (B=1,4,8) + memory budget for max batch.
Matmuls/attn batch natively; DeltaNet decode kernel is batch=1 so it's looped B times
(the realistic 'B independent states' path). Estimates step time / TPOT / throughput per B."""
import time, statistics
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
        def cblk(c): return ttnn.from_torch(c.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(2))
        conv=ttnn.concat([cblk(rnd(key_dim,Kc)),cblk(rnd(key_dim,Kc)),cblk(rnd(val_dim,Kc))],dim=2)
        A=fS(rnd(nv).view(1,1,1,nv).to(torch.bfloat16),3); dt=fS(rnd(nv).view(1,1,1,nv).to(torch.bfloat16),3)
        nw=fR(torch.ones(1,1,1,dv)); rs=fS(torch.zeros(1,nv,dk,dv).to(torch.bfloat16),1); cs=fS(torch.zeros(1,1,cdf,32).to(torch.bfloat16),2)
        vd_p,nv_p=val_dim//TP,nv//TP

        print(f"{'B':>3} {'MLP_ms':>8} {'DNlayer_ms':>11} {'step_est_ms':>12} {'TPOT_tok/s':>11} {'thrupt_tok/s':>13}", flush=True)
        for B in (1, 4, 8):
            xb = fR(torch.randn(1, 1, B, H) * 0.1)   # B on the M (row) dim → one amortized matmul
            def mlp():
                g=ttnn.linear(xb,gu); gg=ttnn.silu(ttnn.slice(g,[0,0,0,0],[1,1,B,Ip])); u=ttnn.slice(g,[0,0,0,Ip],[1,1,B,2*Ip])
                return AR(ttnn.linear(ttnn.mul(gg,u),down))
            def dn_layer():
                qkv=ttnn.linear(xb,Wqkv)
                dot=ttnn.add(ttnn.add(ttnn.mul(taps["w0"],taps["sA"]),ttnn.mul(taps["w1"],taps["sB"])),ttnn.add(ttnn.mul(taps["w2"],taps["sC"]),ttnn.mul(taps["w3"],qkv)))
                qkv=ttnn.silu(dot); zba=ttnn.linear(xb,Wzba)
                z=ttnn.slice(zba,[0,0,0,0],[1,1,B,vd_p]); b=ttnn.slice(zba,[0,0,0,vd_p],[1,1,B,vd_p+nv_p]); a=ttnn.slice(zba,[0,0,0,vd_p+nv_p],[1,1,B,vd_p+2*nv_p])
                # DeltaNet kernel is batch=1: loop B (slice per-b row), representative of B independent states
                outs=[]
                for bi in range(B):
                    qkv_b=ttnn.slice(qkv,[0,0,bi,0],[1,1,bi+1,cdf//TP]); z_b=ttnn.slice(z,[0,0,bi,0],[1,1,bi+1,vd_p])
                    b_b=ttnn.slice(b,[0,0,bi,0],[1,1,bi+1,nv_p]); a_b=ttnn.slice(a,[0,0,bi,0],[1,1,bi+1,nv_p])
                    o,_,_=ttnn.experimental.deltanet_decode_full(qkv_b,z_b,b_b,a_b,cs,rs,conv,A,dt,nw,num_heads=nvp,num_k_heads=nkp,k_head_dim=dk,v_head_dim=dv,conv_dim=cdf//TP,conv_kernel_size=Kc,head_expand_ratio=ratio)
                    outs.append(o)
                o=ttnn.concat(outs,dim=2) if B>1 else outs[0]
                return AR(ttnn.linear(o,Wo))
            mlp_ms=bench(mlp,md); dn_ms=bench(dn_layer,md)
            step=48*dn_ms + 64*mlp_ms + 16*0.40 + 2.0  # attn~0.40, lmhead+norm~2
            tpot=1000/step; thr=B*1000/step
            print(f"{B:3d} {mlp_ms:8.3f} {dn_ms:11.3f} {step:12.1f} {tpot:11.2f} {thr:13.1f}", flush=True)

        # memory budget
        wB = (27.0)/TP  # bf8 weights ~27GB / 4 chips
        print(f"\n[memory/chip] weights ~{wB:.1f}GB of 32GB; free ~{32-wB:.0f}GB", flush=True)
        for seq in (2048, 8192):
            kv_per_B = 1*seq*dv*2*2*16/1e9  # nkvp*seq*hd*2(kv)*2B*16layers
            st_per_B = nvp*dk*dv*2*48/1e9
            print(f"  seq={seq}: KV+state per batch ~{kv_per_B+st_per_B:.3f}GB → max B (mem) ~{int((32-wB)/(kv_per_B+st_per_B))}", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()

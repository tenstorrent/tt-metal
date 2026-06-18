# SPDX-License-Identifier: Apache-2.0
"""B=8 sub-component device-bound profile of the full TP step (batched attn + flattened
DeltaNet + B-on-M MLP) to find what to cut for the last 56->50ms (20 tok/s/user)."""
import time
import torch, ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
TP = 8; B = 8; DT_W, DT_A = ttnn.bfloat8_b, ttnn.bfloat16


def bench(name, fn, md, mult, iters=60, warmup=10):
    for _ in range(warmup): fn()
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    ttnn.synchronize_device(md)
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:28s} {ms:7.3f} ms x{mult:>2} = {ms*mult:6.1f} ms/step", flush=True)
    return ms * mult


def main():
    cfg = Qwen36ModelConfig()
    H, I = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd, rdim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    nv, nk, dk, dv = cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    Kc = cfg.linear_conv_kernel_dim
    key_dim, val_dim = dk*nk, dv*nv; cdf = key_dim*2+val_dim
    nhp, nvp, nkp = nh//TP, nv//TP, nk//TP; ratio = nv//nk; scale = hd**-0.5
    # full_attention GQA: nkv (4) < TP (8) on T3K → replicate each KV head across TP//nkv chips.
    nkvp = max(1, nkv//TP)   # KV heads per chip (1 when replicated)
    kv_slots = nkvp * TP     # total KV-head slots across the mesh (includes replication)
    kdp, vdp, cdp = key_dim//TP, val_dim//TP, cdf//TP; qsz, ksz = nhp*hd*2, nkvp*hd; ctx = 64
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP)); sh=(1,TP)
    SH=lambda d: ttnn.ShardTensor2dMesh(md,dims=(None,d),mesh_shape=sh); REP=ttnn.ReplicateTensorToMesh(md)
    rnd=lambda *s: torch.randn(*s)*0.02
    fW=lambda t,d: ttnn.from_torch(t.unsqueeze(0).unsqueeze(0),dtype=DT_W,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    fR=lambda t: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
    fS=lambda t,d: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    AR=lambda t: ttnn.all_reduce(t,cluster_axis=1,topology=ttnn.Topology.Linear)
    try:
        x=fR(torch.randn(1,1,B,H)*0.1)
        print(f"[B={B} sub-component device-bound; mult = layer count]", flush=True)
        bench("all_reduce [1,1,B,H]", lambda: AR(x), md, 128)
        # MLP
        gu=fW(rnd(2*I,H).T,3); down=fW(rnd(H,I).T,2); Ip=I//TP
        bench("MLP (gu+down+AR)", lambda: AR(ttnn.linear(ttnn.mul(ttnn.silu(ttnn.slice(ttnn.linear(x,gu),[0,0,0,0],[1,1,B,Ip])),ttnn.slice(ttnn.linear(x,gu),[0,0,0,Ip],[1,1,B,2*Ip])),down)), md, 64)
        # attention sub-parts
        Wqkv_a=fW(rnd(nh*hd*2+2*kv_slots*hd,H).T,3); Wo_a=fW(rnd(H,nh*hd).T,2)  # K,V replicated to kv_slots (=TP) heads
        qn=fR((rnd(hd)+1).view(1,1,1,hd)); kn=fR((rnd(hd)+1).view(1,1,1,hd)); cos=fR(torch.randn(1,1,1,rdim)*.5); sin=fR(torch.randn(1,1,1,rdim)*.5)
        kc=ttnn.from_torch(torch.randn(B,nkvp,ctx,hd).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
        vc=ttnn.from_torch(torch.randn(B,nkvp,ctx,hd).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
        cur=ttnn.from_torch(torch.full((B,),ctx-1,dtype=torch.int32),device=md)
        sdpa=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=md.compute_with_storage_grid_size(),q_chunk_size=32,k_chunk_size=32)
        scfg=ttnn.create_sharded_memory_config(shape=(32,hd),core_grid=ttnn.CoreGrid(y=1,x=B*nkvp),strategy=ttnn.ShardStrategy.HEIGHT,orientation=ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
        def rope_tt(t,n):
            tr=ttnn.slice(t,[0,0,0,0],[1,B,n,rdim]); tp=ttnn.slice(t,[0,0,0,rdim],[1,B,n,hd])
            x1=ttnn.slice(tr,[0,0,0,0],[1,B,n,rdim//2]); x2=ttnn.slice(tr,[0,0,0,rdim//2],[1,B,n,rdim])
            return ttnn.concat([ttnn.add(ttnn.mul(tr,cos),ttnn.mul(ttnn.concat([ttnn.neg(x2),x1],dim=-1),sin)),tp],dim=-1)
        def attn_proj():
            qkv=ttnn.linear(x,Wqkv_a); return qkv
        def attn_sdpa():
            qkv=ttnn.linear(x,Wqkv_a)
            qp=ttnn.slice(qkv,[0,0,0,0],[1,1,B,qsz]); kp=ttnn.slice(qkv,[0,0,0,qsz],[1,1,B,qsz+ksz]); vp=ttnn.slice(qkv,[0,0,0,qsz+ksz],[1,1,B,qsz+2*ksz])
            q2=ttnn.reshape(qp,[1,B,nhp,hd*2]); qry=ttnn.slice(q2,[0,0,0,0],[1,B,nhp,hd])
            ky=ttnn.reshape(kp,[1,B,nkvp,hd]); vl=ttnn.reshape(vp,[1,B,nkvp,hd])
            qry=ttnn.rms_norm(qry,epsilon=1e-6,weight=qn); ky=ttnn.rms_norm(ky,epsilon=1e-6,weight=kn)
            qry=rope_tt(qry,nhp); ky=rope_tt(ky,nkvp)
            ttnn.experimental.paged_update_cache(kc,ttnn.to_memory_config(ky,scfg),update_idxs_tensor=cur)
            ttnn.experimental.paged_update_cache(vc,ttnn.to_memory_config(vl,scfg),update_idxs_tensor=cur)
            return ttnn.transformer.scaled_dot_product_attention_decode(qry,kc,vc,cur_pos_tensor=cur,scale=scale,program_config=sdpa)
        bench("attn proj only", attn_proj, md, 16)
        bench("attn proj+norm+rope+kv+sdpa", attn_sdpa, md, 16)
        # DeltaNet sub-parts
        Wqkv=fW(rnd(cdf,H).T,3); Wzba=fW(rnd(val_dim+2*nv,H).T,3); Wo=fW(rnd(H,val_dim).T,2)
        taps={nm: fS(rnd(1,1,1,cdf).to(torch.bfloat16),3) for nm in ("w0","w1","w2","w3","sA","sB","sC")}
        conv=fS(rnd(1,1,B*cdf,32).to(torch.bfloat16),2); A=fS(rnd(1,1,1,B*nv).to(torch.bfloat16),3); dt=fS(rnd(1,1,1,B*nv).to(torch.bfloat16),3)
        nw=fR(torch.ones(1,1,1,dv)); rs=fS(torch.zeros(1,B*nv,dk,dv).to(torch.bfloat16),1); cs=fS(torch.zeros(1,1,B*cdf,32).to(torch.bfloat16),2)
        nv_p=nv//TP
        def dn_conv():
            qkv=ttnn.linear(x,Wqkv)
            return ttnn.silu(ttnn.add(ttnn.add(ttnn.mul(taps["w0"],taps["sA"]),ttnn.mul(taps["w1"],taps["sB"])),ttnn.add(ttnn.mul(taps["w2"],taps["sC"]),ttnn.mul(taps["w3"],qkv))))
        def dn_flatten():
            qkv=dn_conv()
            q=ttnn.reshape(ttnn.slice(qkv,[0,0,0,0],[1,1,B,kdp]),[1,1,1,B*kdp]); k=ttnn.reshape(ttnn.slice(qkv,[0,0,0,kdp],[1,1,B,2*kdp]),[1,1,1,B*kdp]); v=ttnn.reshape(ttnn.slice(qkv,[0,0,0,2*kdp],[1,1,B,cdp]),[1,1,1,B*vdp])
            return ttnn.concat([q,k,v],dim=3)
        def dn_kernel():
            qkvF=dn_flatten(); zba=ttnn.linear(x,Wzba)
            z=ttnn.reshape(ttnn.slice(zba,[0,0,0,0],[1,1,B,vdp]),[1,1,1,B*vdp]); b=ttnn.reshape(ttnn.slice(zba,[0,0,0,vdp],[1,1,B,vdp+nv_p]),[1,1,1,B*nv_p]); a=ttnn.reshape(ttnn.slice(zba,[0,0,0,vdp+nv_p],[1,1,B,vdp+2*nv_p]),[1,1,1,B*nv_p])
            o,_,_=ttnn.experimental.deltanet_decode_full(qkvF,z,b,a,cs,rs,conv,A,dt,nw,num_heads=B*nvp,num_k_heads=B*nkp,k_head_dim=dk,v_head_dim=dv,conv_dim=B*cdp,conv_kernel_size=Kc,head_expand_ratio=ratio)
            return o
        bench("DN Wqkv matmul", lambda: ttnn.linear(x,Wqkv), md, 48)
        bench("DN conv (4mul+3add+silu)", dn_conv, md, 48)
        bench("DN +flatten(slice/reshape/cat)", dn_flatten, md, 48)
        bench("DN +kernel(flattened 96 heads)", dn_kernel, md, 48)
        bench("DN FULL (+zba+oproj+AR)", lambda: AR(ttnn.linear(ttnn.reshape(dn_kernel(),[1,1,B,vdp]),Wo)), md, 48)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Full 64-layer Qwen3.6-27B decode, 8-way TP + BATCH, end-to-end on a 1x8 T3K mesh.
Batch is on the matmul M-dim [1,1,B,H]; attention via batched sdpa_decode; DeltaNet via
flattened-head kernel (B*nv heads, one call). Dummy weights (~7GB/chip). Measures TPOT +
throughput for 8 concurrent users. Per-component correctness validated separately.

  python3 tp_model_batch.py --layers 64 --batch 8 --steps 16
"""
import argparse, time, statistics
import torch, ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
TP = 8; DT_W, DT_A = ttnn.bfloat8_b, ttnn.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=64)
    args = ap.parse_args(); B = args.batch
    cfg = Qwen36ModelConfig(); NL = args.layers
    H, I = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd, rdim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    nv, nk, dk, dv = cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    Kc = cfg.linear_conv_kernel_dim
    key_dim, val_dim = dk*nk, dv*nv; cdf = key_dim*2+val_dim
    nhp, nvp, nkp = nh//TP, nv//TP, nk//TP; ratio = nv//nk; scale = hd**-0.5
    # full_attention GQA: nkv (4) < TP (8) on T3K → replicate each KV head across TP//nkv chips.
    nkvp = max(1, nkv//TP)   # KV heads per chip (1 when replicated)
    kv_slots = nkvp * TP     # total KV-head slots across the mesh (includes replication)
    kdp, vdp, cdp = key_dim//TP, val_dim//TP, cdf//TP
    qsz, ksz = nhp*hd*2, nkvp*hd
    ltypes = cfg.layer_types[:NL]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP)); sh=(1,TP)
    SH=lambda d: ttnn.ShardTensor2dMesh(md,dims=(None,d),mesh_shape=sh); REP=ttnn.ReplicateTensorToMesh(md)
    rnd=lambda *s: torch.zeros(*s)  # dummy weights: zeros (decode TPOT is value-independent; far faster build)
    fW=lambda t,d: ttnn.from_torch(t.unsqueeze(0).unsqueeze(0),dtype=DT_W,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    fR=lambda t: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
    fS=lambda t,d: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    AR=lambda t: ttnn.all_reduce(t,cluster_axis=1,topology=ttnn.Topology.Linear)
    if B*nvp > 110:
        print(f"B={B}: DeltaNet needs {B*nvp} cores > 110 — use B<=9"); return

    print(f"[build] {NL} layers, TP={TP}, B={B} ...", flush=True); t0=time.time()
    layers=[]
    for li in range(NL):
        lt=ltypes[li]; L={"type":lt}
        L["in_norm"]=fR(torch.ones(1,1,1,H)); L["post_norm"]=fR(torch.ones(1,1,1,H))
        L["gu"]=fW(rnd(2*I,H).T,3); L["down"]=fW(rnd(H,I).T,2)
        if lt=="full_attention":
            L["Wqkv"]=fW(rnd(nh*hd*2+2*kv_slots*hd,H).T,3); L["Wo"]=fW(rnd(H,nh*hd).T,2)  # K,V replicated to kv_slots (=TP) heads
            L["qn"]=fR((rnd(hd)+1.0).view(1,1,1,hd)); L["kn"]=fR((rnd(hd)+1.0).view(1,1,1,hd))
            L["kc"]=ttnn.from_torch(torch.zeros(B,nkvp,args.ctx,hd).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
            L["vc"]=ttnn.from_torch(torch.zeros(B,nkvp,args.ctx,hd).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
        else:
            L["Wqkv"]=fW(rnd(cdf,H).T,3); L["Wzba"]=fW(rnd(val_dim+2*nv,H).T,3); L["Wo"]=fW(rnd(H,val_dim).T,2)
            for nm in ("w0","w1","w2","w3","sA","sB","sC"):
                L[nm]=fS(rnd(1,1,1,cdf).to(torch.bfloat16),3)
            L["conv"]=fS(rnd(1,1,B*cdf,32).to(torch.bfloat16),2)
            L["A"]=fS(rnd(1,1,1,B*nv).to(torch.bfloat16),3); L["dt"]=fS(rnd(1,1,1,B*nv).to(torch.bfloat16),3)
            L["nw"]=fR(torch.ones(1,1,1,dv)); L["rs"]=fS(torch.zeros(1,B*nv,dk,dv).to(torch.bfloat16),1)
            L["cs"]=fS(torch.zeros(1,1,B*cdf,32).to(torch.bfloat16),2)
        layers.append(L)
    final_norm=fR(torch.ones(1,1,1,H)); lm_head=fW(rnd(cfg.padded_vocab_size,H).T,3)
    cur=ttnn.from_torch(torch.full((B,),args.ctx-1,dtype=torch.int32),device=md)
    cos=fR(torch.randn(1,1,1,rdim)*0.5); sin=fR(torch.randn(1,1,1,rdim)*0.5)
    sdpa=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=md.compute_with_storage_grid_size(),q_chunk_size=32,k_chunk_size=32)
    def shard_for(nrows): return ttnn.create_sharded_memory_config(shape=(32,hd),core_grid=ttnn.CoreGrid(y=1,x=max(1,nrows)),strategy=ttnn.ShardStrategy.HEIGHT,orientation=ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
    print(f"[build] done in {time.time()-t0:.1f}s", flush=True)

    def rope_tt(t,n):
        tr=ttnn.slice(t,[0,0,0,0],[1,B,n,rdim]); tp=ttnn.slice(t,[0,0,0,rdim],[1,B,n,hd])
        x1=ttnn.slice(tr,[0,0,0,0],[1,B,n,rdim//2]); x2=ttnn.slice(tr,[0,0,0,rdim//2],[1,B,n,rdim])
        rh=ttnn.concat([ttnn.neg(x2),x1],dim=-1)
        return ttnn.concat([ttnn.add(ttnn.mul(tr,cos),ttnn.mul(rh,sin)),tp],dim=-1)

    def attn(h,L):
        qkv=ttnn.linear(h,L["Wqkv"])  # [1,1,B,*] B on M
        qp=ttnn.slice(qkv,[0,0,0,0],[1,1,B,qsz]); kp=ttnn.slice(qkv,[0,0,0,qsz],[1,1,B,qsz+ksz]); vp=ttnn.slice(qkv,[0,0,0,qsz+ksz],[1,1,B,qsz+2*ksz])
        q2=ttnn.reshape(qp,[1,B,nhp,hd*2]); qry=ttnn.slice(q2,[0,0,0,0],[1,B,nhp,hd]); gt=ttnn.slice(q2,[0,0,0,hd],[1,B,nhp,hd*2])
        ky=ttnn.reshape(kp,[1,B,nkvp,hd]); vl=ttnn.reshape(vp,[1,B,nkvp,hd])
        qry=ttnn.rms_norm(qry,epsilon=1e-6,weight=L["qn"]); ky=ttnn.rms_norm(ky,epsilon=1e-6,weight=L["kn"])
        qry=rope_tt(qry,nhp); ky=rope_tt(ky,nkvp)
        ttnn.experimental.paged_update_cache(L["kc"],ttnn.to_memory_config(ky,shard_for(B*nkvp)),update_idxs_tensor=cur)
        ttnn.experimental.paged_update_cache(L["vc"],ttnn.to_memory_config(vl,shard_for(B*nkvp)),update_idxs_tensor=cur)
        att=ttnn.transformer.scaled_dot_product_attention_decode(qry,L["kc"],L["vc"],cur_pos_tensor=cur,scale=scale,program_config=sdpa)
        att=ttnn.reshape(att,[1,1,B,nhp*hd]); gt=ttnn.reshape(gt,[1,1,B,nhp*hd])
        att=ttnn.mul(att,ttnn.sigmoid(gt))
        return AR(ttnn.linear(att,L["Wo"]))

    nv_p=nv//TP
    def deltanet(h,L):
        qkv=ttnn.linear(h,L["Wqkv"])  # [1,1,B,cdp]
        dot=ttnn.add(ttnn.add(ttnn.mul(L["w0"],L["sA"]),ttnn.mul(L["w1"],L["sB"])),ttnn.add(ttnn.mul(L["w2"],L["sC"]),ttnn.mul(L["w3"],qkv)))
        qkv=ttnn.silu(dot)
        zba=ttnn.linear(h,L["Wzba"])
        # flatten B into head dim: [all_q | all_k | all_v]
        q=ttnn.reshape(ttnn.slice(qkv,[0,0,0,0],[1,1,B,kdp]),[1,1,1,B*kdp])
        k=ttnn.reshape(ttnn.slice(qkv,[0,0,0,kdp],[1,1,B,2*kdp]),[1,1,1,B*kdp])
        v=ttnn.reshape(ttnn.slice(qkv,[0,0,0,2*kdp],[1,1,B,cdp]),[1,1,1,B*vdp])
        qkvF=ttnn.concat([q,k,v],dim=3)
        z=ttnn.reshape(ttnn.slice(zba,[0,0,0,0],[1,1,B,vdp]),[1,1,1,B*vdp])
        b=ttnn.reshape(ttnn.slice(zba,[0,0,0,vdp],[1,1,B,vdp+nv_p]),[1,1,1,B*nv_p])
        a=ttnn.reshape(ttnn.slice(zba,[0,0,0,vdp+nv_p],[1,1,B,vdp+2*nv_p]),[1,1,1,B*nv_p])
        # decay/beta host-side (keeps libm off NCRISC): beta=sigmoid(b); decay=exp(-exp(A)*softplus(a+dt))
        beta=ttnn.sigmoid(b)
        decay=ttnn.exp(ttnn.neg(ttnn.mul(ttnn.exp(L["A"]),ttnn.softplus(ttnn.add(a,L["dt"])))))
        o,_,_=ttnn.experimental.deltanet_decode_full(qkvF,z,beta,decay,L["cs"],L["rs"],L["conv"],L["A"],L["dt"],L["nw"],
            num_heads=B*nvp, num_k_heads=B*nkp, k_head_dim=dk, v_head_dim=dv, conv_dim=B*cdp, conv_kernel_size=Kc, head_expand_ratio=ratio)
        o=ttnn.reshape(o,[1,1,B,vdp])
        return AR(ttnn.linear(o,L["Wo"]))

    Ip=I//TP
    def mlp(h,L):
        g=ttnn.linear(h,L["gu"]); gg=ttnn.silu(ttnn.slice(g,[0,0,0,0],[1,1,B,Ip])); u=ttnn.slice(g,[0,0,0,Ip],[1,1,B,2*Ip])
        return AR(ttnn.linear(ttnn.mul(gg,u),L["down"]))

    def step(x):
        for L in layers:
            r=x; h=ttnn.rms_norm(x,epsilon=1e-6,weight=L["in_norm"])
            h=attn(h,L) if L["type"]=="full_attention" else deltanet(h,L)
            x=ttnn.add(r,h)
            r=x; h=ttnn.rms_norm(x,epsilon=1e-6,weight=L["post_norm"])
            x=ttnn.add(r,mlp(h,L))
        x=ttnn.rms_norm(x,epsilon=1e-6,weight=final_norm)
        return ttnn.all_gather(ttnn.linear(x,lm_head),dim=3,cluster_axis=1,topology=ttnn.Topology.Linear)

    try:
        x0=fR(torch.randn(1,1,B,H)*0.1)
        for _ in range(3): step(x0)
        ttnn.synchronize_device(md)
        te=[]
        for _ in range(args.steps):
            t1=time.perf_counter(); step(x0); ttnn.synchronize_device(md); te.append((time.perf_counter()-t1)*1000)
        eag=statistics.median(te)
        # trace
        tid=ttnn.begin_trace_capture(md,cq_id=0); _=step(x0); ttnn.end_trace_capture(md,tid,cq_id=0); ttnn.synchronize_device(md)
        tt=[]
        for _ in range(args.steps):
            t1=time.perf_counter(); ttnn.execute_trace(md,tid,cq_id=0,blocking=False); ttnn.synchronize_device(md); tt.append((time.perf_counter()-t1)*1000)
        tr=statistics.median(tt); ttnn.release_trace(md,tid)
        print(f"\n[B={B}, {NL} layers, 8 chips]", flush=True)
        print(f"  EAGER: {eag:.1f} ms/step  TPOT {1000/eag:.2f}/user  throughput {B*1000/eag:.1f} tok/s", flush=True)
        print(f"  TRACE: {tr:.1f} ms/step  TPOT {1000/tr:.2f}/user  throughput {B*1000/tr:.1f} tok/s", flush=True)
        print(f"  (batch=1 trace was 52ms/19.25 tok/s; target 8 users @ ~20 tok/s)", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Batch scaling of the TP attention layer (B=1,4,8) — sdpa_decode natively supports batch
(unlike the DeltaNet kernel). Quantifies whether attention stays ~flat or scales with B,
to correct the batch=8 throughput estimate (which used a fixed attn cost)."""
import time
import torch, ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
TP = 8; DT_W, DT_A = ttnn.bfloat8_b, ttnn.bfloat16


def bench(fn, md, iters=50, warmup=8):
    for _ in range(warmup): fn()
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    ttnn.synchronize_device(md)
    return (time.perf_counter() - t0) / iters * 1000


def main():
    cfg = Qwen36ModelConfig()
    H = cfg.hidden_size
    nh, nkv, hd, rdim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    nhp = nh//TP; scale = hd**-0.5; ctx = 64
    # full_attention GQA: nkv (4) < TP (8) on T3K → replicate each KV head across TP//nkv chips.
    nkvp = max(1, nkv//TP)   # KV heads per chip (1 when replicated)
    kv_slots = nkvp * TP     # total KV-head slots across the mesh (includes replication)
    qsz, ksz = nhp*hd*2, nkvp*hd
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP)); sh=(1,TP)
    SH=lambda d: ttnn.ShardTensor2dMesh(md,dims=(None,d),mesh_shape=sh); REP=ttnn.ReplicateTensorToMesh(md)
    rnd=lambda *s: torch.randn(*s)*0.02
    fW=lambda t,d: ttnn.from_torch(t.unsqueeze(0).unsqueeze(0),dtype=DT_W,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=SH(d))
    fR=lambda t: ttnn.from_torch(t,dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
    AR=lambda t: ttnn.all_reduce(t,cluster_axis=1,topology=ttnn.Topology.Linear)
    try:
        Wqkv=fW(rnd(nh*hd*2+2*kv_slots*hd,H).T,3); Wo=fW(rnd(H,nh*hd).T,2)  # K,V replicated to kv_slots (=TP) heads
        qn=fR((rnd(hd)+1.0).view(1,1,1,hd)); kn=fR((rnd(hd)+1.0).view(1,1,1,hd))
        cos=fR(torch.randn(1,1,1,rdim)*0.5); sin=fR(torch.randn(1,1,1,rdim)*0.5)
        sdpa=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=md.compute_with_storage_grid_size(),
            q_chunk_size=32,k_chunk_size=32)
        def shard_for(B):
            return ttnn.create_sharded_memory_config(shape=(32,hd),core_grid=ttnn.CoreGrid(y=1,x=max(1,B*nkvp)),
                strategy=ttnn.ShardStrategy.HEIGHT,orientation=ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
        print(f"{'B':>3} {'attn_ms':>9}  (16 layers => {'tot_ms':>7})", flush=True)
        for B in (1,4,8):
            xb=fR(torch.randn(1,1,B,H)*0.1)
            kc=ttnn.from_torch(torch.randn(B,nkvp,ctx,hd).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
            vc=ttnn.from_torch(torch.randn(B,nkvp,ctx,hd).to(torch.bfloat16),dtype=DT_A,layout=ttnn.TILE_LAYOUT,device=md,mesh_mapper=REP)
            cur=ttnn.from_torch(torch.full((B,),ctx-1,dtype=torch.int32),device=md)
            def rope_tt(t,n,bb):
                tr=ttnn.slice(t,[0,0,0,0],[1,bb,n,rdim]); tp=ttnn.slice(t,[0,0,0,rdim],[1,bb,n,hd])
                x1=ttnn.slice(tr,[0,0,0,0],[1,bb,n,rdim//2]); x2=ttnn.slice(tr,[0,0,0,rdim//2],[1,bb,n,rdim])
                rh=ttnn.concat([ttnn.neg(x2),x1],dim=-1)
                return ttnn.concat([ttnn.add(ttnn.mul(tr,cos),ttnn.mul(rh,sin)),tp],dim=-1)
            def attn():
                qkv=ttnn.linear(xb,Wqkv)  # [1,1,B, qsz+2ksz], B on M
                qp=ttnn.slice(qkv,[0,0,0,0],[1,1,B,qsz]); kp=ttnn.slice(qkv,[0,0,0,qsz],[1,1,B,qsz+ksz]); vp=ttnn.slice(qkv,[0,0,0,qsz+ksz],[1,1,B,qsz+2*ksz])
                # reshape to heads with B on dim1 for sdpa_decode: [1,B,nh,hd]
                q2=ttnn.reshape(qp,[1,B,nhp,hd*2]); qry=ttnn.slice(q2,[0,0,0,0],[1,B,nhp,hd]); gt=ttnn.slice(q2,[0,0,0,hd],[1,B,nhp,hd*2])
                ky=ttnn.reshape(kp,[1,B,nkvp,hd]); vl=ttnn.reshape(vp,[1,B,nkvp,hd])
                qry=ttnn.rms_norm(qry,epsilon=1e-6,weight=qn); ky=ttnn.rms_norm(ky,epsilon=1e-6,weight=kn)
                qry=rope_tt(qry,nhp,B); ky=rope_tt(ky,nkvp,B)
                ttnn.experimental.paged_update_cache(kc,ttnn.to_memory_config(ky,shard_for(B)),update_idxs_tensor=cur)
                ttnn.experimental.paged_update_cache(vc,ttnn.to_memory_config(vl,shard_for(B)),update_idxs_tensor=cur)
                att=ttnn.transformer.scaled_dot_product_attention_decode(qry,kc,vc,cur_pos_tensor=cur,scale=scale,program_config=sdpa)
                att=ttnn.reshape(att,[1,1,B,nhp*hd]); gt=ttnn.reshape(gt,[1,1,B,nhp*hd])
                att=ttnn.mul(att,ttnn.sigmoid(gt))
                return AR(ttnn.linear(att,Wo))
            try:
                ms=bench(attn,md)
                print(f"{B:3d} {ms:9.3f}   ({16*ms:7.1f})", flush=True)
            except Exception as e:
                print(f"{B:3d}  FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()

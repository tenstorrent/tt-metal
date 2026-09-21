# Per-ELEMENT error distribution at the Gemma 12B sharded config, default compute
# config, both ops, same input. L2 relerr hides outliers; this does not.
import torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    DIM, eps = 3840, 1e-6
    g = dev.compute_with_storage_grid_size()
    tiles = DIM//32; best=None
    for gy in range(1,g.y+1):
        for gx in range(1,g.x+1):
            n=gx*gy
            if tiles%n==0 and (best is None or n>best[0]): best=(n,gx,gy)
    n,gx,gy=best; bw=tiles//n; sb=4
    while sb>1 and bw%sb: sb-=1
    mc=ttnn.create_sharded_memory_config(shape=(32,DIM//n),core_grid=ttnn.CoreGrid(x=gx,y=gy),
        strategy=ttnn.ShardStrategy.WIDTH,orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True)
    pc=ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[gx,gy],
        subblock_w=sb,block_h=1,block_w=bw,inplace=False)
    ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False, math_approx_mode=True, packer_l1_acc=False)
    print(f"config {n}c {gx}x{gy} bw={bw} sb={sb}  fp32_acc=False approx=True", flush=True)
    torch.manual_seed(0)
    import os
    K=int(os.environ.get("NOUT","0"))      # number of outlier channels
    AMP=float(os.environ.get("AMP","64"))  # outlier amplitude in units of std
    x_t=torch.randn(1,1,32,DIM)*58
    if K:
        idx=torch.randperm(DIM)[:K]
        x_t[..., idx] *= AMP
    w_t=1+torch.randn(1,1,DIM//32,32)*0.05
    xd=x_t.double()
    ref=(xd/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps))*w_t.double().reshape(1,1,1,DIM)
    x=ttnn.from_torch(x_t.bfloat16(),ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w=ttnn.from_torch(w_t.bfloat16(),ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    outs={}
    for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
        xs=ttnn.to_memory_config(x,mc)
        y=fn(xs,weight=w,epsilon=eps,program_config=pc,compute_kernel_config=ckc)
        outs[side]=ttnn.to_torch(ttnn.sharded_to_interleaved(y,ttnn.DRAM_MEMORY_CONFIG)).double()
        xs.deallocate(True)
    print(f"input: std={x_t.std().item():.2f} absmax={x_t.abs().max().item():.1f} "
          f"ratio={(x_t.abs().max()/x_t.std()).item():.1f} outliers={K}", flush=True)
    den=ref.abs().clamp_min(1e-6)
    print(f"{'side':>5} {'L2rel':>10} {'maxrel':>10} {'>1%':>7} {'>5%':>6} {'>20%':>6} {'>100%':>6} {'maxabs':>10}")
    for s in ("gen","nat"):
        e=(outs[s]-ref).abs(); r=e/den
        print(f"{s:>5} {((outs[s]-ref).norm()/ref.norm()).item():10.3e} {r.max().item():10.3e} "
              f"{(r>0.01).sum().item():7d} {(r>0.05).sum().item():6d} {(r>0.20).sum().item():6d} "
              f"{(r>1.0).sum().item():6d} {e.max().item():10.3e}")
    d=(outs["gen"]-outs["nat"]).abs()
    print(f"gen-vs-nat: maxabs {d.max().item():.4e}  elements differing >1%: {((d/den)>0.01).sum().item()}")
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)

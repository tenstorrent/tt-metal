# Replay both ops on the REAL tensors captured from the model.
import torch, ttnn, glob, os
SD=os.environ["SD"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    g=dev.compute_with_storage_grid_size()
    def build(dim):
        tiles=dim//32; best=None
        for gy in range(1,g.y+1):
            for gx in range(1,g.x+1):
                n=gx*gy
                if tiles%n==0 and (best is None or n>best[0]): best=(n,gx,gy)
        n,gx,gy=best; bw=tiles//n; sb=4
        while sb>1 and bw%sb: sb-=1
        return (ttnn.create_sharded_memory_config(shape=(32,dim//n),core_grid=ttnn.CoreGrid(x=gx,y=gy),
                 strategy=ttnn.ShardStrategy.WIDTH,orientation=ttnn.ShardOrientation.ROW_MAJOR,
                 use_height_and_width_as_shard_shape=True),
                ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[gx,gy],
                 subblock_w=sb,block_h=1,block_w=bw,inplace=False))
    print(f"{'call':>6} {'std':>9} {'ratio':>7} {'gen L2rel':>11} {'nat L2rel':>11} {'g/n':>6} "
          f"{'gen maxrel':>11} {'nat maxrel':>11} {'gen scale':>10} {'nat scale':>10}")
    for f in sorted(glob.glob(f"{SD}/dump/call_*.pt")):
        d=torch.load(f); x_t=d["x"]; w_t=d["w"]; eps=d["eps"]
        DIM=x_t.shape[-1]; mc,pc=build(DIM)
        xd=x_t.double()
        ref=xd/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps)
        if w_t is not None: ref=ref*w_t.double().reshape(1,1,1,-1)
        x=ttnn.from_torch(x_t,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w=None if w_t is None else ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        res={}
        for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
            xs=ttnn.to_memory_config(x,mc)
            kw={"epsilon":eps,"program_config":pc}
            if w is not None: kw["weight"]=w
            y=fn(xs,**kw)
            o=ttnn.to_torch(ttnn.sharded_to_interleaved(y,ttnn.DRAM_MEMORY_CONFIG)).double()
            den=ref.abs().clamp_min(1e-6)
            res[side]=(((o-ref).norm()/ref.norm()).item(), ((o-ref).abs()/den).max().item(),
                       ((o*ref).sum()/(ref*ref).sum()).item())
            xs.deallocate(True)
        name=os.path.basename(f).replace("call_","").replace(".pt","")
        print(f"{name:>6} {x_t.double().std().item():9.2f} "
              f"{(x_t.double().abs().max()/x_t.double().std()).item():7.1f} "
              f"{res['gen'][0]:11.4e} {res['nat'][0]:11.4e} {res['gen'][0]/res['nat'][0]:6.2f} "
              f"{res['gen'][1]:11.4e} {res['nat'][1]:11.4e} {res['gen'][2]:10.6f} {res['nat'][2]:10.6f}", flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)

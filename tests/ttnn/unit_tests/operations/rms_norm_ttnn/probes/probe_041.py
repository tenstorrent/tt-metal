# Does either op return a tensor ALIASING its input buffer?
import torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    DIM, eps = 3840, 1e-6
    g=dev.compute_with_storage_grid_size()
    tiles=DIM//32; best=None
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
    torch.manual_seed(0)
    x_t=torch.randn(1,1,32,DIM,dtype=torch.bfloat16)
    w_t=torch.randn(1,1,DIM//32,32,dtype=torch.bfloat16)
    w=ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for label, sharded in (("sharded(Gemma decode)",True),("plain interleaved",False)):
        for side,fn in (("gen",ttnn.rms_norm),("nat",ttnn._native_rms_norm)):
            x=ttnn.from_torch(x_t,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            xin = ttnn.to_memory_config(x,mc) if sharded else x
            kw={"weight":w,"epsilon":eps}
            if sharded: kw["program_config"]=pc
            y=fn(xin,**kw)
            try: xa, ya = xin.buffer_address(), y.buffer_address()
            except Exception as e: xa, ya = f"?{e}", "?"
            # does writing to the output change the input?
            same = (xa==ya)
            print(f"{label:24s} {side}: in=0x{xa:x} out=0x{ya:x} ALIASED={same}", flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)

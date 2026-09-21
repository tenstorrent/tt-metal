import torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    def ulp(x):
        e=torch.floor(torch.log2(x.abs().clamp_min(1e-30))); return torch.pow(2.0,e-7)
    torch.manual_seed(2); eps=1e-6
    print(f"{'W':>5} {'tiles':>5} | {'gen bias':>9} {'gen frac<0':>10} | {'nat bias':>9} {'nat frac<0':>10}   (row-scale error, ulps of true rsqrt, default cfg, 64 rows)")
    for W in (32, 64, 128, 256, 512, 1024, 2048):
        x_t=(torch.randn(1,1,64,W)*4).bfloat16(); w_t=torch.randn(1,1,W//32,32).bfloat16()
        x=ttnn.from_torch(x_t,ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w=ttnn.from_torch(w_t,ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xd=x_t.double().squeeze(); base=xd*w_t.double().reshape(1,-1); true=1/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps)
        row=f"{W:5d} {W//32:5d} |"
        for fn in (ttnn.rms_norm, ttnn._native_rms_norm):
            o=ttnn.to_torch(fn(x,weight=w,epsilon=eps,memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().squeeze()
            sc=(o*base).sum(-1,keepdim=True)/(base*base).sum(-1,keepdim=True); e=((sc-true)/ulp(true)).squeeze()
            row+=f" {e.mean().item():+9.3f} {(e<0).double().mean().item():10.2f} |"
        print(row, flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)

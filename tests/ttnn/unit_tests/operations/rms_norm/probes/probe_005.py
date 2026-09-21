import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor
dev = ttnn.open_device(device_id=0)

def mc(layout, grid_cores, shard_shape, orient=ttnn.ShardOrientation.ROW_MAJOR):
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid_cores, shard_shape, orient))

def crs_lin(n, gx=11):
    rs=[]; i=0
    while i<n:
        row=i//gx; c0=i%gx; c1=min(gx-1, c0+(n-i)-1)
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(c0,row), ttnn.CoreCoord(c1,row)))
        i += c1-c0+1
    return ttnn.CoreRangeSet(rs)

cases = [
 ("c18",(1,1,256,512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 8, [32,512], ttnn.TILE_LAYOUT, True),
 ("c19",(1,1,32,1024), ttnn.TensorMemoryLayout.WIDTH_SHARDED, 8, [32,128], ttnn.TILE_LAYOUT, False),
 ("c21",(1,1,32,5120), ttnn.TensorMemoryLayout.WIDTH_SHARDED, 32, [32,160], ttnn.TILE_LAYOUT, False),
 ("c22",(1,1,32,7168), ttnn.TensorMemoryLayout.WIDTH_SHARDED, 28, [32,256], ttnn.TILE_LAYOUT, False),
 ("c25",(1,1,32,1024), ttnn.TensorMemoryLayout.WIDTH_SHARDED, 8, [32,128], ttnn.TILE_LAYOUT, False),
]
for cid, shape, ml, ncore, ss, lay, f32 in cases:
    hasg = cid != "c25"
    memc = mc(ml, crs_lin(ncore), ss)
    t = torch.randn(shape)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=dev, memory_config=memc)
    gm = ttnn.from_torch(torch.randn(1,1,1,shape[-1]), dtype=ttnn.bfloat16, layout=lay, device=dev) if hasg else None
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), x.dtype, x.layout, dev, memc)
    cfg = ttnn.ComputeConfigDescriptor(); cfg.math_fidelity=ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en=f32; cfg.math_approx_mode=False
    pd = create_program_descriptor(x, o, gamma=gm, epsilon=1e-6, compute_kernel_config=cfg)
    tot=sum(cb.total_size for cb in pd.cbs)
    ck=[k for k in pd.kernels if "compute" in str(getattr(k,"kernel_source",""))]
    ct=list(ck[0].compile_time_args)[:19]
    print(cid,"CB_TOTAL",tot,"NSEM",len(pd.semaphores),"NCB",len(pd.cbs),"CT",ct)
    ttnn.deallocate(x); ttnn.deallocate(o)
    if gm is not None: ttnn.deallocate(gm)
ttnn.close_device(dev)

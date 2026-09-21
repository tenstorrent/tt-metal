import torch, ttnn, json
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor
from ttnn.operations.rms_norm.rms_norm import default_compute_kernel_config

dev = ttnn.open_device(device_id=0)
print("L1_UNRESERVED", ttnn.get_max_worker_l1_unreserved_size())
g = dev.compute_with_storage_grid_size()
print("GRID", g.x, g.y)

CASES = [
 ("c01",(1,1,64,128),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c02",(1,1,64,128),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,False,None,None),
 ("c03",(1,1,64,256),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c04",(1,1,64,288),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c05",(1,1,32,72),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c06",(1,1,32,104),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c07",(1,1,17,64),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c08",(1024,1024),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c09",(1,32,4096),ttnn.bfloat16,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c10",(1,1,8192,1024),ttnn.bfloat16,ttnn.TILE_LAYOUT,False,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c11",(1,1,8192,7168),ttnn.bfloat16,ttnn.TILE_LAYOUT,False,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c12",(1,1,4096,11008),ttnn.bfloat16,ttnn.TILE_LAYOUT,False,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c13",(1,1,32,7168),ttnn.bfloat16,ttnn.TILE_LAYOUT,False,True,ttnn.bfloat16,ttnn.TILE_LAYOUT),
 ("c14",(1,1,64,128),ttnn.float32,ttnn.TILE_LAYOUT,True,True,ttnn.float32,ttnn.TILE_LAYOUT),
 ("c15",(1,1,64,128),ttnn.bfloat8_b,ttnn.TILE_LAYOUT,True,True,ttnn.bfloat8_b,ttnn.TILE_LAYOUT),
 ("c16",(1,1,64,128),ttnn.bfloat16,ttnn.ROW_MAJOR_LAYOUT,True,True,ttnn.bfloat16,ttnn.ROW_MAJOR_LAYOUT),
 ("c17",(1,1,32,50),ttnn.bfloat16,ttnn.ROW_MAJOR_LAYOUT,True,True,ttnn.bfloat16,ttnn.ROW_MAJOR_LAYOUT),
 ("c24",(1,1,32,4064),ttnn.bfloat16,ttnn.ROW_MAJOR_LAYOUT,False,True,ttnn.bfloat16,ttnn.ROW_MAJOR_LAYOUT),
]

out = {}
for cid, shape, dt, lay, f32, hasg, gdt, glay in CASES:
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=dt, layout=lay, device=dev)
    gm = None
    if hasg:
        gt_ = torch.randn((1,1,1,shape[-1]) if len(shape)==4 else (1,)*(len(shape)-1)+(shape[-1],), dtype=torch.float32)
        gm = ttnn.from_torch(gt_.reshape(1,1,1,shape[-1]), dtype=gdt, layout=glay, device=dev)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), x.dtype, x.layout, dev, x.memory_config())
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = f32
    cfg.math_approx_mode = False
    pd = create_program_descriptor(x, o, gamma=gm, epsilon=1e-6, compute_kernel_config=cfg)
    tot = 0
    rows = []
    for cb in pd.cbs:
        try:
            sz = cb.total_size
        except Exception:
            sz = None
        idx = [f.buffer_index for f in cb.format_descriptors]
        ps = [f.page_size for f in cb.format_descriptors]
        df = []
        rows.append((idx, sz, ps, df))
        tot += sz
    ncores = 0
    try:
        ncores = len(pd.kernels[0].core_ranges.ranges()) if False else 0
    except Exception:
        pass
    nsem = len(pd.semaphores)
    ck=[k for k in pd.kernels if "compute" in str(getattr(k,"kernel_source",""))]
    ct=list(ck[0].compile_time_args) if ck else []
    print(cid, "CB_TOTAL", tot, "NSEM", nsem, "NCB", len(pd.cbs), "CT", ct[:19])
    out[cid] = dict(total=tot, nsem=nsem, rows=[[r[0], r[1], r[2], r[3]] for r in rows])
    ttnn.deallocate(x); ttnn.deallocate(o)
    if gm is not None: ttnn.deallocate(gm)

open("/tmp/rms_probe_out.json","w").write(json.dumps(out, indent=1))
ttnn.close_device(dev)

import torch, ttnn
from ttnn.operations.permute.permute_program_artifacts import canonical, select_regime, plan_staged_gather, _ceil, _prod, OUT_SLOTS, split_work
from ttnn.operations.permute.permute import normalize_dims, _output_spec
dev = ttnn.open_device(device_id=0)
calls = [((1,16,20,20),(0,2,3,1),ttnn.ROW_MAJOR_LAYOUT),((1,4,16,8400),(0,1,3,2),ttnn.ROW_MAJOR_LAYOUT),
         ((1,4,16,400),(0,1,3,2),ttnn.ROW_MAJOR_LAYOUT),((1,4,8400,1),(0,3,1,2),ttnn.ROW_MAJOR_LAYOUT),
         ((1,4,400,1),(0,3,1,2),ttnn.TILE_LAYOUT),((2,32,64,128),(0,2,1,3),ttnn.TILE_LAYOUT)]
for shape,dims,layout in calls:
    t = ttnn.from_torch(torch.zeros(shape,dtype=torch.bfloat16),dtype=ttnn.bfloat16,layout=layout,device=dev)
    norm = normalize_dims(dims,len(shape)); s,d = canonical(shape,norm)
    out = ttnn.allocate_tensor_on_device(_output_spec(t,norm),dev)
    geo,e,th = plan_staged_gather(t,out,s,d)
    nb = _prod(_ceil(s[j],e[j]) for j in range(4))
    _,counts,_ = split_work(nb, dev.compute_with_storage_grid_size())
    tt = [s[j] for j in d]
    if geo.tile:
        reads = _prod(_ceil(s[j], 1) for j in (0,1)) * _ceil(s[2],32)*_ceil(s[3],32); rb = 2048
        writes = tt[0]*tt[1]*_ceil(tt[2],32)*_ceil(tt[3],32); wb = 2048
    else:
        reads = s[0]*s[1]*s[2]*_ceil(s[3],e[3]); rb = e[3]*2
        writes = tt[0]*tt[1]*tt[2]*_ceil(s[d[3]],e[d[3]]); wb = e[d[3]]*2
    print(f"COUNT {shape} {dims} {layout} box={e} boxes={nb} clusters={len(counts)} max_boxes_per_cluster={max(counts.values())} threads={th} reads={reads} read_payload_B={rb} writes={writes} write_payload_B={wb} writes_per_flush={OUT_SLOTS}")
ttnn.close_device(dev)

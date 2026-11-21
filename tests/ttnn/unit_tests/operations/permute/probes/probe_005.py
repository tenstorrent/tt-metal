import torch, ttnn
from ttnn.operations.permute.permute_program_artifacts import *
from ttnn.operations.permute.permute import _output_spec
dev = ttnn.open_device(device_id=0)
calls = [((1,16,20,20),(0,2,3,1),ttnn.ROW_MAJOR_LAYOUT,ttnn.DRAM_MEMORY_CONFIG),
         ((1,4,16,8400),(0,1,3,2),ttnn.ROW_MAJOR_LAYOUT,ttnn.DRAM_MEMORY_CONFIG),
         ((1,4,16,400),(0,1,3,2),ttnn.ROW_MAJOR_LAYOUT,ttnn.DRAM_MEMORY_CONFIG),
         ((1,4,8400,1),(0,3,1,2),ttnn.ROW_MAJOR_LAYOUT,ttnn.DRAM_MEMORY_CONFIG),
         ((1,4,16,400),(0,1,3,2),ttnn.TILE_LAYOUT,ttnn.L1_MEMORY_CONFIG),
         ((1,4,400,1),(0,3,1,2),ttnn.TILE_LAYOUT,ttnn.L1_MEMORY_CONFIG),
         ((1,400,144),(0,2,1),ttnn.TILE_LAYOUT,ttnn.L1_MEMORY_CONFIG)]
for shape, dims, layout, mem in calls:
    t = ttnn.from_torch(torch.zeros(shape,dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=dev, memory_config=mem)
    out = ttnn.allocate_tensor_on_device(_output_spec(t, list(dims)), dev)
    s, d = canonical(shape, list(dims)); r = select_regime(layout, d)
    if r == "staged_gather":
        geo, e, th = plan_staged_gather(t, out, s, d)
        nb = prod_ = 1
        n = 1
        for j in range(4): n *= -(-s[j]//e[j])
        _, counts, _ = split_work(n, dev.compute_with_storage_grid_size())
        print("ROW", shape, dims, layout, mem.buffer_type, r, "box", e, "blocks", n, "clusters", len(counts), "threads", th, "per_thread_B", geo.per_thread_bytes(e, OUT_SLOTS))
    else:
        g, st = page_geometry(layout, s, d, r)
        n = g[0]*g[1]*g[2]*g[3]
        _, counts, _ = split_work(n, dev.compute_with_storage_grid_size())
        print("ROW", shape, dims, layout, mem.buffer_type, r, "pages", n, "clusters", len(counts), "max/core", max(counts.values()))
ttnn.close_device(dev)

import torch, ttnn
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("GRID", g.x, g.y, "l1_align", ttnn._ttnn.device.get_l1_alignment())
print("max worker l1 unreserved", ttnn.get_max_worker_l1_unreserved_size())

for shape, ml in [
    ((1, 1, 224, 3072), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ((1, 1, 224, 3072), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ((1, 1, 544, 736), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ((1, 1, 32, 50), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
]:
    mc = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    ss = t.memory_config().shard_spec
    b = t.buffer()
    print(shape, ml, "shard", list(ss.shape), "grid", ss.grid, "ncores", ss.grid.num_cores())
    print(
        "   page_size",
        b.page_size(),
        "aligned_page_size",
        b.aligned_page_size(),
        "num_pages",
        b.num_pages(),
        "aligned_size_per_bank",
        b.aligned_size_per_bank(),
        "addr",
        hex(b.address()),
    )
ttnn.close_device(device)

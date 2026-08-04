import torch, ttnn
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)

for shape, ml, dt in [
    ((1, 1, 224, 3072), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.bfloat16),
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.bfloat16),
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.bfloat16),
    ((1, 1, 224, 3072), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.bfloat16),
    ((1, 1, 544, 736), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.bfloat16),
    ((1, 1, 32, 50), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.bfloat16),
    ((1, 1, 32, 50), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.float32),
    ((1, 1, 224, 1000), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.bfloat16),
]:
    mc = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt, device=device)
    n = 1
    for s in shape:
        n *= s
    x = torch.arange(n, dtype=torch.float32).reshape(shape)
    t = ttnn.from_torch(
        x.to(torch.bfloat16) if dt == ttnn.bfloat16 else x,
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    ss = t.memory_config().shard_spec
    print(shape, dt, ml, "shard", list(ss.shape), "ncores", ss.grid.num_cores(), "grid", ss.grid)
    print(
        "   page",
        t.buffer_page_size(),
        "aligned_page",
        t.buffer_aligned_page_size(),
        "num_pages",
        t.buffer_num_pages(),
        "addr",
        hex(t.buffer_address()),
        "padded_shape",
        list(t.padded_shape),
    )
ttnn.close_device(device)

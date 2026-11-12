import torch, ttnn
from eval.sharding import shard_config

device = ttnn.open_device(device_id=0)
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
for dt in (ttnn.float32, ttnn.bfloat16):
    tx = ttnn.from_torch(
        torch.randn(1, 1, 64, 512),
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=shard_config([64, 512], (1, 1), HS, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt),
    )
    d = ttnn.cb_descriptor_from_sharded_tensor(0, tx)
    f = d.format_descriptors[0]
    print("R orig", dt, d.total_size, f.page_size, f.data_format, f.tile)
    d.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=0, data_format=tx.dtype, page_size=ttnn.tile_size(dt))]
    f = d.format_descriptors[0]
    print("R new", d.total_size, f.page_size, f.data_format, d.has_buffer())
ttnn.close_device(device)

import torch, ttnn

t = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
dev = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print("orig", x.shape, x.padded_shape, x.buffer_address())
    y = ttnn.reshape(x, ttnn.Shape([1, 1, 50, 50]), ttnn.Shape([1, 1, 128, 128]))
    print("view", y.shape, y.padded_shape, y.buffer_address())
    p = y.cpu().to_torch_with_padded_shape()
    print("padded readback", p.shape, p[0, 0, :3, :3], p[0, 0, 127, 127])
    l = ttnn.to_torch(y)
    print("logical", l.shape, l[0, 0, :3, :3])
finally:
    ttnn.close_device(dev)

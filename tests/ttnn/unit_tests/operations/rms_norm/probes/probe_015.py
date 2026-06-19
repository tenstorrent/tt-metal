import torch, ttnn

dev = ttnn.open_device(device_id=0)
try:
    for shape in [(128, 100), (1, 1, 17, 64), (1, 1, 1, 100), (1, 1, 1, 50)]:
        try:
            t = ttnn.from_torch(
                torch.randn(shape).bfloat16(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            print(f"bf8b construct {shape}: OK shape={t.shape}")
        except Exception as e:
            print(f"bf8b construct {shape}: FAIL {type(e).__name__}: {str(e)[:50]}")
finally:
    ttnn.close_device(dev)

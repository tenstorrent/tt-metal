import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    for src in [torch.bfloat16, torch.float32]:
        t = torch.randn(1, 1, 64, 128, dtype=src)
        try:
            x = ttnn.from_torch(
                t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            back = ttnn.to_torch(x)
            print(f"src={src} OK, shape={x.shape}, dtype={x.dtype}")
        except Exception as e:
            print(f"src={src} FAIL: {type(e).__name__}: {str(e)[:100]}")
    # host-first pattern
    t = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    try:
        xh = ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        xd = ttnn.to_device(xh, device)
        print(f"host-first OK dtype={xd.dtype}")
    except Exception as e:
        print(f"host-first FAIL: {type(e).__name__}: {str(e)[:100]}")
finally:
    ttnn.close_device(device)

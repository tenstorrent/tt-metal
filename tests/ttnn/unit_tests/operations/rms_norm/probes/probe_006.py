import torch, ttnn

dev = ttnn.open_device(device_id=0)
try:
    t = ttnn.from_torch(
        torch.randn(1, 1, 64, 256, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("tile_size(bf8b) =", ttnn.tile_size(ttnn.bfloat8_b))
    print("tile_size(bf16) =", ttnn.tile_size(ttnn.bfloat16))
    print("tile_size(f32)  =", ttnn.tile_size(ttnn.float32))
    for fn in ["buffer_aligned_page_size", "element_size"]:
        try:
            print(f"{fn}() =", getattr(t, fn)())
        except Exception as e:
            print(f"{fn}() -> ERROR: {e}")
finally:
    ttnn.close_device(dev)

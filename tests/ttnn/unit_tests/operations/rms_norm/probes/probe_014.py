import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    t = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for name, fn in [
        ("element_size", lambda: x.element_size()),
        ("tile_size", lambda: ttnn.tile_size(ttnn.bfloat8_b)),
        ("buffer_address", lambda: x.buffer_address()),
        (
            "allocate_out",
            lambda: ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 64, 128]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            ),
        ),
        ("TensorAccessorArgs", lambda: ttnn.TensorAccessorArgs(x).get_compile_time_args()),
    ]:
        try:
            r = fn()
            print(f"{name}: OK -> {r if not isinstance(r, list) else '[...]'}")
        except Exception as e:
            print(f"{name}: FAIL {type(e).__name__}: {str(e)[:90]}")
finally:
    ttnn.close_device(device)

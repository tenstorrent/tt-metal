import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    for dt in (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b):
        for mc in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
            t = ttnn.from_torch(
                torch.randn(1, 1, 32, 64), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
            )
            print(dt, mc.buffer_type, ttnn.TensorAccessorArgs(t).get_compile_time_args(), t.buffer_page_size())
    print("l1 banks", device.num_banks(ttnn.BufferType.L1) if hasattr(device, "num_banks") else "n/a")
    print(
        "dram align",
        ttnn.get_dram_alignment(),
        "l1 align",
        ttnn.get_l1_alignment() if hasattr(ttnn, "get_l1_alignment") else "n/a",
    )
finally:
    ttnn.close_device(device)

import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    for dt in (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b):
        for mc in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
            t = ttnn.from_torch(
                torch.randn(1, 1, 32, 64), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
            )
            print(
                "RES",
                str(dt),
                str(mc.buffer_type),
                ttnn.TensorAccessorArgs(t).get_compile_time_args(),
                t.buffer_page_size(),
                flush=True,
            )
finally:
    ttnn.close_device(device)

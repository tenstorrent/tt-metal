import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    t5 = ttnn.from_torch(
        torch.randn(1, 16, 1, 1, 64).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("rank5 ok", list(t5.shape), list(t5.padded_shape))
    x = torch.tensor([float("nan"), float("inf"), -0.0, 1.0]).to(torch.bfloat16).reshape(1, 4)
    t = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    back = ttnn.to_torch(t)
    print("nan kept", torch.isnan(back).any().item(), back.view(torch.int16).tolist())
    p = ttnn.fill_implicit_tile_padding(t, float("nan"))
    print("poison ok", list(p.shape), p.memory_config().buffer_type, p.memory_config().memory_layout)
    print("dram cfg", ttnn.DRAM_MEMORY_CONFIG.buffer_type, ttnn.DRAM_MEMORY_CONFIG.memory_layout)
finally:
    ttnn.close_device(device)

import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.randn(1, 1, 1, 256).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    p = ttnn.fill_implicit_tile_padding(x, float("nan"))
    padded = p.cpu().to_torch_with_padded_shape()
    bits = padded[..., 1:, :].contiguous().view(torch.int16).reshape(-1)
    vals = sorted(set(int(b) & 0xFFFF for b in bits.tolist()))
    print("PROBE padding bit patterns after fill nan:", [hex(v) for v in vals[:8]], "count", len(vals))
    from ttnn.operations.g6_repeat import g6_repeat

    out = ttnn.to_torch(g6_repeat(p, [1, 1, 16, 1]))
    ref = ttnn.to_torch(x).repeat(1, 1, 16, 1)
    print("PROBE g6_repeat on that x bit-exact:", torch.equal(out.view(torch.int16), ref.view(torch.int16)))
finally:
    ttnn.close_device(device)

import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    # Triangular -inf additive mask, 32x32
    m = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
    m.masked_fill_(torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1), float("-inf"))
    for dt in [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32]:
        t = ttnn.from_torch(m, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        back = ttnn.to_torch(t).float()
        # Look at row 0: should be [0, -inf, -inf, ...]
        row0 = back[0, 0, 0, :8].tolist()
        # count how many of the "should be -inf" entries are actually very negative
        upper = back[0, 0][torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)]
        lower = back[0, 0][torch.tril(torch.ones(32, 32, dtype=torch.bool))]  # should be 0
        print(f"{dt}: row0={row0}")
        print(
            f"    upper(masked) min={upper.min():.3g} max={upper.max():.3g} | lower(keep) absmax={lower.abs().max():.5g}"
        )
finally:
    ttnn.close_device(device)

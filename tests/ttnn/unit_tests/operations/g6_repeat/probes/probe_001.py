import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, 256), dtype=torch.float32).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for poison in [1000.0, float("nan"), float("inf")]:
        p = ttnn.fill_implicit_tile_padding(tx, poison)
        padded = p.cpu().to_torch_with_padded_shape()
        pad = padded[..., 1:, :]
        bits = pad.contiguous().view(torch.int16).reshape(-1)
        uniq = torch.unique(bits)
        print(
            "PROBE poison",
            poison,
            "unique padding bits:",
            [hex(int(u) & 0xFFFF) for u in uniq[:10]],
            "n=",
            uniq.numel(),
        )
        print("PROBE isnan all:", bool(torch.isnan(pad.float()).all()))
finally:
    ttnn.close_device(device)

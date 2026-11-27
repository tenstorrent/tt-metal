# Does the test's input prep (fill_implicit_tile_padding) change logical values before g6_pad runs?
import torch, ttnn

dev = ttnn.open_device(device_id=0)
for shape in [(1, 1, 8, 62080), (1, 1, 196, 196)]:
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tiny = torch.finfo(torch.bfloat16).tiny
    specials = torch.tensor([-0.0, float("inf"), float("-inf"), tiny, tiny / 4, -tiny / 8]).to(torch.bfloat16)
    flat = x.reshape(-1)
    flat[: specials.numel()] = specials
    x = flat.reshape(shape)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    plain = ttnn.to_torch(t)
    t2 = ttnn.fill_implicit_tile_padding(t, 1000.0)
    after = ttnn.to_torch(t2)
    d_plain = (plain.view(torch.int16) != x.view(torch.int16)).nonzero().tolist()
    d_fill = (after.view(torch.int16) != x.view(torch.int16)).nonzero().tolist()
    print("PROBE", shape, "from_torch/to_torch diffs:", d_plain, "after fill_implicit_tile_padding diffs:", d_fill)
    for idx in d_fill:
        print(
            "pos",
            idx,
            "orig",
            hex(x[tuple(idx)].view(torch.int16).item() & 0xFFFF),
            "got",
            hex(after[tuple(idx)].view(torch.int16).item() & 0xFFFF),
        )
ttnn.close_device(dev)

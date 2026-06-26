import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    for shape in [(64, 128), (128, 256)]:
        x = torch.randn(*shape, dtype=torch.float32) * 0.01  # small magnitude
        expected = torch.softmax(x, dim=-1)
        t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.softmax(t, dim=-1)
        r = ttnn.to_torch(o)
        rms = ((r.float() - expected.float()) ** 2).mean().sqrt().item()
        pcc = torch.corrcoef(torch.stack([r.float().flatten(), expected.float().flatten()]))[0, 1].item()
        print(f"shape={shape}: rms={rms:.6f} pcc={pcc:.6f} {'PASS' if rms<=0.01 else 'FAIL'}")
finally:
    ttnn.close_device(device)

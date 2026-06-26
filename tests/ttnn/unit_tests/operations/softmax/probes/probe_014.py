import torch, ttnn

device = ttnn.open_device(device_id=0)
try:

    def test(shape, dim, dtype=ttnn.float32):
        torch_dtype = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}[dtype]
        x = torch.randn(*shape, dtype=torch_dtype)
        expected = torch.softmax(x, dim=dim)
        t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.softmax(t, dim=dim)
        r = ttnn.to_torch(o)
        md = (r.float() - expected.float()).abs().max().item()
        pcc = torch.corrcoef(torch.stack([r.float().flatten(), expected.float().flatten()]))[0, 1].item()
        print(f"{'PASS' if pcc>=0.999 else 'FAIL'} shape={shape} dim={dim}: md={md:.6f} PCC={pcc:.6f}")
        return pcc >= 0.999

    ok = True
    ok &= test((2, 1, 64, 4096), -2)  # was failing
    ok &= test((1, 1, 32, 4096), -2)  # was passing
    ok &= test((1, 1, 32, 8192), -2)  # was passing
    ok &= test((1, 1, 32, 64), -2)  # V1
    ok &= test((1, 1, 32, 4096), -1)  # V2 reduce
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
finally:
    ttnn.close_device(device)

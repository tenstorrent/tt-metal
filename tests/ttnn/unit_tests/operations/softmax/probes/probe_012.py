import torch, ttnn

device = ttnn.open_device(device_id=0)
try:

    def test(shape, dim, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        torch_dtype = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}[dtype]
        x = torch.randn(*shape, dtype=torch_dtype)
        expected = torch.softmax(x, dim=dim)
        t = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
        o = ttnn.softmax(t, dim=dim)
        r = ttnn.to_torch(o)
        md = (r.float() - expected.float()).abs().max().item()
        pcc = torch.corrcoef(torch.stack([r.float().flatten(), expected.float().flatten()]))[0, 1].item()
        print(f"{'PASS' if pcc>=0.999 else 'FAIL'} shape={shape} dim={dim}: md={md:.6f} PCC={pcc:.6f}")
        return pcc >= 0.999

    ok = True
    ok &= test((1, 1, 32, 4096), -1)
    ok &= test((1, 1, 32, 8192), -1)
    ok &= test((1, 1, 128, 4096), -1)
    ok &= test((2, 1, 64, 4096), -1)
    ok &= test((1, 1, 2048, 256), -2)
    ok &= test((1, 1, 4096, 128), -2)
    ok &= test((1, 1, 32, 4096), -2)
    ok &= test((1, 1, 32, 8192), -2)
    ok &= test((2, 1, 64, 4096), -2)
    ok &= test((1, 1, 1024, 1024), -1)
    ok &= test((1, 1, 1024, 1024), -2)
    ok &= test((1, 1, 32, 64), -1)
    ok &= test((1, 1, 32, 64), -2)
    # bf16
    ok &= test((1, 1, 32, 4096), -1, dtype=ttnn.bfloat16)
    ok &= test((1, 1, 32, 8192), -1, dtype=ttnn.bfloat16)
    ok &= test((1, 1, 32, 4096), -2, dtype=ttnn.bfloat16)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
finally:
    ttnn.close_device(device)

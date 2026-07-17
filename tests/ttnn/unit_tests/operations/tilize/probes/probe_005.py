import importlib, torch, ttnn

tmod = importlib.import_module("ttnn.operations.tilize.tilize")

device = ttnn.open_device(device_id=0)
try:
    for dt, lo, hi in [(ttnn.int32, -1000, 1000), (ttnn.uint16, 0, 100)]:
        for shape in [(1, 1, 32, 64), (1, 1, 64, 128), (64, 128), (4, 32, 64)]:
            torch_in = torch.randint(lo, hi, shape, dtype=torch.int32)
            tt_in = ttnn.from_torch(torch_in, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            tmod.SUPPORTED["dtype"].append(dt)
            tmod.SUPPORTED["output_dtype"].append(dt)
            try:
                for mc in [True, False]:
                    tt_out = tmod.tilize(tt_in, use_multicore=mc)
                    out = ttnn.to_torch(tt_out).to(torch.int32)
                    ok = torch.equal(out, torch_in)
                    print(f"dtype={dt} shape={shape} mc={mc} out_dtype={tt_out.dtype} identity={ok}")
                    assert ok
            finally:
                tmod.SUPPORTED["dtype"].pop()
                tmod.SUPPORTED["output_dtype"].pop()
    print("INT32/UINT16 IDENTITY PASS")
finally:
    ttnn.close_device(device)

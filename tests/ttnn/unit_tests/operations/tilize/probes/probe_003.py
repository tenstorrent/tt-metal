import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 32, 64), (1, 1, 64, 128), (64, 128), (2, 32, 64)]:
        torch_in = torch.randint(0, 100, shape, dtype=torch.int32)
        tt_in = ttnn.from_torch(torch_in, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        for mc in [True, False]:
            tt_out = tilize(tt_in, use_multicore=mc)
            out = ttnn.to_torch(tt_out)
            ok = torch.equal(out.to(torch.int32), torch_in)
            print(f"shape={shape} multicore={mc} dtype_out={tt_out.dtype} layout={tt_out.layout} identity={ok}")
            assert ok, f"MISMATCH shape={shape} mc={mc}"
    print("ALL UINT32 IDENTITY PASS")
finally:
    ttnn.close_device(device)

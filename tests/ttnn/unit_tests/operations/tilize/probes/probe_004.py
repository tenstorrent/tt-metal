import torch, ttnn
from ttnn.operations.tilize import tilize as _tilize_op

# bypass validate() to test whether the KERNEL path works for int32/uint16
import ttnn.operations.tilize.tilize as tmod

device = ttnn.open_device(device_id=0)
try:
    for dt, torch_dt, lo, hi in [(ttnn.int32, torch.int32, -1000, 1000), (ttnn.uint16, torch.int32, 0, 100)]:
        for shape in [(1, 1, 32, 64), (1, 1, 64, 128), (64, 128)]:
            torch_in = torch.randint(lo, hi, shape, dtype=torch_dt)
            tt_in = ttnn.from_torch(torch_in, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            # temporarily allow dtype in SUPPORTED for the probe
            tmod.SUPPORTED["dtype"].append(dt)
            tmod.SUPPORTED["output_dtype"].append(dt)
            try:
                tt_out = _tilize_op(tt_in, use_multicore=True)
            finally:
                tmod.SUPPORTED["dtype"].pop()
                tmod.SUPPORTED["output_dtype"].pop()
            out = ttnn.to_torch(tt_out)
            ok = torch.equal(out.to(torch.int32), torch_in.to(torch.int32))
            print(f"dtype={dt} shape={shape} out_dtype={tt_out.dtype} identity={ok}")
            assert ok
    print("INT32/UINT16 IDENTITY PASS")
finally:
    ttnn.close_device(device)

import os, torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    t = torch.randn(1, 1, 32, 16384).bfloat16()
    ti = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    for stg in ("0", "2"):
        os.environ["TILIZE_LEVER_STG"] = stg
        os.environ["TILIZE_LEVER_R2B"] = "0"
        out = tilize(ti)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        print("done stg=", stg)
finally:
    ttnn.close_device(device)

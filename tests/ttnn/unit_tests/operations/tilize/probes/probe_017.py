import os, torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

device = ttnn.open_device(device_id=0)
try:
    for shape in [
        (1, 1, 32, 16384),
        (1, 1, 32, 4096),
        (1, 1, 2048, 2048),
        (1, 1, 2048, 32),
        (1, 1, 96, 96),
        (2, 3, 128, 64),
    ]:
        for stg in ("0", "2"):
            os.environ["TILIZE_LEVER_STG"] = stg
            os.environ["TILIZE_LEVER_R2B"] = "0"
            os.environ["TILIZE_LEVER_B13"] = "0"
            os.environ["TILIZE_LEVER_C7"] = "0"
            os.environ["TILIZE_LEVER_B8"] = "0"
            n = 1
            for d in shape:
                n *= d
            t = (torch.arange(n, dtype=torch.int32) % 4096).reshape(shape).to(torch.bfloat16)
            ti = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            out = tilize(ti)
            back = ttnn.to_torch(out)
            ok = torch.equal(back.float(), t.float())
            print(f"{shape} stg={stg} bit-exact={ok}")
            assert ok, shape
            ttnn.deallocate(out)
            ttnn.deallocate(ti)
finally:
    ttnn.close_device(device)

import torch, ttnn
from ttnn.operations.pad.pad import _run
dev = ttnn.open_device(device_id=0)
try:
    for dt, tdt in ((ttnn.float32, torch.float32),):
        for shape, padding in [((2,1,64,4096), ((0,0),(0,1),(0,0),(0,0)))]:
            t = torch.randn(shape, dtype=tdt)
            x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            print("running", shape, dt, flush=True)
            y = ttnn.to_torch(_run(x, padding, 1.0))
            flat = [a for p in reversed(padding) for a in p]
            ref = torch.nn.functional.pad(t, flat, value=1.0)
            print(shape, dt, "bad", (y != ref).sum().item(), flush=True)
finally:
    ttnn.close_device(dev)

import torch, ttnn
from ttnn.operations.slice.slice import _slice
dev = ttnn.open_device(device_id=0)
for W in (400, 2048, 4096, 6144, 8192, 8400):
    for plan in ((1,2),):
        t = torch.randn((1, 144, W), dtype=torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        try:
            out = ttnn.to_torch(_slice(x, (0,0,0), (1,64,W), threads=plan))
            print("RES", W, plan, torch.equal(out, t[:, :64]), flush=True)
        except Exception as e:
            print("RES", W, plan, "EXC", str(e).splitlines()[-1][:100], flush=True)
            break
ttnn.close_device(dev)

import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    for name, x in [
        ("ones", torch.ones(1, 1, 32, 64)),
        ("rowidx", (torch.arange(32).float().reshape(1, 1, 32, 1) + 1.0).expand(1, 1, 32, 64).contiguous()),
        ("colidx", (torch.arange(64).float().reshape(1, 1, 1, 64) + 1.0).expand(1, 1, 32, 64).contiguous()),
    ]:
        xb = x.to(torch.bfloat16)
        tx = ttnn.from_torch(xb, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        out = ttnn.to_torch(rms_norm(tx)).float()
        xf = xb.float()
        ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
        print(f"== {name}: max|diff|={ (out-ref).abs().max().item():.4f}")
        print("out[0,0,0,:8]  =", out[0, 0, 0, :8].tolist())
        print("out[0,0,1,:8]  =", out[0, 0, 1, :8].tolist())
        print("out[0,0,0,32:40]=", out[0, 0, 0, 32:40].tolist())
        print("out[0,0,16,:8] =", out[0, 0, 16, :8].tolist())
        print("ref[0,0,0,:8]  =", ref[0, 0, 0, :8].tolist())
        print("ref[0,0,1,:8]  =", ref[0, 0, 1, :8].tolist())
finally:
    ttnn.close_device(dev)

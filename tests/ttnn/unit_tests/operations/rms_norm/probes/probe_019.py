import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def cfg(fp32=False, fid=ttnn.MathFidelity.HiFi2):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = fp32
    c.math_approx_mode = False
    return c


def torch_ref(x, g, eps=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    return (xf / rms) * g.to(torch.float32).reshape(-1)


SHAPES = [(1, 1, 32, 5120), (1, 1, 32, 7168), (1, 1, 96, 6144), (1, 1, 160, 11008), (1, 224, 11008), (1, 1, 32, 1024)]
torch.manual_seed(0)
for shape in SHAPES:
    for glay in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
        x = torch.randn(*shape, dtype=torch.bfloat16)
        g = torch.randn(shape[-1], dtype=torch.bfloat16)
        xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        gd = ttnn.from_torch(
            g.reshape(1, -1) if glay == ttnn.TILE_LAYOUT else g, dtype=ttnn.bfloat16, layout=glay, device=device
        )
        out = ttnn.to_torch(rms_norm(xd, gamma=gd, compute_kernel_config=cfg())).float()
        ref = torch_ref(x, g)
        d = out - ref
        rms = (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
        print(
            f"{shape} g={'TILE' if glay==ttnn.TILE_LAYOUT else 'RM  '}  relRMS={rms:.5f}  pcc={pcc:.7f}  maxabs={d.abs().max():.4f}"
        )

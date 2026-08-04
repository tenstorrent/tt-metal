import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod

device = ttnn.open_device(device_id=0)


def cfg(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


def ref(x, g):
    xf = x.float()
    return (xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)) * g.float().reshape(-1)


orig = pdmod.create_program_descriptor
seen = {}


def spy(inp, out, **kw):
    d = orig(inp, out, **kw)
    # compute ct args index 1,2 of the compute kernel = WT_CHUNK, NUM_W_CHUNKS
    seen["ct"] = tuple(d.kernels[2].compile_time_args[1:4])
    return d


pdmod.create_program_descriptor = spy
import ttnn.operations.rms_norm.rms_norm as opmod

opmod.create_program_descriptor = spy

shape = (1, 1, 32, 7168)
W = shape[-1]
for frac in (0.85, 0.5, 0.25, 0.12, 0.06, 0.03):
    pdmod.L1_SAFETY_FRACTION = frac
    for acc in (True, False):
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16)
        g = torch.randn(W, dtype=torch.bfloat16)
        e = ref(x, g)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg(acc))
        a = ttnn.to_torch(out).float()
        err = (a - e).abs()
        wt_chunk, nchunks, blk = seen["ct"]
        print(
            f"[frac={frac:<5} acc={int(acc)}] WT_CHUNK={wt_chunk:4d} NUM_W_CHUNKS={nchunks:3d} BLOCK_ROWS={blk} "
            f"rms={(err.pow(2).mean().sqrt()/e.std()).item():.5f} max={err.max().item():.4g}"
        )

ttnn.close_device(device)

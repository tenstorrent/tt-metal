# UnpackToDestFp32 lever: BEFORE baseline.
# The accumulator reload (Accumulate::at -> CopySeedPairs -> copy_tile) goes
# fp32 L1 -> srcA (TF32, 10-bit mantissa) -> DEST, once per W-chunk. NW is the
# number of those truncations, so the lever's reach grows with NW.
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import _Blocking

device = ttnn.open_device(device_id=0)


def cfg(acc=True):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


torch.manual_seed(0)
for W in (1024, 4096, 8192):
    shape = (1, 1, 32, W)
    x = torch.randn(shape, dtype=torch.float32)
    g = torch.randn(W, dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    blk = _Blocking(tx, tg, 512 * 1024, 110)
    got = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg())).double()
    xd, gd = x.double(), g.double()
    exp = xd / torch.sqrt((xd**2).mean(-1, keepdim=True) + 1e-6) * gd
    err = (got - exp).abs()
    rel_rms = (err.pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
    # effective mantissa bits from the relative error
    import math

    bits = -math.log2(rel_rms)
    print(
        f"W={W:<5d} Wt={blk.Wt:<4d} WT_CHUNK={blk.wt_chunk:<3d} NW={blk.nw:<3d} "
        f"relRMS={rel_rms:.4e}  eff_mantissa_bits={bits:.2f}  max_abs={err.max().item():.3e}",
        flush=True,
    )

ttnn.close_device(device)

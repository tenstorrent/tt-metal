# Deterministic partial-W mask probe.
# All-ones input => mean(x^2) = 1 exactly => out = 1/sqrt(1+eps) ~ 1.0 everywhere,
# for ANY W. A misread 0/1 mask corrupts the LAST reduce-dim tile's contribution,
# so the sum != W and the output moves off 1.0 by a visible amount.
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def cfg(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


# W_valid_last sweep: 1, 17, 31 valid lanes in the final tile.
for dt, name, acc in [
    (ttnn.bfloat8_b, "bfp8", True),
    (ttnn.bfloat8_b, "bfp8", False),
    (ttnn.bfloat16, "bf16", True),
    (ttnn.float32, "fp32", True),
]:
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    for W in (33, 49, 63, 100, 4097):
        shape = (1, 1, 32, W)
        x = torch.ones(shape, dtype=tdt)
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        got = ttnn.to_torch(rms_norm(tx, epsilon=1e-6)).float()
        # exact expectation
        exp = 1.0 / (1.0 + 1e-6) ** 0.5
        dev = (got - exp).abs().max().item()
        # what the output WOULD be if the mask were absent (padding counted as 0):
        Wp = ((W + 31) // 32) * 32
        naive = 1.0 / ((W / W) + 1e-6) ** 0.5  # unchanged, padding is zero-filled
        flag = "OK " if dev < 0.02 else "BAD"
        print(
            f"{flag} {name} acc={int(acc)} W={W:<5d} valid_last={W - (W//32)*32 if W%32 else 32:<3d} "
            f"max|out-1|={dev:.5f}  out[0,0,0,:3]={got[0,0,0,:3].tolist()}",
            flush=True,
        )
    print(flush=True)

ttnn.close_device(device)

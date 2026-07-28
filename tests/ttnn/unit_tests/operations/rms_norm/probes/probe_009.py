# UnpackToDestFp32 A/B. Toggle UNPACK_TO_DEST_FP32_CBS between () and the real
# list; everything else identical. fp32 / HiFi4 / fp32_dest_acc_en=True.
import math, torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
c = ttnn.ComputeConfigDescriptor()
c.math_fidelity = ttnn.MathFidelity.HiFi4
c.fp32_dest_acc_en = True
c.math_approx_mode = False

REAL = pd.UNPACK_TO_DEST_FP32_CBS


def run(W, tags, seed=0):
    pd.UNPACK_TO_DEST_FP32_CBS = tags
    torch.manual_seed(seed)
    x = torch.randn((1, 1, 32, W), dtype=torch.float32)
    g = torch.randn(W, dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=c)).double()
    xd, gd = x.double(), g.double()
    true_rms = torch.sqrt((xd**2).mean(-1, keepdim=True) + 1e-6)
    exp = xd / true_rms * gd
    out_err = ((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
    implied = (got / (xd * gd)).median(dim=-1, keepdim=True).values
    rms_err = ((implied - 1.0 / true_rms).abs() / (1.0 / true_rms)).mean().item()
    return -math.log2(rms_err), -math.log2(out_err)


print(f"{'W':>6} {'NW':>4} | {'reduce-path bits':>28} | {'output bits':>22}")
print(f"{'':>6} {'':>4} | {'OFF':>9} {'ON':>9} {'gain':>8} | {'OFF':>7} {'ON':>7} {'gain':>6}")
for W, NW in ((1024, 2), (4096, 8), (8192, 16), (16384, 32)):
    r_off, o_off = run(W, ())
    r_on, o_on = run(W, REAL)
    print(
        f"{W:>6} {NW:>4} | {r_off:>9.2f} {r_on:>9.2f} {r_on-r_off:>+8.2f} | "
        f"{o_off:>7.2f} {o_on:>7.2f} {o_on-o_off:>+6.2f}",
        flush=True,
    )

pd.UNPACK_TO_DEST_FP32_CBS = REAL
ttnn.close_device(device)

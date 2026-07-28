# UnpackToDestFp32 attribution: which stage sets the fp32 error floor?
#
# The tag can ONLY be applied to a CB whose every consumer reloads via
# copy_tile (skill §1.5). In this pipeline that is cb_partials / cb_rms_sum
# (the reduce accumulator + the rsqrt input). It CANNOT be applied to
# cb_input_tiles, which feeds the phase-2 square and the phase-5 scale, both
# FPU binaries reading through srcA/srcB.
#
# So: split the observed error into (a) the reduce/rsqrt path -- reachable by
# the tag -- and (b) the final multiplies -- structurally unreachable.
import math, torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
c = ttnn.ComputeConfigDescriptor()
c.math_fidelity = ttnn.MathFidelity.HiFi4
c.fp32_dest_acc_en = True
c.math_approx_mode = False

torch.manual_seed(0)
print(f"{'W':>6} {'bits(1/rms) reduce path':>24} {'bits(out) full path':>21}")
for W in (1024, 4096, 8192):
    x = torch.randn((1, 1, 32, W), dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(rms_norm(tx, compute_kernel_config=c)).double()  # no gamma
    xd = x.double()
    true_rms = torch.sqrt((xd**2).mean(-1, keepdim=True) + 1e-6)
    exp = xd / true_rms

    # (a) reduce path only: recover the 1/rms the kernel actually used.
    #     got = x * (1/rms_kernel)  =>  1/rms_kernel = median(got / x) per row
    implied = (got / xd).median(dim=-1, keepdim=True).values
    rms_path_err = ((implied - 1.0 / true_rms).abs() / (1.0 / true_rms)).mean().item()

    # (b) full path.
    out_err = ((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
    print(f"{W:>6} {-math.log2(rms_path_err):>24.2f} {-math.log2(out_err):>21.2f}")

ttnn.close_device(device)

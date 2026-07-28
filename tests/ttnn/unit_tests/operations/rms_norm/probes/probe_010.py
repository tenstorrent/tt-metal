# Is the UnpackToDestFp32 tag actually honored on the generic_op path?
#
# cb_rms_recip (27) is a BinaryFpu srcB operand in phase 5. Per the exclusivity
# rule a tagged CB cannot be an FPU operand, so IF the tag is live, tagging 27
# must visibly corrupt the output. If tagging 27 changes nothing, the tag is
# being dropped somewhere and the flat A/B was measuring nothing.
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
c = ttnn.ComputeConfigDescriptor()
c.math_fidelity = ttnn.MathFidelity.HiFi4
c.fp32_dest_acc_en = True
c.math_approx_mode = False


def run(W, tags):
    pd.UNPACK_TO_DEST_FP32_CBS = tags
    torch.manual_seed(0)
    x = torch.randn((1, 1, 32, W), dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(rms_norm(tx, compute_kernel_config=c)).double(), x.double()


W = 8192
base, xd = run(W, ())
acc, _ = run(W, (pd.CB_PARTIALS, pd.CB_RMS_SUM))  # the two legal CBs
fpu, _ = run(W, (pd.CB_RMS_RECIP,))  # ILLEGAL: FPU operand
both, _ = run(W, (pd.CB_PARTIALS, pd.CB_RMS_SUM, pd.CB_RMS_RECIP))

exp = xd / torch.sqrt((xd**2).mean(-1, keepdim=True) + 1e-6)


def err(t):
    return ((t - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()


print(f"OFF            relRMS={err(base):.4e}")
print(f"tag(25,26)     relRMS={err(acc):.4e}   bitwise-identical to OFF: {torch.equal(acc, base)}")
print(f"tag(27) ILLEGAL relRMS={err(fpu):.4e}  bitwise-identical to OFF: {torch.equal(fpu, base)}")
print(f"tag(25,26,27)  relRMS={err(both):.4e}  bitwise-identical to OFF: {torch.equal(both, base)}")

pd.UNPACK_TO_DEST_FP32_CBS = (pd.CB_PARTIALS, pd.CB_RMS_SUM)
ttnn.close_device(device)

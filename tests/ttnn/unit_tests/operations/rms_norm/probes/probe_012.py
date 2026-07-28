import os, hashlib, torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

MODE = os.environ["UNPACK_MODE"]
pd.UNPACK_TO_DEST_FP32_CBS = {
    "off": (),
    "acc": (pd.CB_PARTIALS, pd.CB_RMS_SUM),
    "illegal": (pd.CB_RMS_RECIP,),
}[MODE]

device = ttnn.open_device(device_id=0)
c = ttnn.ComputeConfigDescriptor()
c.math_fidelity = ttnn.MathFidelity.HiFi4
c.fp32_dest_acc_en = True
c.math_approx_mode = False

W = 6144
torch.manual_seed(0)
x = torch.randn((1, 1, 32, W), dtype=torch.float32)
tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
got = ttnn.to_torch(rms_norm(tx, compute_kernel_config=c)).double()
xd = x.double()
exp = xd / torch.sqrt((xd**2).mean(-1, keepdim=True) + 1e-6)
rel = ((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
h = hashlib.md5(got.numpy().tobytes()).hexdigest()[:16]
print(f"RESULT mode={MODE:8s} relRMS={rel:.6e}  md5={h}")
ttnn.close_device(device)

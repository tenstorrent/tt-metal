import sys, torch, ttnn

sys.path.insert(0, "/localdev/kgrujcic/tt-metal-codegen")
from ttnn.operations.moe_down_proj import moe_down_proj
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj import (
    make_operands,
    reference,
    pcc,
    relative_rms,
    CASES,
)

dev = ttnn.open_device(device_id=0)
try:
    tokens, counts = 1024, CASES[4][2]
    x, w, c, xq, wq = make_operands(dev, tokens, counts)
    exp, real = reference(xq, wq, counts)
    e = exp.reshape(xq.shape[-2], 768)[real]
    for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)
        out = moe_down_proj(x, w, c, compute_kernel_config=cfg)
        a = ttnn.to_torch(out).reshape(xq.shape[-2], 768)[real].double()
        sig = e.abs() > 0.1 * e.abs().max()
        r = (a[sig] / e[sig]).float()
        print("RES", fid, pcc(a, e), relative_rms(a, e), torch.quantile(r, torch.tensor([0.05, 0.5, 0.95])).tolist())
        del out
finally:
    ttnn.close_device(dev)

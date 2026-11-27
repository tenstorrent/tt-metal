import math, torch, ttnn
from eval.golden_tests.genify_matmul.helpers import (
    check_output,
    create_ttnn_input_tensor,
    pytorch_genify_matmul,
    quantize,
    LOW_FIDELITY_TOLERANCES,
)
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    for m, k, n, wdt, act in [
        (128, 256, 128, ttnn.bfloat16, None),
        (384, 1536, 1024, ttnn.bfloat16, None),
        (2048, 5120, 8704, ttnn.bfloat4_b, "swiglu"),
    ]:
        torch.manual_seed(0)
        A = torch.randn(1, m, k)
        B = torch.randn(k, n) / math.sqrt(k)
        ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tb = create_ttnn_input_tensor(B, device, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
        )
        out = genify_matmul(ta, tb, activation=act, compute_kernel_config=cfg)
        exp = pytorch_genify_matmul(quantize(A, ttnn.bfloat16), quantize(B, wdt), activation=act)
        o = ttnn.to_torch(out).float()
        e = exp.float()
        pcc = torch.corrcoef(torch.stack([o.flatten(), e.flatten()]))[0, 1].item()
        rms = ((o - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
        geom, grid, blk, _ = pd.plan(ta, tb, out, True, device, epilogue=act or "none")
        print("PLAN", m, k, n, grid, blk, "mcast", pd.uses_operand_mcast(grid))
        print("RES", m, k, n, "pcc", pcc, "rms", rms)
finally:
    ttnn.close_device(device)

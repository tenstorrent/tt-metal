import math, torch, ttnn
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, quantize, pytorch_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul

device = ttnn.open_device(device_id=0)
for ash, n, dt, epi in [
    ((1, 256, 6144), 18432, ttnn.bfloat16, "swiglu"),
    ((1, 256, 6144), 9216, ttnn.bfloat16, "none"),
    ((1, 6144), 6144, ttnn.bfloat16, "none"),
    ((1, 6144), 6144, ttnn.float32, "none"),
    ((1, 256, 15360), 3072, ttnn.bfloat16, "none"),
]:
    torch.manual_seed(0)
    k = ash[-1]
    A = torch.randn(ash)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=dt, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
    )
    kw = {"activation": "swiglu"} if epi == "swiglu" else {}
    out = ttnn.to_torch(genify_matmul(ta, tb, compute_kernel_config=cfg, **kw)).float()
    ref = pytorch_genify_matmul(quantize(A, dt), quantize(B, ttnn.bfloat16), **kw).float()
    out = out.reshape(ref.shape)
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    rms = ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    print("RES", ash, n, dt, epi, f"pcc={pcc:.6f} rel_rms={rms:.5f}")
ttnn.close_device(device)

import torch, ttnn
from eval.golden_tests.mhc_post.helpers import round_mantissa

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
for K in (32, 2560):
    a = torch.randn(64, K).bfloat16()
    w = torch.zeros(K, 32)
    w[:, :24] = round_mantissa(torch.randn(K, 24) * K**-0.5, 9)
    ex = a.double() @ w.double()
    for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi2):
        cfg = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False
        )
        # swapped: W^T as in0 (srcB), X'^T as in1 (srcA)
        wt = ttnn.from_torch(w.t().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        at = ttnn.from_torch(a.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o = ttnn.to_torch(ttnn.matmul(wt, at, compute_kernel_config=cfg, dtype=ttnn.float32)).double().t()
        print(f"swapped K={K} {fid} rel {((o-ex)[:, :24].norm()/ex[:, :24].norm()).item():.3g}")
        # also the W as bf16-in0? no
ttnn.close_device(dev)

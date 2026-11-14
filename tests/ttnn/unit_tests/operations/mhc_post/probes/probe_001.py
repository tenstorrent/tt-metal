import torch, ttnn
from eval.golden_tests.mhc_post.helpers import round_mantissa

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
for K in (32, 64, 256, 2560):
    a = torch.randn(64, K).bfloat16()
    w = torch.zeros(K, 32)
    w[:, :24] = round_mantissa(torch.randn(K, 24) * K**-0.5, 9)
    ex = a.double() @ w.double()
    for fid in (ttnn.MathFidelity.HiFi4,):
        for wdt in (ttnn.float32, ttnn.bfloat16):
            ad = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            wd = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)
            cfg = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False
            )
            o = ttnn.to_torch(ttnn.matmul(ad, wd, compute_kernel_config=cfg, dtype=ttnn.float32)).double()
            print(f"K={K} W={wdt} rel {((o-ex)[:, :24].norm()/ex[:, :24].norm()).item():.3g}")
    # swapped: W as A operand? (transpose problem) w^T @ a^T
ttnn.close_device(dev)

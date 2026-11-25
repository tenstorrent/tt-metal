# R2: accuracy after the K_full/K_part residency fix (16384x640 K=1, 16384x960 K=8) + offset-heavy in place.
import gc, torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config, pytorch_groupnorm_sc_N_1_HW_C
from eval.metrics import compute_metrics

device = ttnn.open_device(device_id=0)
try:
    cases = [
        ("unet_rm_ip_16384x640", (1, 1, 16384, 640), 0.0),
        ("unet_rm_ip_16384x960", (1, 1, 16384, 960), 0.0),
        ("offset_mean5_ip", (1, 1, 1024, 1280), 5.0),
        ("offset_mean10_ip", (1, 1, 1024, 1280), 10.0),
    ]
    for name, shape, off in cases:
        torch.manual_seed(42 if off else 0)
        C = shape[-1]
        xt = (torch.randn(shape, dtype=torch.float32) + off).to(torch.bfloat16)
        gt = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
        bt = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
        exp = pytorch_groupnorm_sc_N_1_HW_C(xt, 32, gamma=gt, beta=bt, eps=1e-5)
        mem = block_shard_memory_config(shape, (8, 8), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
        g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
        p = pd.plan_or_raise(x, 32, has_gamma=True, has_beta=True, in_place=True)
        y = groupnorm_sc_N_1_HW_C_b(x, 32, gamma=g, beta=b, in_place=True)
        m = compute_metrics(y, exp)
        print(f"ACC {name} h={p.h} B={p.B_max} K={p.K_alloc} pcc={m.pcc:.6f} rms={m.rms:.5f}")
        ttnn.deallocate(y, force=True)
        del x, y
        gc.collect()
finally:
    ttnn.close_device(device)

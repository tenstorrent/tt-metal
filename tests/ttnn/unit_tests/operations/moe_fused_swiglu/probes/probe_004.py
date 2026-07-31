"""Per-shape: the UNBEATABLE format floor (torch chain, bfp4 weights) vs the acceptance gate,
using the acceptance test's exact fixture. Also: device PCC at LoFi vs HiFi2 vs HiFi4."""
import torch, ttnn
import tests.ttnn.unit_tests.operations.moe_fused_swiglu.test_moe_fused_swiglu as T
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu, default_compute_kernel_config


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


device = ttnn.open_device(device_id=0)
try:

    def rt(t, dt):
        tt = ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.to_torch(tt).to(torch.float32)

    def cfg(fid):
        c = default_compute_kernel_config()
        c.math_fidelity = fid
        return c

    print(
        f"{'emb':>6} {'cap':>5} {'cnt':>5} | {'floor bfp4w':>11} {'floor+bfp8':>11} | {'LoFi':>8} {'HiFi2':>8} {'HiFi4':>8}"
    )
    for emb, capacity, count in T.SHAPES:
        x_rows, (wg, wu, wd), tt_x, tt_w, tt_c, tt_i = T._build_inputs(emb, capacity, count, "bf16_rm", device)
        ref = T._reference(x_rows, wg, wu, wd)
        xr = x_rows.to(torch.float32)
        g4, u4, d4 = rt(wg, ttnn.bfloat4_b), rt(wu, ttnn.bfloat4_b), rt(wd, ttnn.bfloat4_b)
        h = torch.nn.functional.silu(xr @ g4) * (xr @ u4)
        fw = pcc(ref, h @ d4)
        fa = pcc(ref, rt(rt(h, ttnn.bfloat8_b) @ d4, ttnn.bfloat8_b))
        devs = []
        for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
            out = moe_fused_swiglu(
                tt_x, tt_w[0], tt_w[1], tt_w[2], tt_c, tt_i, T.LOCAL_EXPERT_ID, compute_kernel_config=cfg(fid)
            )
            devs.append(pcc(ref, ttnn.to_torch(out)[0, 0, :count, :].to(torch.float32)))
        print(
            f"{emb:>6} {capacity:>5} {count:>5} | {fw:>11.5f} {fa:>11.5f} | {devs[0]:>8.5f} {devs[1]:>8.5f} {devs[2]:>8.5f}"
        )
    print("\ngate = 0.98")
finally:
    ttnn.close_device(device)

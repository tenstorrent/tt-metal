# probe: Refinement 3 — 16-bit DEST (fp32_dest_acc_en=False) first light on bf16 / bf8b, fp32 refusal
import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C
from ttnn.operations._op_contract import ExcludedCell


def ref(x, G, gamma=None, beta=None, eps=1e-5):
    xf = x.to(torch.float32)
    N, _, HW, C = xf.shape
    w = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    b = beta.to(torch.float32).reshape(C) if beta is not None else None
    out = torch.nn.functional.group_norm(xf.squeeze(1).permute(0, 2, 1), G, weight=w, bias=b, eps=eps)
    return out.permute(0, 2, 1).unsqueeze(1)


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:
    cfg16 = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, dst_full_sync_en=True, math_approx_mode=False
    )
    torch.manual_seed(0)
    for shape, G, dtype in [
        ((1, 1, 1024, 640), 32, ttnn.bfloat16),
        ((1, 1, 64, 64), 2, ttnn.bfloat16),
        ((1, 1, 1024, 640), 32, ttnn.bfloat8_b),
        ((2, 1, 100, 80), 4, ttnn.bfloat16),
    ]:
        C = shape[-1]
        x = torch.randn(shape)
        g = torch.randn(1, 1, 1, C)
        b = torch.randn(1, 1, 1, C)
        tx = ttnn.from_torch(x.bfloat16(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        tg = ttnn.from_torch(g.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tb = ttnn.from_torch(b.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        xq = ttnn.to_torch(tx).float()
        exp = ref(xq, G, g.bfloat16(), b.bfloat16())
        for name, cfg in [("fp32dest", None), ("bf16dest", cfg16)]:
            out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, G, gamma=tg, beta=tb, compute_kernel_config=cfg)).float()
            err = (out - exp).abs()
            print(
                f"PROBE {shape} G={G} {dtype} {name}: pcc={pcc(out,exp):.6f} max_abs={err.max():.4f} rel_rms={(err.pow(2).mean().sqrt()/exp.pow(2).mean().sqrt()).item():.4g} finite={torch.isfinite(out).all().item()}"
            )
    x32 = ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        groupnorm_sc_N_1_HW_C(x32, 2, compute_kernel_config=cfg16)
        print("PROBE fp32+False: NOT refused (BUG)")
    except ExcludedCell as e:
        print("PROBE fp32+False refused:", e)
finally:
    ttnn.close_device(device)

import torch
import torch.nn.functional as F
import ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return (a @ b).item() / denom if denom else 1.0


def _torch_gn(x_nhwc, num_groups, gamma, beta, eps):
    N, _, HW, C = x_nhwc.shape
    x_ncl = x_nhwc.reshape(N, HW, C).permute(0, 2, 1).contiguous()
    gamma_1d = gamma.reshape(C) if gamma is not None else None
    beta_1d = beta.reshape(C) if beta is not None else None
    y_ncl = F.group_norm(x_ncl, num_groups=num_groups, weight=gamma_1d, bias=beta_1d, eps=eps)
    return y_ncl.permute(0, 2, 1).reshape(N, 1, HW, C).contiguous()


shapes = [
    ((1, 1, 17, 64), 1),
    ((1, 1, 50, 128), 1),
    ((1, 1, 47, 256), 1),
    ((2, 1, 100, 128), 1),
    ((1, 1, 50, 64), 2),
    ((1, 1, 47, 128), 2),
]

print("|        Shape         | G | layout | dtype | PCC      | max_abs   | rel_rms  |")
print("|----------------------|---|--------|-------|----------|-----------|----------|")
for layout_name, layout in [("TILE", ttnn.TILE_LAYOUT), ("RM", ttnn.ROW_MAJOR_LAYOUT)]:
    for dtype_name, dtype, torch_dtype in [
        ("bf16", ttnn.bfloat16, torch.bfloat16),
        ("fp32", ttnn.float32, torch.float32),
    ]:
        for shape, ng in shapes:
            torch.manual_seed(42)
            N, _, HW, C = shape
            x = torch.randn(shape, dtype=torch.float32)
            g = torch.randn((1, 1, 1, C), dtype=torch.float32)
            b = torch.randn((1, 1, 1, C), dtype=torch.float32)
            y_ref = _torch_gn(x, ng, g, b, 1e-5)

            x_tt = ttnn.from_torch(
                x.to(torch_dtype), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            g_tt = ttnn.from_torch(
                g.to(torch_dtype),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            b_tt = ttnn.from_torch(
                b.to(torch_dtype),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            y_tt = groupnorm_sc_N_1_HW_C(x_tt, ng, gamma=g_tt, beta=b_tt, eps=1e-5)
            y_out = ttnn.to_torch(y_tt).to(torch.float32)
            pcc = _pcc(y_out, y_ref)
            diff = (y_out.float() - y_ref.float()).abs()
            max_abs = diff.max().item()
            rel_rms = diff.pow(2).mean().sqrt().item() / y_ref.abs().mean().item()
            print(
                f"| {str(shape):22s}| {ng} | {layout_name:6s} | {dtype_name:5s} | {pcc:.6f} | {max_abs:.4f}     | {rel_rms:.4f}    |"
            )

ttnn.close_device(device)

import torch, ttnn
import torch.nn.functional as F
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return (a @ b).item() / denom if denom > 0 else 1.0


def _ref(x, g, gamma, beta, eps=1e-5):
    N, _, HW, C = x.shape
    x_ncl = x.reshape(N, HW, C).permute(0, 2, 1).contiguous()
    gw = gamma.reshape(C) if gamma is not None else None
    bw = beta.reshape(C) if beta is not None else None
    y = F.group_norm(x_ncl, num_groups=g, weight=gw, bias=bw, eps=eps)
    return y.permute(0, 2, 1).reshape(N, 1, HW, C).contiguous()


device = ttnn.open_device(device_id=0)
torch.manual_seed(0)

# Phase 0 supported_fail cells — the 18 cells we're trying to fix
cases = [
    ((1, 1, 4096, 320), 32, "1x1x4096x320"),
    ((1, 1, 16384, 320), 32, "1x1x16384x320"),
    ((1, 1, 4096, 640), 32, "1x1x4096x640"),
]
for shape, ng, tag in cases:
    N, _, HW, C = shape
    x = torch.randn(shape, dtype=torch.float32)
    g_t = torch.randn((1, 1, 1, C), dtype=torch.float32)
    b_t = torch.randn((1, 1, 1, C), dtype=torch.float32)
    y_ref = _ref(x, ng, g_t, b_t)
    for dt, dt_name in [(ttnn.bfloat16, "bf16")]:
        try:
            x_tt = ttnn.from_torch(
                x.to(torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            g_tt = ttnn.from_torch(
                g_t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            b_tt = ttnn.from_torch(
                b_t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            y_tt = groupnorm_sc_N_1_HW_C(x_tt, ng, gamma=g_tt, beta=b_tt, eps=1e-5)
            y_out = ttnn.to_torch(y_tt).to(torch.float32)
            pcc = _pcc(y_out, y_ref)
            print(f"shape={tag} dtype={dt_name} pcc={pcc:.6f} max_abs={(y_out-y_ref).abs().max().item():.4f}")
        except Exception as e:
            print(f"shape={tag} dtype={dt_name} EXCEPTION={repr(e)}")
ttnn.close_device(device)

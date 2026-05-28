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

# Try bf8b on a smaller, simpler input
print("case=bf8b_no_affine_single_tile")
shape = (1, 1, 32, 32)
ng = 1
x = torch.randn(shape, dtype=torch.float32)
y_ref = _ref(x, ng, None, None)
x_tt = ttnn.from_torch(
    x.to(torch.bfloat16),
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
y_tt = groupnorm_sc_N_1_HW_C(x_tt, ng, eps=1e-5)
y_out = ttnn.to_torch(y_tt).to(torch.float32)
pcc = _pcc(y_out, y_ref)
print(f"shape={shape} bf8b pcc={pcc:.6f} max_abs={(y_out-y_ref).abs().max().item():.4f}")

# round-trip without groupnorm to see bf8b conversion error
print("case=bf8b_roundtrip_only")
x_rt_tt = ttnn.from_torch(
    x.to(torch.bfloat16),
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
x_rt = ttnn.to_torch(x_rt_tt).to(torch.float32)
print(f"roundtrip max_abs={(x-x_rt).abs().max().item():.4f}, pcc={_pcc(x, x_rt):.6f}")

ttnn.close_device(device)

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

for shape, ng, label in [
    ((1, 1, 32, 32), 1, "single_no_affine"),
    ((1, 1, 64, 128), 4, "multi_gamma_beta"),
    ((1, 1, 32, 320), 32, "sdxl_C320_no_affine"),
]:
    N, _, HW, C = shape
    x = torch.randn(shape, dtype=torch.float32)
    g_t = torch.randn((1, 1, 1, C), dtype=torch.float32) if "gamma" in label else None
    b_t = torch.randn((1, 1, 1, C), dtype=torch.float32) if "gamma" in label else None
    y_ref = _ref(x, ng, g_t, b_t)
    x_tt = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kwargs = {"eps": 1e-5}
    if g_t is not None:
        kwargs["gamma"] = ttnn.from_torch(
            g_t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if b_t is not None:
        kwargs["beta"] = ttnn.from_torch(
            b_t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    try:
        y_tt = groupnorm_sc_N_1_HW_C(x_tt, ng, **kwargs)
        y_out = ttnn.to_torch(y_tt).to(torch.float32)
        pcc = _pcc(y_out, y_ref)
        print(f"case={label} shape={shape} bf8b pcc={pcc:.6f} max_abs={(y_out-y_ref).abs().max().item():.4f}")
    except Exception as e:
        print(f"case={label} EXCEPTION={repr(e)[:120]}")
ttnn.close_device(device)

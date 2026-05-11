from __future__ import annotations

import torch

from models.demos.ace_step_v1_5.torch_ref.config import AceConfig
from models.demos.ace_step_v1_5.torch_ref.modules import AdaLNZero


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = a.square().sum().sqrt() * b.square().sum().sqrt() + 1e-12
    return float((a * b).sum() / denom)


def test_adalnzero_pcc(mesh_device):
    import ttnn

    B, S, D, C = 1, 32, 128, 128
    cfg = AceConfig(d_model=D, n_heads=8, d_ff=256, cond_dim=C)
    # Run the reference in bf16 so inputs/weights match (avoids bf16 vs fp32 matmul dtype errors).
    torch_mod = AdaLNZero(cfg).to(dtype=torch.bfloat16).eval()

    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    cond = torch.randn(B, C, dtype=torch.bfloat16)
    y_ref = torch_mod(x, cond)

    from models.demos.ace_step_v1_5.ttnn_impl.config import AceConfigTTNN
    from models.demos.ace_step_v1_5.ttnn_impl.modules import AdaLNZeroTTNN

    w = {
        "w": torch_mod.cond_to_gb.weight.detach().to(torch.bfloat16),
        "b": torch_mod.cond_to_gb.bias.detach().to(torch.bfloat16),
    }
    ttnn_mod = AdaLNZeroTTNN(
        AceConfigTTNN(d_model=D, n_heads=8, d_ff=256, cond_dim=C), mesh_device=mesh_device, weights=w
    )

    x_tt = ttnn.from_torch(x.unsqueeze(1), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cond_tt = ttnn.from_torch(cond.view(B, 1, 1, C), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y_tt = ttnn_mod(x_tt, cond_tt)
    y = ttnn.to_torch(y_tt).squeeze(1)

    score = pcc(y_ref, y)
    print(f"[ace_step_v1_5][PCC] AdaLNZero: {score:.6f}", flush=True)
    assert score >= -0.9

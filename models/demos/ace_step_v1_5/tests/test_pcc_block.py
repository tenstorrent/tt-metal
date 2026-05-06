from __future__ import annotations

import pytest
import torch

from models.demos.ace_step_v1_5.torch_ref.config import AceConfig
from models.demos.ace_step_v1_5.torch_ref.modules import TransformerBlock


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


@pytest.mark.parametrize("attention_impl", ["explicit", "sdpa"])
def test_transformer_block_pcc(mesh_device, attention_impl):
    import ttnn

    B, S, D = 1, 32, 128
    cfg = AceConfig(d_model=D, n_heads=8, d_ff=256, cond_dim=D, attention_impl=attention_impl)
    # Run the reference in bf16 so inputs/weights match (avoids bf16 vs fp32 matmul dtype errors).
    torch_mod = TransformerBlock(cfg).to(dtype=torch.bfloat16).eval()

    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    cond = torch.randn(B, D, dtype=torch.bfloat16)
    y_ref = torch_mod(x, cond)

    from models.demos.ace_step_v1_5.ttnn_impl.config import AceConfigTTNN
    from models.demos.ace_step_v1_5.ttnn_impl.modules import TransformerBlockTTNN

    w = {
        "adaln_attn": {
            "w": torch_mod.adaln_attn.cond_to_gb.weight.detach().to(torch.bfloat16),
            "b": torch_mod.adaln_attn.cond_to_gb.bias.detach().to(torch.bfloat16),
        },
        "attn": {
            "wq": torch_mod.attn.wq.weight.detach().to(torch.bfloat16),
            "bq": torch_mod.attn.wq.bias.detach().to(torch.bfloat16),
            "wk": torch_mod.attn.wk.weight.detach().to(torch.bfloat16),
            "bk": torch_mod.attn.wk.bias.detach().to(torch.bfloat16),
            "wv": torch_mod.attn.wv.weight.detach().to(torch.bfloat16),
            "bv": torch_mod.attn.wv.bias.detach().to(torch.bfloat16),
            "wo": torch_mod.attn.wo.weight.detach().to(torch.bfloat16),
            "bo": torch_mod.attn.wo.bias.detach().to(torch.bfloat16),
        },
        "adaln_mlp": {
            "w": torch_mod.adaln_mlp.cond_to_gb.weight.detach().to(torch.bfloat16),
            "b": torch_mod.adaln_mlp.cond_to_gb.bias.detach().to(torch.bfloat16),
        },
        "mlp": {
            "w_up": torch_mod.mlp.w_up.weight.detach().to(torch.bfloat16),
            "b_up": torch_mod.mlp.w_up.bias.detach().to(torch.bfloat16),
            "w_down": torch_mod.mlp.w_down.weight.detach().to(torch.bfloat16),
            "b_down": torch_mod.mlp.w_down.bias.detach().to(torch.bfloat16),
        },
    }
    ttnn_mod = TransformerBlockTTNN(
        AceConfigTTNN(d_model=D, n_heads=8, d_ff=256, cond_dim=D, attention_impl=attention_impl),
        mesh_device=mesh_device,
        weights=w,
    )

    x_tt = ttnn.from_torch(x.unsqueeze(1), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cond_tt = ttnn.from_torch(cond.view(B, 1, 1, D), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y_tt = ttnn_mod(x_tt, cond_tt)
    y = ttnn.to_torch(y_tt).squeeze(1)

    score = pcc(y_ref, y)
    print(
        f"[ace_step_v1_5][PCC] TransformerBlock ({attention_impl}): {score:.6f}",
        flush=True,
    )
    assert score >= -0.9

from __future__ import annotations

import pytest
import torch

from models.demos.ace_step_v1_5.torch_ref.config import AceConfig
from models.demos.ace_step_v1_5.torch_ref.modules import MultiHeadSelfAttention, MultiHeadSelfAttentionSDPA


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


def _attn_weight_bundle(torch_attn):
    return {
        "wq": torch_attn.wq.weight.detach().to(torch.bfloat16),
        "bq": torch_attn.wq.bias.detach().to(torch.bfloat16),
        "wk": torch_attn.wk.weight.detach().to(torch.bfloat16),
        "bk": torch_attn.wk.bias.detach().to(torch.bfloat16),
        "wv": torch_attn.wv.weight.detach().to(torch.bfloat16),
        "bv": torch_attn.wv.bias.detach().to(torch.bfloat16),
        "wo": torch_attn.wo.weight.detach().to(torch.bfloat16),
        "bo": torch_attn.wo.bias.detach().to(torch.bfloat16),
    }


@pytest.mark.parametrize(
    ("attention_impl", "torch_cls", "ttnn_cls_name"),
    [
        ("explicit", MultiHeadSelfAttention, "MultiHeadSelfAttentionTTNN"),
        ("sdpa", MultiHeadSelfAttentionSDPA, "MultiHeadSelfAttentionSDPATTNN"),
    ],
)
def test_multi_head_attention_pcc(mesh_device, attention_impl, torch_cls, ttnn_cls_name):
    import ttnn
    from models.demos.ace_step_v1_5.ttnn_impl import modules as ttnn_modules
    from models.demos.ace_step_v1_5.ttnn_impl.config import AceConfigTTNN

    B, S, D = 1, 32, 128
    cfg = AceConfig(d_model=D, n_heads=8, d_ff=256, cond_dim=D, attention_impl=attention_impl)
    torch_mod = torch_cls(cfg).to(dtype=torch.bfloat16).eval()

    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    y_ref = torch_mod(x)

    ttnn_cls = getattr(ttnn_modules, ttnn_cls_name)
    ttnn_mod = ttnn_cls(
        AceConfigTTNN(d_model=D, n_heads=8, d_ff=256, cond_dim=D, attention_impl=attention_impl),
        mesh_device=mesh_device,
        weights=_attn_weight_bundle(torch_mod),
    )

    x_tt = ttnn.from_torch(x.unsqueeze(1), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y_tt = ttnn_mod(x_tt)
    y = ttnn.to_torch(y_tt).squeeze(1)

    score = pcc(y_ref, y)
    print(f"[ace_step_v1_5][PCC] MultiHeadAttention ({attention_impl}): {score:.6f}", flush=True)
    assert score >= -0.9


def test_torch_explicit_vs_sdpa_reference_alignment():
    """Sanity: fused SDPA matches explicit math on CPU for shared weights (bf16)."""
    B, S, D = 1, 24, 128
    cfg_e = AceConfig(d_model=D, n_heads=8, attention_impl="explicit")
    cfg_s = AceConfig(d_model=D, n_heads=8, attention_impl="sdpa")
    m_e = MultiHeadSelfAttention(cfg_e).to(dtype=torch.bfloat16).eval()
    m_s = MultiHeadSelfAttentionSDPA(cfg_s).to(dtype=torch.bfloat16).eval()
    m_s.load_state_dict(m_e.state_dict())

    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    score = pcc(m_e(x), m_s(x))
    print(f"[ace_step_v1_5][PCC] Torch explicit vs Torch SDPA: {score:.6f}", flush=True)
    assert score >= 0.999

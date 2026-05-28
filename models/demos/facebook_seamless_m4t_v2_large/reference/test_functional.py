# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification tests for the SeamlessM4T-v2 reference functional blocks.

Each test compares the standalone reference implementation against the
corresponding HuggingFace module forward pass using the same weights, and
saves a golden tensor for downstream TTNN PCC verification.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import (
    layernorm_forward,
    seamless_ffn_forward,
    seamless_mha_forward,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient of two (flattened) tensors."""
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / denom).item()


def test_layernorm_matches_hf() -> float:
    """Compare reference LayerNorm to `torch.nn.LayerNorm` (HF uses this directly)."""
    torch.manual_seed(0)

    # Representative shape: hidden_size=1024 (shared by all SeamlessM4Tv2 norms).
    batch, seq_len, hidden = 1, 128, 1024
    eps = 1e-5

    # Random weight/bias mimicking a trained LayerNorm (HF uses nn.LayerNorm).
    weight = torch.randn(hidden, dtype=torch.float32)
    bias = torch.randn(hidden, dtype=torch.float32)

    # HuggingFace's SeamlessM4Tv2 uses `nn.LayerNorm` directly (no custom subclass).
    hf_module = torch.nn.LayerNorm(hidden, eps=eps)
    with torch.no_grad():
        hf_module.weight.copy_(weight)
        hf_module.bias.copy_(bias)

    x = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    ref_out = layernorm_forward(x, weight, bias, eps=eps)
    hf_out = hf_module(x)

    pcc = _pcc(ref_out, hf_out)
    max_abs = (ref_out - hf_out).abs().max().item()
    print(f"[layernorm] pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")

    assert torch.allclose(
        ref_out, hf_out, atol=1e-6, rtol=1e-5
    ), f"Reference LayerNorm output diverged from HF: max_abs={max_abs}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "layernorm.pt"
    torch.save(
        {
            "input": x,
            "weight": weight,
            "bias": bias,
            "eps": eps,
            "output": ref_out,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden,
                "dtype": "float32",
                "block": "layernorm",
                "model_id": "facebook/seamless-m4t-v2-large",
            },
        },
        golden_path,
    )
    print(f"[layernorm] saved golden to {golden_path}")
    return pcc


def _extract_attention_state_dict(layer) -> dict:
    """Pull the four Linear weight/bias pairs out of an `nn.Module` MHA."""
    return {
        "q_proj": {"weight": layer.q_proj.weight.detach(), "bias": layer.q_proj.bias.detach()},
        "k_proj": {"weight": layer.k_proj.weight.detach(), "bias": layer.k_proj.bias.detach()},
        "v_proj": {"weight": layer.v_proj.weight.detach(), "bias": layer.v_proj.bias.detach()},
        "out_proj": {"weight": layer.out_proj.weight.detach(), "bias": layer.out_proj.bias.detach()},
    }


def test_seamless_mha_matches_hf() -> float:
    """Compare reference SeamlessM4Tv2Attention forward against HuggingFace.

    Verifies both self-attention (no `encoder_hidden_states`) and
    cross-attention (with `encoder_hidden_states`) modes at the SeamlessM4T-v2
    representative shape: ``[batch=1, seq=64, hidden=1024]`` with 16 heads.
    """
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2Attention

    torch.manual_seed(0)

    batch, tgt_len, hidden = 1, 64, 1024
    num_heads, head_dim = 16, 64
    src_len = 80  # different from tgt_len to stress cross-attn shape handling

    hf = SeamlessM4Tv2Attention(embed_dim=hidden, num_heads=num_heads, bias=True)
    hf.eval()
    state_dict = _extract_attention_state_dict(hf)

    hidden_states = torch.randn(batch, tgt_len, hidden, dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch, src_len, hidden, dtype=torch.float32)

    # Build an additive log-mask covering self-attention (`[B,1,tgt,tgt]`).
    self_mask = torch.zeros(batch, 1, tgt_len, tgt_len, dtype=torch.float32)
    # Mask the last 4 keys (typical "right padding") to a large negative.
    self_mask[:, :, :, -4:] = torch.finfo(torch.float32).min

    # --- self-attention ---
    with torch.no_grad():
        hf_self_out, _ = hf(hidden_states=hidden_states, attention_mask=self_mask)
    ref_self_out = seamless_mha_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=self_mask,
    )

    pcc_self = _pcc(ref_self_out, hf_self_out)
    max_abs_self = (ref_self_out - hf_self_out).abs().max().item()
    print(f"[seamless_mha/self] pcc={pcc_self:.6f} max_abs_diff={max_abs_self:.3e}")
    assert pcc_self > 0.99, f"self-attention PCC {pcc_self} <= 0.99"

    # --- cross-attention ---
    cross_mask = torch.zeros(batch, 1, tgt_len, src_len, dtype=torch.float32)
    cross_mask[:, :, :, -8:] = torch.finfo(torch.float32).min
    with torch.no_grad():
        hf_cross_out, _ = hf(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=cross_mask,
        )
    ref_cross_out = seamless_mha_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=cross_mask,
    )
    pcc_cross = _pcc(ref_cross_out, hf_cross_out)
    max_abs_cross = (ref_cross_out - hf_cross_out).abs().max().item()
    print(f"[seamless_mha/cross] pcc={pcc_cross:.6f} max_abs_diff={max_abs_cross:.3e}")
    assert pcc_cross > 0.99, f"cross-attention PCC {pcc_cross} <= 0.99"

    # Exact match (both reference and HF are fp32, same op sequence).
    assert torch.allclose(
        ref_self_out, hf_self_out, atol=1e-5, rtol=1e-4
    ), f"self-attn diverged: max_abs={max_abs_self}"
    assert torch.allclose(
        ref_cross_out, hf_cross_out, atol=1e-5, rtol=1e-4
    ), f"cross-attn diverged: max_abs={max_abs_cross}"

    golden_path = GOLDEN_DIR / "seamless_mha.pt"
    torch.save(
        {
            "input": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "self_attention_mask": self_mask,
            "cross_attention_mask": cross_mask,
            "state_dict": state_dict,
            "output_self": ref_self_out,
            "output_cross": ref_cross_out,
            "config": {
                "batch": batch,
                "tgt_len": tgt_len,
                "src_len": src_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "dtype": "float32",
                "block": "seamless_mha",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2Attention",
            },
        },
        golden_path,
    )
    print(f"[seamless_mha] saved golden to {golden_path}")
    return min(pcc_self, pcc_cross)


def test_seamless_ffn_matches_hf() -> float:
    """Compare reference ``SeamlessM4Tv2FeedForwardNetwork`` forward against HuggingFace.

    Builds an HF ``SeamlessM4Tv2FeedForwardNetwork`` at the SeamlessM4T-v2-Large
    sizes (``hidden_size=1024``, ``ffn_dim=8192``, activation=``relu``,
    ``activation_dropout=0.0``), extracts its fc1/fc2 weights and biases, and
    verifies the standalone reference matches bit-for-bit.
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2FeedForwardNetwork

    torch.manual_seed(0)

    batch, seq_len, hidden = 1, 64, 1024
    ffn_dim = 8192

    # Use the canonical SeamlessM4T-v2 config (activation="relu", dropout=0.0).
    config = SeamlessM4Tv2Config()
    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert config.activation_function == "relu", f"unexpected activation {config.activation_function}"

    hf = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=ffn_dim)
    hf.eval()

    fc1_weight = hf.fc1.weight.detach().clone()
    fc1_bias = hf.fc1.bias.detach().clone()
    fc2_weight = hf.fc2.weight.detach().clone()
    fc2_bias = hf.fc2.bias.detach().clone()

    x = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    with torch.no_grad():
        hf_out = hf(x)
    ref_out = seamless_ffn_forward(
        x,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
        dropout_p=config.activation_dropout,
    )

    pcc = _pcc(ref_out, hf_out)
    max_abs = (ref_out - hf_out).abs().max().item()
    print(f"[seamless_ffn] pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")
    assert pcc > 0.99, f"seamless_ffn PCC {pcc} <= 0.99"
    # fp32 reference + fp32 HF -> identical op sequence -> exact match.
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"seamless_ffn diverged: max_abs={max_abs}"

    golden_path = GOLDEN_DIR / "seamless_ffn.pt"
    torch.save(
        {
            "input": x,
            "fc1_weight": fc1_weight,
            "fc1_bias": fc1_bias,
            "fc2_weight": fc2_weight,
            "fc2_bias": fc2_bias,
            "output": ref_out,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden,
                "ffn_dim": ffn_dim,
                "activation": "relu",
                "activation_dropout": config.activation_dropout,
                "dtype": "float32",
                "block": "seamless_ffn",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2FeedForwardNetwork",
            },
        },
        golden_path,
    )
    print(f"[seamless_ffn] saved golden to {golden_path}")
    return pcc


if __name__ == "__main__":
    pcc_ln = test_layernorm_matches_hf()
    print(f"\nFINAL PCC layernorm: {pcc_ln:.6f}")
    pcc_mha = test_seamless_mha_matches_hf()
    print(f"\nFINAL PCC seamless_mha: {pcc_mha:.6f}")
    pcc_ffn = test_seamless_ffn_matches_hf()
    print(f"\nFINAL PCC seamless_ffn: {pcc_ffn:.6f}")

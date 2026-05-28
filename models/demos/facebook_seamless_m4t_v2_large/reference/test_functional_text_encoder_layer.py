# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``text_encoder_layer_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2EncoderLayer`` module at shape ``[1, 64, 1024]``
(batch, seq_len, hidden) using the v2-Large config, then saves a golden
tensor for downstream TTNN PCC verification.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import text_encoder_layer_forward

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


def _ln_sd(ln) -> dict:
    return {
        "weight": ln.weight.detach().clone(),
        "bias": ln.bias.detach().clone(),
    }


def _linear_sd(lin) -> dict:
    sd = {"weight": lin.weight.detach().clone()}
    if lin.bias is not None:
        sd["bias"] = lin.bias.detach().clone()
    return sd


def _self_attn_sd(attn) -> dict:
    return {
        "q_proj": _linear_sd(attn.q_proj),
        "k_proj": _linear_sd(attn.k_proj),
        "v_proj": _linear_sd(attn.v_proj),
        "out_proj": _linear_sd(attn.out_proj),
    }


def _ffn_sd(ffn) -> dict:
    return {
        "fc1": _linear_sd(ffn.fc1),
        "fc2": _linear_sd(ffn.fc2),
    }


def _extract_text_encoder_layer_state_dict(layer) -> dict:
    """Pull all sub-block weights out of an HF SeamlessM4Tv2EncoderLayer."""
    return {
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _self_attn_sd(layer.self_attn),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def test_text_encoder_layer_matches_hf() -> float:
    """Compare reference text_encoder_layer forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config:
        - hidden_size = 1024
        - encoder_attention_heads = 16 (head_dim = 64)
        - encoder_ffn_dim = 8192
        - activation_function = 'relu'
        - layer_norm_eps = 1e-5
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2EncoderLayer

    torch.manual_seed(0)

    batch, seq_len, hidden = 1, 64, 1024
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert config.encoder_attention_heads == num_heads, f"unexpected num_heads {config.encoder_attention_heads}"
    assert config.encoder_ffn_dim == 8192, f"unexpected encoder_ffn_dim {config.encoder_ffn_dim}"
    assert config.activation_function == "relu", f"unexpected activation_function {config.activation_function}"

    eps = config.layer_norm_eps  # 1e-5

    hf = SeamlessM4Tv2EncoderLayer(config)
    hf.eval()
    # Be explicit: zero out all dropouts in case downstream configs change.
    hf.attn_dropout.p = 0.0
    hf.ffn_dropout.p = 0.0
    hf.self_attn.dropout = 0.0
    hf.ffn.dropout.p = 0.0

    state_dict = _extract_text_encoder_layer_state_dict(hf)

    hidden_states = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    # --- unmasked path ---
    with torch.no_grad():
        hf_out_tuple = hf(
            hidden_states=hidden_states,
            attention_mask=None,
            output_attentions=False,
        )
    hf_out = hf_out_tuple[0]
    ref_out = text_encoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        eps=eps,
        activation_function="relu",
    )
    pcc_unmasked = _pcc(ref_out, hf_out)
    max_abs_unmasked = (ref_out - hf_out).abs().max().item()
    print(f"[text_encoder_layer/unmasked] pcc={pcc_unmasked:.6f} max_abs_diff={max_abs_unmasked:.3e}")
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path: additive log-mask masking the last 8 src positions ---
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=torch.float32)
    attention_mask[:, :, :, -8:] = torch.finfo(torch.float32).min

    with torch.no_grad():
        hf_out_masked_tuple = hf(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
        )
    hf_out_masked = hf_out_masked_tuple[0]
    ref_out_masked = text_encoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=attention_mask,
        eps=eps,
        activation_function="relu",
    )
    pcc_masked = _pcc(ref_out_masked, hf_out_masked)
    max_abs_masked = (ref_out_masked - hf_out_masked).abs().max().item()
    print(f"[text_encoder_layer/masked]   pcc={pcc_masked:.6f} max_abs_diff={max_abs_masked:.3e}")
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(
        ref_out_masked, hf_out_masked, atol=1e-5, rtol=1e-4
    ), f"masked diverged: max_abs={max_abs_masked}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "text_encoder_layer.pt"
    torch.save(
        {
            "input": hidden_states,
            "attention_mask": attention_mask,
            "state_dict": state_dict,
            "output_unmasked": ref_out,
            "output_masked": ref_out_masked,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "encoder_ffn_dim": 8192,
                "activation_function": "relu",
                "eps": eps,
                "dtype": "float32",
                "block": "text_encoder_layer",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2EncoderLayer",
            },
        },
        golden_path,
    )
    print(f"[text_encoder_layer] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_text_encoder_layer_matches_hf()
    print(f"\nFINAL PCC text_encoder_layer: {pcc:.6f}")

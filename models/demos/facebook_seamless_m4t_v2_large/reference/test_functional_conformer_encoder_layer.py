# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``conformer_encoder_layer_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2ConformerEncoderLayer`` module at shape ``[1, 64, 1024]``
(batch, seq_len, hidden) using the v2-Large config, then saves a golden
tensor for downstream TTNN PCC verification.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import conformer_encoder_layer_forward

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


def _conv1d_sd(conv) -> dict:
    sd = {"weight": conv.weight.detach().clone()}
    if conv.bias is not None:
        sd["bias"] = conv.bias.detach().clone()
    return sd


def _ffn_sd(ffn) -> dict:
    return {
        "intermediate_dense": _linear_sd(ffn.intermediate_dense),
        "output_dense": _linear_sd(ffn.output_dense),
    }


def _self_attn_sd(attn) -> dict:
    return {
        "linear_q": _linear_sd(attn.linear_q),
        "linear_k": _linear_sd(attn.linear_k),
        "linear_v": _linear_sd(attn.linear_v),
        "linear_out": _linear_sd(attn.linear_out),
    }


def _conv_module_sd(conv) -> dict:
    return {
        "layer_norm": _ln_sd(conv.layer_norm),
        "pointwise_conv1": _conv1d_sd(conv.pointwise_conv1),
        "depthwise_conv": _conv1d_sd(conv.depthwise_conv),
        "depthwise_layer_norm": _ln_sd(conv.depthwise_layer_norm),
        "pointwise_conv2": _conv1d_sd(conv.pointwise_conv2),
    }


def _extract_encoder_layer_state_dict(layer) -> dict:
    """Pull all sub-block weights out of an HF SeamlessM4Tv2ConformerEncoderLayer."""
    return {
        "ffn1_layer_norm": _ln_sd(layer.ffn1_layer_norm),
        "ffn1": _ffn_sd(layer.ffn1),
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _self_attn_sd(layer.self_attn),
        "conv_module": _conv_module_sd(layer.conv_module),
        "ffn2_layer_norm": _ln_sd(layer.ffn2_layer_norm),
        "ffn2": _ffn_sd(layer.ffn2),
        "final_layer_norm": _ln_sd(layer.final_layer_norm),
    }


def test_conformer_encoder_layer_matches_hf() -> float:
    """Compare reference ConformerEncoderLayer forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config:
        - hidden_size = 1024
        - speech_encoder_attention_heads = 16 (head_dim = 64)
        - position_embeddings_type = 'relative_key'
        - left_max_position_embeddings = 64
        - right_max_position_embeddings = 8
        - conv_depthwise_kernel_size = 31
        - speech_encoder_hidden_act = 'swish'
        - layer_norm_eps = 1e-5
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ConformerEncoderLayer

    torch.manual_seed(0)

    batch, seq_len, hidden = 1, 64, 1024
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert (
        config.speech_encoder_attention_heads == num_heads
    ), f"unexpected num_heads {config.speech_encoder_attention_heads}"
    assert (
        config.position_embeddings_type == "relative_key"
    ), f"unexpected position_embeddings_type {config.position_embeddings_type}"
    assert (
        config.conv_depthwise_kernel_size == 31
    ), f"unexpected conv_depthwise_kernel_size {config.conv_depthwise_kernel_size}"
    assert (
        config.speech_encoder_hidden_act == "swish"
    ), f"unexpected speech_encoder_hidden_act {config.speech_encoder_hidden_act}"

    left_max = config.left_max_position_embeddings  # 64
    right_max = config.right_max_position_embeddings  # 8
    conv_kernel = config.conv_depthwise_kernel_size  # 31
    eps = config.layer_norm_eps  # 1e-5

    hf = SeamlessM4Tv2ConformerEncoderLayer(config)
    hf.eval()
    # speech_encoder_dropout / activation_dropout default to 0.0 in v2-Large,
    # but be explicit in case downstream configs change.
    hf.self_attn_dropout.p = 0.0
    hf.self_attn.dropout.p = 0.0
    hf.conv_module.dropout.p = 0.0
    hf.ffn1.intermediate_dropout.p = 0.0
    hf.ffn1.output_dropout.p = 0.0
    hf.ffn2.intermediate_dropout.p = 0.0
    hf.ffn2.output_dropout.p = 0.0

    state_dict = _extract_encoder_layer_state_dict(hf)
    distance_embedding_weight = hf.self_attn.distance_embedding.weight.detach().clone()

    hidden_states = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    # --- unmasked path ---
    with torch.no_grad():
        hf_out, _ = hf(
            hidden_states=hidden_states,
            attention_mask=None,
            output_attentions=False,
            conv_attention_mask=None,
        )
    ref_out = conformer_encoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        conv_kernel_size=conv_kernel,
        act_fn="swish",
        eps=eps,
        attention_mask=None,
        conv_attention_mask=None,
    )
    pcc_unmasked = _pcc(ref_out, hf_out)
    max_abs_unmasked = (ref_out - hf_out).abs().max().item()
    print(f"[conformer_encoder_layer/unmasked] pcc={pcc_unmasked:.6f} " f"max_abs_diff={max_abs_unmasked:.3e}")
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path: additive log-mask for self-attn + bool padding mask for conv ---
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=torch.float32)
    attention_mask[:, :, :, -8:] = torch.finfo(torch.float32).min
    conv_attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    conv_attention_mask[:, -8:] = False

    with torch.no_grad():
        hf_out_masked, _ = hf(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            conv_attention_mask=conv_attention_mask,
        )
    ref_out_masked = conformer_encoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        conv_kernel_size=conv_kernel,
        act_fn="swish",
        eps=eps,
        attention_mask=attention_mask,
        conv_attention_mask=conv_attention_mask,
    )
    pcc_masked = _pcc(ref_out_masked, hf_out_masked)
    max_abs_masked = (ref_out_masked - hf_out_masked).abs().max().item()
    print(f"[conformer_encoder_layer/masked]   pcc={pcc_masked:.6f} " f"max_abs_diff={max_abs_masked:.3e}")
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(
        ref_out_masked, hf_out_masked, atol=1e-5, rtol=1e-4
    ), f"masked diverged: max_abs={max_abs_masked}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "conformer_encoder_layer.pt"
    torch.save(
        {
            "input": hidden_states,
            "attention_mask": attention_mask,
            "conv_attention_mask": conv_attention_mask,
            "state_dict": state_dict,
            "distance_embedding_weight": distance_embedding_weight,
            "output_unmasked": ref_out,
            "output_masked": ref_out_masked,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "left_max_position_embeddings": left_max,
                "right_max_position_embeddings": right_max,
                "position_embeddings_type": "relative_key",
                "conv_kernel_size": conv_kernel,
                "act_fn": "swish",
                "eps": eps,
                "dtype": "float32",
                "block": "conformer_encoder_layer",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2ConformerEncoderLayer",
            },
        },
        golden_path,
    )
    print(f"[conformer_encoder_layer] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_conformer_encoder_layer_matches_hf()
    print(f"\nFINAL PCC conformer_encoder_layer: {pcc:.6f}")

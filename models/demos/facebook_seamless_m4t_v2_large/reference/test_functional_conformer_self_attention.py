# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``conformer_self_attention_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2ConformerSelfAttention`` module at the model's representative
shape ``[1, 128, 1024]`` with the ``relative_key`` positional bias enabled,
and saves a golden tensor for downstream TTNN PCC verification.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import conformer_self_attention_forward

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


def _extract_conformer_self_attn_state_dict(layer) -> dict:
    """Pull linear_q/k/v/out weight & bias tensors out of an HF module."""
    return {
        "linear_q": {
            "weight": layer.linear_q.weight.detach().clone(),
            "bias": layer.linear_q.bias.detach().clone(),
        },
        "linear_k": {
            "weight": layer.linear_k.weight.detach().clone(),
            "bias": layer.linear_k.bias.detach().clone(),
        },
        "linear_v": {
            "weight": layer.linear_v.weight.detach().clone(),
            "bias": layer.linear_v.bias.detach().clone(),
        },
        "linear_out": {
            "weight": layer.linear_out.weight.detach().clone(),
            "bias": layer.linear_out.bias.detach().clone(),
        },
    }


def test_conformer_self_attention_matches_hf() -> float:
    """Compare reference ConformerSelfAttention forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config: ``hidden_size=1024``,
    ``speech_encoder_attention_heads=16`` (head_dim=64),
    ``position_embeddings_type='relative_key'``,
    ``left_max_position_embeddings=64``, ``right_max_position_embeddings=8``.

    Verifies both the unmasked and additive log-mask paths.
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ConformerSelfAttention

    torch.manual_seed(0)

    batch, seq_len, hidden = 1, 128, 1024
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert (
        config.speech_encoder_attention_heads == num_heads
    ), f"unexpected num_heads {config.speech_encoder_attention_heads}"
    assert (
        config.position_embeddings_type == "relative_key"
    ), f"unexpected position_embeddings_type {config.position_embeddings_type}"

    left_max = config.left_max_position_embeddings  # 64
    right_max = config.right_max_position_embeddings  # 8

    hf = SeamlessM4Tv2ConformerSelfAttention(config, use_position_embeddings=True)
    hf.eval()
    # speech_encoder_dropout defaults to 0.0 in v2-Large, so eval mode is enough,
    # but be explicit and disable just in case downstream configs change.
    hf.dropout.p = 0.0

    state_dict = _extract_conformer_self_attn_state_dict(hf)
    distance_embedding_weight = hf.distance_embedding.weight.detach().clone()

    hidden_states = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    # --- unmasked path ---
    with torch.no_grad():
        hf_out, _ = hf(hidden_states=hidden_states, attention_mask=None)
    ref_out = conformer_self_attention_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        attention_mask=None,
    )
    pcc_unmasked = _pcc(ref_out, hf_out)
    max_abs_unmasked = (ref_out - hf_out).abs().max().item()
    print(f"[conformer_self_attention/unmasked] pcc={pcc_unmasked:.6f} " f"max_abs_diff={max_abs_unmasked:.3e}")
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path: additive log-mask covering [B, 1, T, T] ---
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=torch.float32)
    # Mask the last 16 keys (typical "right padding") to a large negative.
    attention_mask[:, :, :, -16:] = torch.finfo(torch.float32).min
    with torch.no_grad():
        hf_out_masked, _ = hf(hidden_states=hidden_states, attention_mask=attention_mask)
    ref_out_masked = conformer_self_attention_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        attention_mask=attention_mask,
    )
    pcc_masked = _pcc(ref_out_masked, hf_out_masked)
    max_abs_masked = (ref_out_masked - hf_out_masked).abs().max().item()
    print(f"[conformer_self_attention/masked]   pcc={pcc_masked:.6f} " f"max_abs_diff={max_abs_masked:.3e}")
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(
        ref_out_masked, hf_out_masked, atol=1e-5, rtol=1e-4
    ), f"masked diverged: max_abs={max_abs_masked}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "conformer_self_attention.pt"
    torch.save(
        {
            "input": hidden_states,
            "attention_mask": attention_mask,
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
                "dtype": "float32",
                "block": "conformer_self_attention",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2ConformerSelfAttention",
            },
        },
        golden_path,
    )
    print(f"[conformer_self_attention] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_conformer_self_attention_matches_hf()
    print(f"\nFINAL PCC conformer_self_attention: {pcc:.6f}")

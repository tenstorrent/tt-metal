# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``text_decoder_layer_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2DecoderLayer`` module using the v2-Large config at
decoder shape ``[1, 32, 1024]`` (batch, tgt_len, hidden) with cross-attention
to encoder hidden ``[1, 64, 1024]`` (batch, src_len, hidden). Saves a golden
tensor for downstream TTNN PCC verification.

Three paths are verified:
    1. Self-attention only (no encoder_hidden_states) - exercises the early-
       return branch.
    2. Self-attention + cross-attention, both unmasked.
    3. Self-attention with a triangular causal mask + cross-attention with a
       padding log-mask on the last 8 encoder positions.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import text_decoder_layer_forward

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


def _attn_sd(attn) -> dict:
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


def _extract_text_decoder_layer_state_dict(layer) -> dict:
    """Pull all sub-block weights out of an HF SeamlessM4Tv2DecoderLayer."""
    return {
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _attn_sd(layer.self_attn),
        "cross_attention_layer_norm": _ln_sd(layer.cross_attention_layer_norm),
        "cross_attention": _attn_sd(layer.cross_attention),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def test_text_decoder_layer_matches_hf() -> float:
    """Compare reference text_decoder_layer forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config:
        - hidden_size = 1024
        - decoder_attention_heads = 16 (head_dim = 64)
        - decoder_ffn_dim = 8192
        - activation_function = 'relu'
        - layer_norm_eps = 1e-5
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2DecoderLayer

    torch.manual_seed(0)

    batch, tgt_len, src_len, hidden = 1, 32, 64, 1024
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert config.decoder_attention_heads == num_heads, f"unexpected num_heads {config.decoder_attention_heads}"
    assert config.decoder_ffn_dim == 8192, f"unexpected decoder_ffn_dim {config.decoder_ffn_dim}"
    assert config.activation_function == "relu", f"unexpected activation_function {config.activation_function}"

    eps = config.layer_norm_eps  # 1e-5

    hf = SeamlessM4Tv2DecoderLayer(config, layer_idx=0)
    hf.eval()
    # Be explicit: zero out all dropouts in case downstream configs change.
    hf.attn_dropout.p = 0.0
    hf.ffn_dropout.p = 0.0
    hf.self_attn.dropout = 0.0
    hf.cross_attention.dropout = 0.0
    hf.ffn.dropout.p = 0.0

    state_dict = _extract_text_decoder_layer_state_dict(hf)

    hidden_states = torch.randn(batch, tgt_len, hidden, dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch, src_len, hidden, dtype=torch.float32)

    # --- self-attn only (no encoder_hidden_states) ---
    with torch.no_grad():
        hf_out_tuple = hf(
            hidden_states=hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            use_cache=False,
            past_key_values=None,
        )
    hf_self_only = hf_out_tuple[0]
    ref_self_only = text_decoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=None,
        self_attention_mask=None,
        encoder_attention_mask=None,
        eps=eps,
        activation_function="relu",
    )
    pcc_self_only = _pcc(ref_self_only, hf_self_only)
    max_abs_self_only = (ref_self_only - hf_self_only).abs().max().item()
    print(f"[text_decoder_layer/self_only]    pcc={pcc_self_only:.6f} max_abs_diff={max_abs_self_only:.3e}")
    assert pcc_self_only > 0.99, f"self_only PCC {pcc_self_only} <= 0.99"
    assert torch.allclose(
        ref_self_only, hf_self_only, atol=1e-5, rtol=1e-4
    ), f"self_only diverged: max_abs={max_abs_self_only}"

    # --- self-attn + cross-attn, unmasked ---
    with torch.no_grad():
        hf_out_tuple = hf(
            hidden_states=hidden_states,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            output_attentions=False,
            use_cache=False,
            past_key_values=None,
        )
    hf_unmasked = hf_out_tuple[0]
    ref_unmasked = text_decoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=encoder_hidden_states,
        self_attention_mask=None,
        encoder_attention_mask=None,
        eps=eps,
        activation_function="relu",
    )
    pcc_unmasked = _pcc(ref_unmasked, hf_unmasked)
    max_abs_unmasked = (ref_unmasked - hf_unmasked).abs().max().item()
    print(f"[text_decoder_layer/unmasked]     pcc={pcc_unmasked:.6f} max_abs_diff={max_abs_unmasked:.3e}")
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(
        ref_unmasked, hf_unmasked, atol=1e-5, rtol=1e-4
    ), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path: triangular causal self-attn + cross-attn padding mask ---
    # Causal self-attention mask: lower-triangular zeros, upper-triangular -inf.
    neg_inf = torch.finfo(torch.float32).min
    causal_mask = torch.full((tgt_len, tgt_len), neg_inf, dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    self_attention_mask = causal_mask.view(1, 1, tgt_len, tgt_len).expand(batch, 1, tgt_len, tgt_len).contiguous()

    # Cross-attention mask: mask the last 8 encoder positions.
    encoder_attention_mask = torch.zeros(batch, 1, tgt_len, src_len, dtype=torch.float32)
    encoder_attention_mask[:, :, :, -8:] = neg_inf

    with torch.no_grad():
        hf_out_tuple = hf(
            hidden_states=hidden_states,
            attention_mask=self_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=False,
            use_cache=False,
            past_key_values=None,
        )
    hf_masked = hf_out_tuple[0]
    ref_masked = text_decoder_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=encoder_hidden_states,
        self_attention_mask=self_attention_mask,
        encoder_attention_mask=encoder_attention_mask,
        eps=eps,
        activation_function="relu",
    )
    pcc_masked = _pcc(ref_masked, hf_masked)
    max_abs_masked = (ref_masked - hf_masked).abs().max().item()
    print(f"[text_decoder_layer/masked]       pcc={pcc_masked:.6f} max_abs_diff={max_abs_masked:.3e}")
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(ref_masked, hf_masked, atol=1e-5, rtol=1e-4), f"masked diverged: max_abs={max_abs_masked}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "text_decoder_layer.pt"
    torch.save(
        {
            "input": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "self_attention_mask": self_attention_mask,
            "encoder_attention_mask": encoder_attention_mask,
            "state_dict": state_dict,
            "output_self_only": ref_self_only,
            "output_unmasked": ref_unmasked,
            "output_masked": ref_masked,
            "config": {
                "batch": batch,
                "tgt_len": tgt_len,
                "src_len": src_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "decoder_ffn_dim": 8192,
                "activation_function": "relu",
                "eps": eps,
                "dtype": "float32",
                "block": "text_decoder_layer",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2DecoderLayer",
            },
        },
        golden_path,
    )
    print(f"[text_decoder_layer] saved golden to {golden_path}")
    return min(pcc_self_only, pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_text_decoder_layer_matches_hf()
    print(f"\nFINAL PCC text_decoder_layer: {pcc:.6f}")

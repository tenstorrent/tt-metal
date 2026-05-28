# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``text_decoder_forward``.

Compares the standalone reference implementation of the full
SeamlessM4T-v2 NLLB-style text decoder against the HuggingFace
``SeamlessM4Tv2Decoder`` module. To keep verification fast, the HF
config is overridden to ``decoder_layers=2`` -- the per-layer block is
already separately verified by
:mod:`test_functional_text_decoder_layer`, and the role of this test is
to exercise the embedding + position + mask plumbing + final LayerNorm
stack around the existing layers.

Two paths are verified at decoder shape ``[1, 8]`` against encoder
hidden ``[1, 16, 1024]``:

    1. Unmasked: ``attention_mask=None`` (HF still adds the triangular
       causal mask internally), no encoder padding.
    2. Masked: 2D decoder padding mask + 2D encoder padding mask.

PCC > 0.99 is required (should be effectively 1.0 since every sub-block
is already verified).
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import text_decoder_forward

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
    return {
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _attn_sd(layer.self_attn),
        "cross_attention_layer_norm": _ln_sd(layer.cross_attention_layer_norm),
        "cross_attention": _attn_sd(layer.cross_attention),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def _extract_text_decoder_state_dict(decoder) -> dict:
    """Pull all weights out of an HF SeamlessM4Tv2Decoder into the nested form
    consumed by :func:`text_decoder_forward`."""
    return {
        "embed_tokens": {"weight": decoder.embed_tokens.weight.detach().clone()},
        "embed_positions_weights": decoder.embed_positions.weights.detach().clone(),
        "layers": [_extract_text_decoder_layer_state_dict(layer) for layer in decoder.layers],
        "layer_norm": _ln_sd(decoder.layer_norm),
    }


def test_text_decoder_matches_hf() -> float:
    """Compare reference ``text_decoder_forward`` against HuggingFace.

    Uses SeamlessM4T-v2-Large config but with ``decoder_layers=2`` for
    speed -- the per-layer block is separately verified.
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2Decoder

    torch.manual_seed(0)

    batch, tgt_len, src_len, hidden = 1, 8, 16, 1024
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    # Reduce the decoder to 2 layers for verification speed. All other
    # v2-Large defaults (hidden_size, attention_heads, ffn_dim, eps, ...) stay.
    config.decoder_layers = 2

    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert config.decoder_attention_heads == num_heads, f"unexpected num_heads {config.decoder_attention_heads}"
    assert config.decoder_ffn_dim == 8192, f"unexpected decoder_ffn_dim {config.decoder_ffn_dim}"
    assert config.activation_function == "relu", f"unexpected activation_function {config.activation_function}"
    assert config.pad_token_id == 0, f"unexpected pad_token_id {config.pad_token_id}"
    assert config.scale_embedding is True, f"unexpected scale_embedding {config.scale_embedding}"

    eps = config.layer_norm_eps  # 1e-5
    padding_idx = config.pad_token_id  # 0

    hf = SeamlessM4Tv2Decoder(config)
    hf.eval()
    # Be explicit: zero out all dropouts in case downstream configs change.
    hf.dropout = 0.0
    for layer in hf.layers:
        layer.attn_dropout.p = 0.0
        layer.ffn_dropout.p = 0.0
        layer.self_attn.dropout = 0.0
        layer.cross_attention.dropout = 0.0
        layer.ffn.dropout.p = 0.0

    state_dict = _extract_text_decoder_state_dict(hf)

    # Decoder input tokens: avoid 0 (padding_idx) at positions we want active.
    # Keep them well within vocab_size (256102).
    decoder_input_ids = torch.tensor(
        [[2, 17, 42, 101, 7, 88, 31, 5]],
        dtype=torch.long,
    )
    assert decoder_input_ids.shape == (batch, tgt_len)

    encoder_hidden_states = torch.randn(batch, src_len, hidden, dtype=torch.float32)

    # --- unmasked path (HF still adds triangular causal mask internally) ---
    with torch.no_grad():
        hf_out_obj = hf(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    hf_unmasked = hf_out_obj.last_hidden_state

    ref_unmasked = text_decoder_forward(
        decoder_input_ids,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        eps=eps,
        activation_function="relu",
        padding_idx=padding_idx,
    )
    pcc_unmasked = _pcc(ref_unmasked, hf_unmasked)
    max_abs_unmasked = (ref_unmasked - hf_unmasked).abs().max().item()
    print(f"[text_decoder/unmasked]  pcc={pcc_unmasked:.6f} max_abs_diff={max_abs_unmasked:.3e}")
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(
        ref_unmasked, hf_unmasked, atol=1e-5, rtol=1e-4
    ), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path: 2D decoder padding mask + 2D encoder padding mask ---
    # Decoder padding mask: mask the last 2 decoder positions (pad them out).
    decoder_attention_mask = torch.ones(batch, tgt_len, dtype=torch.long)
    decoder_attention_mask[:, -2:] = 0

    # Encoder padding mask: mask the last 4 encoder positions.
    encoder_padding_mask = torch.ones(batch, src_len, dtype=torch.long)
    encoder_padding_mask[:, -4:] = 0

    with torch.no_grad():
        hf_out_obj = hf(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_padding_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    hf_masked = hf_out_obj.last_hidden_state

    ref_masked = text_decoder_forward(
        decoder_input_ids,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=decoder_attention_mask,
        encoder_attention_mask=encoder_padding_mask,
        eps=eps,
        activation_function="relu",
        padding_idx=padding_idx,
    )
    pcc_masked = _pcc(ref_masked, hf_masked)
    max_abs_masked = (ref_masked - hf_masked).abs().max().item()
    print(f"[text_decoder/masked]    pcc={pcc_masked:.6f} max_abs_diff={max_abs_masked:.3e}")
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(ref_masked, hf_masked, atol=1e-5, rtol=1e-4), f"masked diverged: max_abs={max_abs_masked}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "text_decoder.pt"
    torch.save(
        {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_attention_mask": encoder_padding_mask,
            "state_dict": state_dict,
            "output_unmasked": ref_unmasked,
            "output_masked": ref_masked,
            "config": {
                "batch": batch,
                "tgt_len": tgt_len,
                "src_len": src_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "decoder_layers": config.decoder_layers,
                "decoder_ffn_dim": 8192,
                "activation_function": "relu",
                "eps": eps,
                "padding_idx": padding_idx,
                "scale_embedding": True,
                "dtype": "float32",
                "block": "text_decoder",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2Decoder",
                "note": (
                    "decoder_layers reduced to 2 from full 24 for fast verification; "
                    "per-layer block verified separately via text_decoder_layer."
                ),
            },
        },
        golden_path,
    )
    print(f"[text_decoder] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_text_decoder_matches_hf()
    print(f"\nFINAL PCC text_decoder: {pcc:.6f}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``seamless_m4t_v2_forward`` (top-level T2TT composition).

Compares the standalone reference implementation of the top-level
SeamlessM4T-v2 Text-to-Text (T2TT) forward against the HuggingFace
``SeamlessM4Tv2ForTextToText`` module. To keep verification fast (and the
golden tensor commit-able), the HF config is overridden to
``encoder_layers=2, decoder_layers=2`` -- per-layer blocks are already
separately verified, and the role of this top-level test is to exercise
the composition (encoder -> decoder -> lm_head) plumbing.

Scope:
    SeamlessM4T-v2 has 5 sub-models. The full S2ST pipeline requires
    autoregressive generation + NAR T2U upsampling + vocoder, which sits
    above any deterministic forward pass. This top-level reference
    verifies the *T2TT* deterministic forward pass
    (``text_encoder -> text_decoder -> lm_head``) -- the simplest
    end-to-end composition that exercises 2 of the 5 sub-models without
    requiring generation. The other sub-blocks are independently verified.

Two paths are exercised at ``input_ids [1, 8]`` -> ``decoder_input_ids [1, 4]``:

    1. Unmasked: ``attention_mask=None`` (HF still adds the triangular
       causal mask internally on the decoder), no encoder/decoder padding.
    2. Masked: 2D encoder padding mask + 2D decoder padding mask.

PCC > 0.99 is required (should be effectively 1.0 since every sub-block
is already verified).
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import seamless_m4t_v2_forward

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


def _extract_text_encoder_layer_state_dict(layer) -> dict:
    return {
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _attn_sd(layer.self_attn),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
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


def _extract_text_encoder_state_dict(encoder) -> dict:
    return {
        "embed_tokens": {"weight": encoder.embed_tokens.weight.detach().clone()},
        "embed_positions_weights": encoder.embed_positions.weights.detach().clone(),
        "layers": [_extract_text_encoder_layer_state_dict(layer) for layer in encoder.layers],
        "final_layer_norm": _ln_sd(encoder.layer_norm),
    }


def _extract_text_decoder_state_dict(decoder) -> dict:
    return {
        "embed_tokens": {"weight": decoder.embed_tokens.weight.detach().clone()},
        "embed_positions_weights": decoder.embed_positions.weights.detach().clone(),
        "layers": [_extract_text_decoder_layer_state_dict(layer) for layer in decoder.layers],
        "layer_norm": _ln_sd(decoder.layer_norm),
    }


def _extract_t2tt_state_dict(model) -> dict:
    """Pull all weights out of an HF SeamlessM4Tv2ForTextToText into the nested
    form consumed by :func:`seamless_m4t_v2_forward`."""
    return {
        "text_encoder": _extract_text_encoder_state_dict(model.text_encoder),
        "text_decoder": _extract_text_decoder_state_dict(model.text_decoder),
        "lm_head": _linear_sd(model.lm_head),
    }


def test_seamless_m4t_v2_t2tt_matches_hf() -> float:
    """Compare reference ``seamless_m4t_v2_forward`` (T2TT path) against HuggingFace.

    Uses SeamlessM4T-v2-Large config but with ``encoder_layers=2`` and
    ``decoder_layers=2`` for speed -- the per-layer blocks are separately
    verified.
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ForTextToText

    torch.manual_seed(0)

    batch, src_len, tgt_len, hidden = 1, 8, 4, 1024
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    # Reduce both encoder + decoder to 2 layers for verification speed. All
    # other v2-Large defaults stay the same.
    config.encoder_layers = 2
    config.decoder_layers = 2

    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert config.encoder_attention_heads == num_heads, f"unexpected encoder num_heads {config.encoder_attention_heads}"
    assert config.decoder_attention_heads == num_heads, f"unexpected decoder num_heads {config.decoder_attention_heads}"
    assert config.encoder_ffn_dim == 8192, f"unexpected encoder_ffn_dim {config.encoder_ffn_dim}"
    assert config.decoder_ffn_dim == 8192, f"unexpected decoder_ffn_dim {config.decoder_ffn_dim}"
    assert config.activation_function == "relu", f"unexpected activation_function {config.activation_function}"
    assert config.pad_token_id == 0, f"unexpected pad_token_id {config.pad_token_id}"
    assert config.scale_embedding is True, f"unexpected scale_embedding {config.scale_embedding}"

    eps = config.layer_norm_eps  # 1e-5
    # NOTE: SeamlessM4Tv2Config() default constructor sets pad_token_id=0
    # (NOT 1 as in the real "facebook/seamless-m4t-v2-large" checkpoint
    # which uses the NLLB convention). The encoder/decoder both pick up
    # padding_idx from config.pad_token_id, so we use that here to stay
    # consistent with the constructed HF module's padding_idx.
    encoder_padding_idx = config.pad_token_id  # 0 with default config
    decoder_padding_idx = config.pad_token_id  # 0 with default config

    hf = SeamlessM4Tv2ForTextToText(config)
    hf.eval()
    # Zero out every dropout so a behavioural change in dropout-during-eval
    # wouldn't surprise us.
    hf.text_encoder.dropout = 0.0
    for layer in hf.text_encoder.layers:
        layer.attn_dropout.p = 0.0
        layer.ffn_dropout.p = 0.0
        layer.self_attn.dropout = 0.0
        layer.ffn.dropout.p = 0.0
    hf.text_decoder.dropout = 0.0
    for layer in hf.text_decoder.layers:
        layer.attn_dropout.p = 0.0
        layer.ffn_dropout.p = 0.0
        layer.self_attn.dropout = 0.0
        layer.cross_attention.dropout = 0.0
        layer.ffn.dropout.p = 0.0

    state_dict = _extract_t2tt_state_dict(hf)

    # With config.pad_token_id=0 (default), avoid 0 in active positions for
    # both encoder + decoder so we don't accidentally mask out non-padding
    # tokens via the padding-aware sinusoidal positional embedding.
    # Keep all ids well within vocab_size (256102).
    input_ids = torch.tensor([[256047, 4, 17, 42, 101, 7, 88, 31]], dtype=torch.long)
    decoder_input_ids = torch.tensor([[3, 17, 42, 101]], dtype=torch.long)
    assert input_ids.shape == (batch, src_len)
    assert decoder_input_ids.shape == (batch, tgt_len)

    # --- Unmasked path (HF still adds triangular causal mask internally) ---
    with torch.no_grad():
        hf_out_obj = hf(
            input_ids=input_ids,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    hf_unmasked_logits = hf_out_obj.logits
    hf_unmasked_enc = hf_out_obj.encoder_last_hidden_state

    ref_unmasked = seamless_m4t_v2_forward(
        input_ids,
        decoder_input_ids,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        decoder_attention_mask=None,
        eps=eps,
        activation_function="relu",
        encoder_padding_idx=encoder_padding_idx,
        decoder_padding_idx=decoder_padding_idx,
    )
    pcc_unmasked_logits = _pcc(ref_unmasked["logits"], hf_unmasked_logits)
    pcc_unmasked_enc = _pcc(ref_unmasked["encoder_last_hidden_state"], hf_unmasked_enc)
    max_abs_unmasked = (ref_unmasked["logits"] - hf_unmasked_logits).abs().max().item()
    print(
        f"[seamless_m4t_v2/unmasked] pcc_logits={pcc_unmasked_logits:.6f} "
        f"pcc_enc={pcc_unmasked_enc:.6f} max_abs_diff_logits={max_abs_unmasked:.3e}"
    )
    assert pcc_unmasked_logits > 0.99, f"unmasked logits PCC {pcc_unmasked_logits} <= 0.99"
    assert pcc_unmasked_enc > 0.99, f"unmasked encoder PCC {pcc_unmasked_enc} <= 0.99"
    assert torch.allclose(
        ref_unmasked["logits"], hf_unmasked_logits, atol=1e-4, rtol=1e-4
    ), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- Masked path: 2D encoder padding mask + 2D decoder padding mask ---
    # Encoder: mask out the last 3 source positions.
    encoder_attention_mask = torch.ones(batch, src_len, dtype=torch.long)
    encoder_attention_mask[:, -3:] = 0

    # Decoder: mask out the last decoder position.
    decoder_attention_mask = torch.ones(batch, tgt_len, dtype=torch.long)
    decoder_attention_mask[:, -1:] = 0

    with torch.no_grad():
        hf_out_obj = hf(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    hf_masked_logits = hf_out_obj.logits
    hf_masked_enc = hf_out_obj.encoder_last_hidden_state

    ref_masked = seamless_m4t_v2_forward(
        input_ids,
        decoder_input_ids,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=encoder_attention_mask,
        decoder_attention_mask=decoder_attention_mask,
        eps=eps,
        activation_function="relu",
        encoder_padding_idx=encoder_padding_idx,
        decoder_padding_idx=decoder_padding_idx,
    )
    pcc_masked_logits = _pcc(ref_masked["logits"], hf_masked_logits)
    pcc_masked_enc = _pcc(ref_masked["encoder_last_hidden_state"], hf_masked_enc)
    max_abs_masked = (ref_masked["logits"] - hf_masked_logits).abs().max().item()
    print(
        f"[seamless_m4t_v2/masked]   pcc_logits={pcc_masked_logits:.6f} "
        f"pcc_enc={pcc_masked_enc:.6f} max_abs_diff_logits={max_abs_masked:.3e}"
    )
    assert pcc_masked_logits > 0.99, f"masked logits PCC {pcc_masked_logits} <= 0.99"
    assert pcc_masked_enc > 0.99, f"masked encoder PCC {pcc_masked_enc} <= 0.99"
    assert torch.allclose(
        ref_masked["logits"], hf_masked_logits, atol=1e-4, rtol=1e-4
    ), f"masked diverged: max_abs={max_abs_masked}"

    # Save a tiny golden tensor (without the full state_dict; the
    # encoder/decoder weights for a 2-layer slice are still ~hundreds of MB
    # at hidden=1024, ffn=8192. Save only the inputs + outputs.).
    golden_path = GOLDEN_DIR / "seamless_m4t_v2.pt"
    torch.save(
        {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "logits_unmasked": ref_unmasked["logits"],
            "logits_masked": ref_masked["logits"],
            "encoder_last_hidden_state_unmasked": ref_unmasked["encoder_last_hidden_state"],
            "encoder_last_hidden_state_masked": ref_masked["encoder_last_hidden_state"],
            "config": {
                "batch": batch,
                "src_len": src_len,
                "tgt_len": tgt_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "encoder_layers": config.encoder_layers,
                "decoder_layers": config.decoder_layers,
                "ffn_dim": 8192,
                "activation_function": "relu",
                "eps": eps,
                "encoder_padding_idx": encoder_padding_idx,
                "decoder_padding_idx": decoder_padding_idx,
                "scale_embedding": True,
                "dtype": "float32",
                "block": "seamless_m4t_v2",
                "scope": "T2TT (text_encoder -> text_decoder -> lm_head)",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2ForTextToText",
                "note": (
                    "encoder_layers/decoder_layers reduced to 2 from full 24 for fast "
                    "verification; sub-blocks verified separately. State_dict NOT saved "
                    "(weights too large + identical to per-sub-block goldens)."
                ),
            },
        },
        golden_path,
    )
    print(f"[seamless_m4t_v2] saved golden to {golden_path}")
    return min(pcc_unmasked_logits, pcc_unmasked_enc, pcc_masked_logits, pcc_masked_enc)


if __name__ == "__main__":
    pcc = test_seamless_m4t_v2_t2tt_matches_hf()
    print(f"\nFINAL PCC seamless_m4t_v2 (T2TT): {pcc:.6f}")

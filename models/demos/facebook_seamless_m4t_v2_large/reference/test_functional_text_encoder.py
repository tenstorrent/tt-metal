# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``text_encoder_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2Encoder`` module at small scale (input_ids ``[1, 16]`` with
``num_hidden_layers=2`` to keep memory/golden small), then saves a golden
tensor for downstream TTNN PCC verification.

The full v2-Large text encoder has 24 layers; we override
``encoder_layers=2`` on the HF config (and leave everything else at v2-Large
defaults) so the saved golden file stays small while still exercising the
multi-layer + final LayerNorm path. Per-layer correctness against HF is
already verified in ``test_functional_text_encoder_layer.py``.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import (
    build_sinusoidal_positional_embedding_weights,
    text_encoder_forward,
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
    return {
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _self_attn_sd(layer.self_attn),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def _extract_text_encoder_state_dict(encoder) -> dict:
    """Pull all sub-block weights out of an HF SeamlessM4Tv2Encoder."""
    return {
        "embed_tokens": {"weight": encoder.embed_tokens.weight.detach().clone()},
        # HF's SinusoidalPositionalEmbedding stores its precomputed table on
        # `self.weights` (registered as a buffer). Clone it so the golden /
        # state_dict are independent of the HF module's lifetime.
        "embed_positions_weights": encoder.embed_positions.weights.detach().clone(),
        "layers": [_extract_text_encoder_layer_state_dict(layer) for layer in encoder.layers],
        "final_layer_norm": _ln_sd(encoder.layer_norm),
    }


def test_text_encoder_matches_hf() -> float:
    """Compare reference text_encoder forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config but with ``encoder_layers=2`` (vs the
    real model's 24) to keep memory + golden file size reasonable. Per-layer
    fidelity is already covered by ``test_functional_text_encoder_layer.py``;
    this test exercises the embed + pos + stacked-layers + final-LN
    composition.
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2Encoder

    torch.manual_seed(0)

    batch, seq_len = 1, 16
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    # Sanity-check expected v2-Large defaults.
    assert config.hidden_size == 1024, f"unexpected hidden_size {config.hidden_size}"
    assert (
        config.encoder_attention_heads == num_heads
    ), f"unexpected encoder_attention_heads {config.encoder_attention_heads}"
    assert config.encoder_ffn_dim == 8192, f"unexpected encoder_ffn_dim {config.encoder_ffn_dim}"
    assert config.activation_function == "relu", f"unexpected activation_function {config.activation_function}"
    assert config.scale_embedding, "expected scale_embedding=True"
    # Note: SeamlessM4Tv2Config() default constructor sets pad_token_id=0;
    # the real "facebook/seamless-m4t-v2-large" checkpoint config uses
    # pad_token_id=1 (NLLB convention). We test against whatever the config
    # holds so the per-encoder padding_idx is consistent with HF.

    # Override depth to keep golden small. 2 layers is enough to exercise
    # the multi-layer composition path while staying well under 100MB.
    config.encoder_layers = 2

    eps = config.layer_norm_eps  # 1e-5 (also nn.LayerNorm default)
    padding_idx = config.pad_token_id  # 1
    hidden_size = config.hidden_size  # 1024

    encoder = SeamlessM4Tv2Encoder(config, is_t2u_encoder=False)
    encoder.eval()
    # Belt-and-braces: zero any layer-level dropouts. They're already off in
    # eval but this protects against accidental config drift.
    encoder.dropout = 0.0
    for layer in encoder.layers:
        layer.attn_dropout.p = 0.0
        layer.ffn_dropout.p = 0.0
        layer.self_attn.dropout = 0.0
        layer.ffn.dropout.p = 0.0

    state_dict = _extract_text_encoder_state_dict(encoder)

    # Pick input ids that mix padding (token id == padding_idx) and
    # non-padding tokens, so that both the scaled-embedding lookup and the
    # padding-aware sinusoidal-position path get exercised. Keep ids well
    # below vocab_size (~256k).
    # We avoid 0 inside the body and place padding tokens (== padding_idx)
    # only at the tail, mirroring how NLLB-style padding actually appears.
    non_pad = 7 if padding_idx == 0 else 0
    pad = padding_idx
    base = [5, 17, 42, non_pad, 88, 200, 4, 9, 12, 31, 21, 7, 64, 128, 256, pad]
    # Make sure last few entries are padding to exercise the padding mask path.
    base[-3:] = [pad, pad, pad]
    input_ids = torch.tensor([base], dtype=torch.long)
    assert input_ids.shape == (batch, seq_len)

    # --- unmasked path: no attention mask (full self-attention). ---
    with torch.no_grad():
        hf_out = encoder(
            input_ids=input_ids,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    ref_out = text_encoder_forward(
        input_ids,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        eps=eps,
        activation_function="relu",
        padding_idx=padding_idx,
    )
    pcc_unmasked = _pcc(ref_out, hf_out)
    max_abs_unmasked = (ref_out - hf_out).abs().max().item()
    print(f"[text_encoder/unmasked] pcc={pcc_unmasked:.6f} max_abs_diff={max_abs_unmasked:.3e}")
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path: 2-D HF-style attention mask (1=keep, 0=pad). ---
    # Mask out the trailing 4 positions so the encoder must avoid attending
    # to them. We pass a 2-D mask in, just like a real HF caller would, and
    # expand it ourselves into the 4-D additive form that text_encoder_forward
    # expects internally.
    attention_mask_2d = torch.ones(batch, seq_len, dtype=torch.long)
    attention_mask_2d[:, -4:] = 0

    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    # HF expands using inputs_embeds.dtype (float32 here).
    attention_mask_4d = _prepare_4d_attention_mask(attention_mask_2d, torch.float32)

    with torch.no_grad():
        hf_out_masked = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask_2d,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    ref_out_masked = text_encoder_forward(
        input_ids,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=attention_mask_4d,
        eps=eps,
        activation_function="relu",
        padding_idx=padding_idx,
    )
    pcc_masked = _pcc(ref_out_masked, hf_out_masked)
    max_abs_masked = (ref_out_masked - hf_out_masked).abs().max().item()
    print(f"[text_encoder/masked]   pcc={pcc_masked:.6f} max_abs_diff={max_abs_masked:.3e}")
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(
        ref_out_masked, hf_out_masked, atol=1e-5, rtol=1e-4
    ), f"masked diverged: max_abs={max_abs_masked}"

    # Sanity: build_sinusoidal_positional_embedding_weights should reproduce
    # the table HF instantiates on the encoder. This guards against drift
    # between our standalone builder and HF's get_embedding.
    rebuilt = build_sinusoidal_positional_embedding_weights(
        encoder.embed_positions.weights.shape[0],
        hidden_size,
        padding_idx=padding_idx,
    )
    assert torch.allclose(
        rebuilt, encoder.embed_positions.weights, atol=1e-6
    ), "build_sinusoidal_positional_embedding_weights drifted from HF table"

    # Save golden tensor for downstream TTNN PCC checks. Note we save the
    # 2-layer state_dict (NOT the 24-layer one) for the same memory reason
    # as the test itself -- downstream consumers should match this scale.
    #
    # CRITICAL: we do NOT serialize the full embed_tokens table
    # (vocab_size=256102 x 1024 = ~1 GB at fp32). Downstream TTNN consumers
    # should load the embed table from the HF checkpoint directly. Instead
    # we save (a) the small subset of rows actually indexed by input_ids
    # and (b) the embed_scale used, so the test_functional consumer can
    # still reproduce the gather without the full table.
    used_ids = torch.unique(input_ids).tolist()
    embed_rows_used = {
        int(tok_id): state_dict["embed_tokens"]["weight"][tok_id].detach().clone() for tok_id in used_ids
    }
    golden_state_dict = {
        # NOTE: 'embed_tokens' key intentionally omitted; reload from HF.
        "embed_tokens_rows_used": embed_rows_used,
        "embed_positions_weights": state_dict["embed_positions_weights"],
        "layers": state_dict["layers"],
        "final_layer_norm": state_dict["final_layer_norm"],
    }
    golden_path = GOLDEN_DIR / "text_encoder.pt"
    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask_2d": attention_mask_2d,
            "attention_mask_4d": attention_mask_4d,
            "state_dict": golden_state_dict,
            "output_unmasked": ref_out,
            "output_masked": ref_out_masked,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden_size,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "encoder_ffn_dim": config.encoder_ffn_dim,
                "encoder_layers_in_golden": config.encoder_layers,  # 2 (small)
                "encoder_layers_full_model": 24,
                "activation_function": "relu",
                "eps": eps,
                "padding_idx": padding_idx,
                "embed_scale": float(hidden_size) ** 0.5,
                "max_position_embeddings": config.max_position_embeddings,
                "dtype": "float32",
                "block": "text_encoder",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2Encoder",
                "notes": (
                    "Reduced to encoder_layers=2 to keep golden file small "
                    "(real model has 24). Per-layer correctness covered "
                    "separately by test_functional_text_encoder_layer.py."
                ),
            },
        },
        golden_path,
    )
    print(f"[text_encoder] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_text_encoder_matches_hf()
    print(f"\nFINAL PCC text_encoder: {pcc:.6f}")

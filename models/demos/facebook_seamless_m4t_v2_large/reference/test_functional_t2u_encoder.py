# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``t2u_encoder_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2Encoder`` module instantiated with ``is_t2u_encoder=True`` at
small scale (input ``[1, 16, 1024]`` with ``t2u_encoder_layers=2`` to keep
memory/golden small), then saves a golden tensor for downstream TTNN PCC
verification.

The full v2-Large T2U encoder has 6 layers; we override
``encoder_layers=2`` on the HF config to keep the saved golden file small
while still exercising the multi-layer + final LayerNorm path. Per-layer
correctness against HF is already verified in
``test_functional_text_encoder_layer.py`` (the T2U encoder layer is
structurally identical to the text encoder layer).
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import t2u_encoder_forward

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


def _extract_encoder_layer_state_dict(layer) -> dict:
    return {
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _self_attn_sd(layer.self_attn),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def _extract_t2u_encoder_state_dict(encoder) -> dict:
    """Pull all sub-block weights out of an HF SeamlessM4Tv2Encoder (T2U variant).

    Note: T2U encoder has no embed_tokens / no embed_positions (they are
    only present when ``is_t2u_encoder=False``).
    """
    return {
        "layers": [_extract_encoder_layer_state_dict(layer) for layer in encoder.layers],
        "final_layer_norm": _ln_sd(encoder.layer_norm),
    }


def test_t2u_encoder_matches_hf() -> float:
    """Compare reference t2u_encoder forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config (with ``t2u_*`` values mapped to base
    config attributes the same way HF's ``SeamlessM4Tv2TextToUnitForConditionalGeneration``
    does at init time) but with ``encoder_layers=2`` (vs the real T2U
    encoder's 6) to keep memory + golden file size reasonable. Per-layer
    fidelity is covered by ``test_functional_text_encoder_layer.py`` (T2U
    encoder layer == text encoder layer structurally).
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2Encoder

    torch.manual_seed(0)

    batch, seq_len = 1, 16
    num_heads, head_dim = 16, 64
    hidden_size = 1024

    # Build T2U-style config. HF's `SeamlessM4Tv2TextToUnitForConditionalGeneration`
    # remaps `t2u_*` config attrs onto the base config attrs (encoder_layers,
    # encoder_attention_heads, encoder_ffn_dim, ...) before constructing the
    # encoder. Replicate that mapping here so the encoder we instantiate has
    # the right T2U-side architecture.
    config = SeamlessM4Tv2Config()
    import copy as _copy

    config = _copy.deepcopy(config)
    for param, val in config.to_dict().items():
        if param.startswith("t2u_"):
            config.__setattr__(param[4:], val)

    # Sanity-check expected v2-Large T2U defaults after remapping.
    assert config.hidden_size == hidden_size, f"unexpected hidden_size {config.hidden_size}"
    assert (
        config.encoder_attention_heads == num_heads
    ), f"unexpected encoder_attention_heads {config.encoder_attention_heads} (expected t2u=16)"
    assert config.encoder_ffn_dim == 8192, f"unexpected encoder_ffn_dim {config.encoder_ffn_dim} (expected t2u=8192)"
    assert config.activation_function == "relu", f"unexpected activation_function {config.activation_function}"
    # Default constructor's t2u_encoder_layers is 6 -> remapped to encoder_layers=6.
    assert config.encoder_layers == 6, f"unexpected encoder_layers after t2u remap {config.encoder_layers} (expected 6)"

    # Override depth to keep golden small. 2 layers is enough to exercise
    # the multi-layer composition path while staying well under 100MB.
    config.encoder_layers = 2

    eps = config.layer_norm_eps  # 1e-5 (also nn.LayerNorm default)

    encoder = SeamlessM4Tv2Encoder(config, is_t2u_encoder=True)
    encoder.eval()
    # Belt-and-braces: zero any layer-level dropouts. They're already off in
    # eval but this protects against accidental config drift.
    encoder.dropout = 0.0
    for layer in encoder.layers:
        layer.attn_dropout.p = 0.0
        layer.ffn_dropout.p = 0.0
        layer.self_attn.dropout = 0.0
        layer.ffn.dropout.p = 0.0

    state_dict = _extract_t2u_encoder_state_dict(encoder)

    # Test input: random float32 embeddings of shape [1, 16, 1024]. The T2U
    # encoder consumes inputs_embeds directly (no token lookup).
    inputs_embeds = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32)

    # --- unmasked path: no attention mask (full self-attention). ---
    with torch.no_grad():
        hf_out = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    ref_out = t2u_encoder_forward(
        inputs_embeds,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        eps=eps,
        activation_function="relu",
    )
    pcc_unmasked = _pcc(ref_out, hf_out)
    max_abs_unmasked = (ref_out - hf_out).abs().max().item()
    print(f"[t2u_encoder/unmasked] pcc={pcc_unmasked:.6f} max_abs_diff={max_abs_unmasked:.3e}")
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path: 2-D HF-style attention mask (1=keep, 0=pad). ---
    # Mask out the trailing 4 positions so the encoder must avoid attending
    # to them. We pass a 2-D mask in, just like a real HF caller would, and
    # expand it ourselves into the 4-D additive form that t2u_encoder_forward
    # expects internally.
    attention_mask_2d = torch.ones(batch, seq_len, dtype=torch.long)
    attention_mask_2d[:, -4:] = 0

    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    # HF expands using inputs_embeds.dtype (float32 here).
    attention_mask_4d = _prepare_4d_attention_mask(attention_mask_2d, torch.float32)

    with torch.no_grad():
        hf_out_masked = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_2d,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    ref_out_masked = t2u_encoder_forward(
        inputs_embeds,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=attention_mask_4d,
        eps=eps,
        activation_function="relu",
    )
    pcc_masked = _pcc(ref_out_masked, hf_out_masked)
    max_abs_masked = (ref_out_masked - hf_out_masked).abs().max().item()
    print(f"[t2u_encoder/masked]   pcc={pcc_masked:.6f} max_abs_diff={max_abs_masked:.3e}")
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(
        ref_out_masked, hf_out_masked, atol=1e-5, rtol=1e-4
    ), f"masked diverged: max_abs={max_abs_masked}"

    # Save golden tensor for downstream TTNN PCC checks. Note we save the
    # 2-layer state_dict (NOT the 6-layer one) for the same memory reason
    # as the test itself -- downstream consumers should match this scale.
    golden_path = GOLDEN_DIR / "t2u_encoder.pt"
    torch.save(
        {
            "inputs_embeds": inputs_embeds,
            "attention_mask_2d": attention_mask_2d,
            "attention_mask_4d": attention_mask_4d,
            "state_dict": state_dict,
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
                "encoder_layers_full_model": 6,  # real T2U encoder has 6 layers
                "activation_function": "relu",
                "eps": eps,
                "dtype": "float32",
                "block": "t2u_encoder",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2Encoder(is_t2u_encoder=True)",
                "notes": (
                    "Reduced to encoder_layers=2 to keep golden file small "
                    "(real T2U encoder has 6). Per-layer correctness "
                    "covered separately by test_functional_text_encoder_layer.py "
                    "(structurally identical layer)."
                ),
            },
        },
        golden_path,
    )
    print(f"[t2u_encoder] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_t2u_encoder_matches_hf()
    print(f"\nFINAL PCC t2u_encoder: {pcc:.6f}")

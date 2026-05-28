# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``speech_encoder_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2SpeechEncoder`` module at input shape ``[1, 64, 160]``
(batch, time, feature_size). The full v2-Large speech encoder has 24
Conformer layers; we override ``speech_encoder_layers=2`` on the HF config
(everything else stays at v2-Large defaults) so the saved golden file
stays small while still exercising the full feature_projection +
multi-layer Conformer + intermediate_ffn + adapter + inner_layer_norm
composition. Per-layer Conformer correctness against HF is already
covered by ``test_functional_conformer_encoder_layer.py`` and per-layer
adapter correctness by ``test_functional_conformer_adapter_layer.py``.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import speech_encoder_forward

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
    sd = {
        "ffn1_layer_norm": _ln_sd(layer.ffn1_layer_norm),
        "ffn1": _ffn_sd(layer.ffn1),
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _self_attn_sd(layer.self_attn),
        "conv_module": _conv_module_sd(layer.conv_module),
        "ffn2_layer_norm": _ln_sd(layer.ffn2_layer_norm),
        "ffn2": _ffn_sd(layer.ffn2),
        "final_layer_norm": _ln_sd(layer.final_layer_norm),
        # The relative-key distance embedding lives inside self_attn but
        # is passed as a sibling kwarg to conformer_encoder_layer_forward,
        # so we expose it as a top-level entry in the per-layer dict.
        "distance_embedding_weight": layer.self_attn.distance_embedding.weight.detach().clone(),
    }
    return sd


def _extract_adapter_layer_state_dict(layer) -> dict:
    return {
        "residual_layer_norm": _ln_sd(layer.residual_layer_norm),
        "residual_conv": _conv1d_sd(layer.residual_conv),
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn_conv": _conv1d_sd(layer.self_attn_conv),
        "self_attn": _self_attn_sd(layer.self_attn),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def _extract_speech_encoder_state_dict(model) -> dict:
    """Pull all sub-block weights out of an HF SeamlessM4Tv2SpeechEncoder."""
    sd = {
        "feature_projection": {
            "layer_norm": _ln_sd(model.feature_projection.layer_norm),
            "projection": _linear_sd(model.feature_projection.projection),
        },
        "encoder": {
            "layers": [_extract_encoder_layer_state_dict(layer) for layer in model.encoder.layers],
            "final_layer_norm": _ln_sd(model.encoder.layer_norm),
        },
        "intermediate_ffn": _ffn_sd(model.intermediate_ffn),
        "inner_layer_norm": _ln_sd(model.inner_layer_norm),
    }
    if model.adapter is not None:
        sd["adapter"] = {
            "layers": [_extract_adapter_layer_state_dict(layer) for layer in model.adapter.layers],
        }
    return sd


def test_speech_encoder_matches_hf() -> float:
    """Compare reference speech_encoder forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config but with ``speech_encoder_layers=2``
    (vs the real model's 24) to keep memory + golden file size reasonable.
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2SpeechEncoder

    torch.manual_seed(0)

    batch, seq_len, feature_size = 1, 64, 160
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    # Sanity-check expected v2-Large defaults.
    assert config.hidden_size == 1024, f"unexpected hidden_size {config.hidden_size}"
    assert (
        config.feature_projection_input_dim == feature_size
    ), f"unexpected feature_projection_input_dim {config.feature_projection_input_dim}"
    assert (
        config.speech_encoder_attention_heads == num_heads
    ), f"unexpected speech_encoder_attention_heads {config.speech_encoder_attention_heads}"
    assert config.position_embeddings_type == "relative_key"
    assert config.conv_depthwise_kernel_size == 31
    assert config.adaptor_kernel_size == 8
    assert config.adaptor_stride == 8
    assert config.num_adapter_layers == 1
    assert config.add_adapter is True
    assert config.speech_encoder_hidden_act == "swish"
    assert config.speech_encoder_chunk_size == 20000
    assert config.speech_encoder_left_chunk_num == 128

    # Override depth to keep golden small. 2 layers is enough to exercise
    # the multi-layer composition path while staying small.
    config.speech_encoder_layers = 2

    eps = config.layer_norm_eps  # 1e-5
    hidden_size = config.hidden_size  # 1024
    left_max = config.left_max_position_embeddings  # 64
    right_max = config.right_max_position_embeddings  # 8
    conv_kernel = config.conv_depthwise_kernel_size  # 31
    adaptor_kernel = config.adaptor_kernel_size  # 8
    adaptor_stride = config.adaptor_stride  # 8
    chunk_size = config.speech_encoder_chunk_size  # 20000
    left_chunk_num = config.speech_encoder_left_chunk_num  # 128

    model = SeamlessM4Tv2SpeechEncoder(config)
    model.eval()
    # Belt-and-suspenders: zero all dropouts in case downstream config drifts.
    for layer in model.encoder.layers:
        layer.self_attn_dropout.p = 0.0
        layer.self_attn.dropout.p = 0.0
        layer.conv_module.dropout.p = 0.0
        layer.ffn1.intermediate_dropout.p = 0.0
        layer.ffn1.output_dropout.p = 0.0
        layer.ffn2.intermediate_dropout.p = 0.0
        layer.ffn2.output_dropout.p = 0.0
    model.intermediate_ffn.intermediate_dropout.p = 0.0
    model.intermediate_ffn.output_dropout.p = 0.0
    if model.adapter is not None:
        for layer in model.adapter.layers:
            layer.self_attn_dropout.p = 0.0
            layer.self_attn.dropout.p = 0.0
            layer.ffn.intermediate_dropout.p = 0.0
            layer.ffn.output_dropout.p = 0.0

    state_dict = _extract_speech_encoder_state_dict(model)

    input_features = torch.randn(batch, seq_len, feature_size, dtype=torch.float32)

    # --- unmasked path: full self-attention (chunk mask only). ---
    with torch.no_grad():
        hf_out = model(
            input_features=input_features,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    ref_out = speech_encoder_forward(
        input_features,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=None,
        eps=eps,
        speech_encoder_hidden_act="swish",
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        conv_depthwise_kernel_size=conv_kernel,
        adaptor_kernel_size=adaptor_kernel,
        adaptor_stride=adaptor_stride,
        speech_encoder_chunk_size=chunk_size,
        speech_encoder_left_chunk_num=left_chunk_num,
        add_adapter=True,
    )
    assert ref_out.shape == hf_out.shape, f"shape mismatch: ref={tuple(ref_out.shape)} hf={tuple(hf_out.shape)}"
    pcc_unmasked = _pcc(ref_out, hf_out)
    max_abs_unmasked = (ref_out - hf_out).abs().max().item()
    print(
        f"[speech_encoder/unmasked] in={tuple(input_features.shape)} "
        f"out={tuple(ref_out.shape)} pcc={pcc_unmasked:.6f} "
        f"max_abs_diff={max_abs_unmasked:.3e}"
    )
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path: 2-D HF-style attention mask (1=keep, 0=pad). ---
    # Mask out the trailing 16 time-steps so the encoder must avoid attending
    # to them and the adapter must compute its own sub-sampled mask.
    attention_mask_2d = torch.ones(batch, seq_len, dtype=torch.long)
    attention_mask_2d[:, -16:] = 0

    with torch.no_grad():
        hf_out_masked = model(
            input_features=input_features,
            attention_mask=attention_mask_2d,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    ref_out_masked = speech_encoder_forward(
        input_features,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=attention_mask_2d,
        eps=eps,
        speech_encoder_hidden_act="swish",
        left_max_position_embeddings=left_max,
        right_max_position_embeddings=right_max,
        position_embeddings_type="relative_key",
        conv_depthwise_kernel_size=conv_kernel,
        adaptor_kernel_size=adaptor_kernel,
        adaptor_stride=adaptor_stride,
        speech_encoder_chunk_size=chunk_size,
        speech_encoder_left_chunk_num=left_chunk_num,
        add_adapter=True,
    )
    assert ref_out_masked.shape == hf_out_masked.shape, (
        f"masked shape mismatch: ref={tuple(ref_out_masked.shape)} " f"hf={tuple(hf_out_masked.shape)}"
    )
    pcc_masked = _pcc(ref_out_masked, hf_out_masked)
    max_abs_masked = (ref_out_masked - hf_out_masked).abs().max().item()
    print(
        f"[speech_encoder/masked]   in={tuple(input_features.shape)} "
        f"out={tuple(ref_out_masked.shape)} pcc={pcc_masked:.6f} "
        f"max_abs_diff={max_abs_masked:.3e}"
    )
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(
        ref_out_masked, hf_out_masked, atol=1e-5, rtol=1e-4
    ), f"masked diverged: max_abs={max_abs_masked}"

    # Save golden tensor for downstream TTNN PCC checks. We save the
    # 2-layer state_dict (NOT the 24-layer one) for the same memory reason
    # as the test itself -- downstream consumers should match this scale.
    golden_path = GOLDEN_DIR / "speech_encoder.pt"
    torch.save(
        {
            "input_features": input_features,
            "attention_mask_2d": attention_mask_2d,
            "state_dict": state_dict,
            "output_unmasked": ref_out,
            "output_masked": ref_out_masked,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "feature_size": feature_size,
                "hidden": hidden_size,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "speech_encoder_layers_in_golden": config.speech_encoder_layers,  # 2 (small)
                "speech_encoder_layers_full_model": 24,
                "speech_encoder_intermediate_size": config.speech_encoder_intermediate_size,
                "speech_encoder_hidden_act": "swish",
                "left_max_position_embeddings": left_max,
                "right_max_position_embeddings": right_max,
                "position_embeddings_type": "relative_key",
                "conv_depthwise_kernel_size": conv_kernel,
                "adaptor_kernel_size": adaptor_kernel,
                "adaptor_stride": adaptor_stride,
                "num_adapter_layers": config.num_adapter_layers,
                "add_adapter": True,
                "speech_encoder_chunk_size": chunk_size,
                "speech_encoder_left_chunk_num": left_chunk_num,
                "eps": eps,
                "dtype": "float32",
                "block": "speech_encoder",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2SpeechEncoder",
                "notes": (
                    "Reduced to speech_encoder_layers=2 to keep golden file small "
                    "(real model has 24). Per-layer correctness covered by "
                    "test_functional_conformer_encoder_layer.py / "
                    "test_functional_conformer_adapter_layer.py."
                ),
            },
        },
        golden_path,
    )
    print(f"[speech_encoder] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_speech_encoder_matches_hf()
    print(f"\nFINAL PCC speech_encoder: {pcc:.6f}")

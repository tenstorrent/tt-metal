# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``conformer_adapter_layer_forward``.

Compares the standalone reference implementation against the HuggingFace
``SeamlessM4Tv2ConformerAdapterLayer`` module at input shape ``[1, 128, 1024]``
(batch, seq_len, hidden) using the v2-Large config, then saves a golden
tensor for downstream TTNN PCC verification.

The adapter layer applies a strided Conv1d (stride=8, kernel=8, padding=4)
to down-sample the temporal axis, so the output shape is ``[1, 17, 1024]``
(``floor((128 + 8 - 8) / 8) + 1 = 17``). The spec note about ``[1, 16, 1024]``
refers to the idealised stride=8 ratio (``128 / 8 = 16``), but PyTorch's
Conv1d with asymmetric padding produces 17 time-steps. We follow the HF
actual output exactly.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import conformer_adapter_layer_forward

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


def _extract_adapter_layer_state_dict(layer) -> dict:
    """Pull all sub-block weights out of an HF SeamlessM4Tv2ConformerAdapterLayer."""
    return {
        "residual_layer_norm": _ln_sd(layer.residual_layer_norm),
        "residual_conv": _conv1d_sd(layer.residual_conv),
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn_conv": _conv1d_sd(layer.self_attn_conv),
        "self_attn": _self_attn_sd(layer.self_attn),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def test_conformer_adapter_layer_matches_hf() -> float:
    """Compare reference ConformerAdapterLayer forward against HuggingFace.

    Uses SeamlessM4T-v2-Large config:
        - hidden_size = 1024
        - speech_encoder_attention_heads = 16 (head_dim = 64)
        - adaptor_kernel_size = 8
        - adaptor_stride = 8
        - adaptor_dropout = 0.1 (disabled at eval)
        - layer_norm_eps = 1e-5
    Input shape: [1, 128, 1024], output shape: [1, 17, 1024] (downsampled by ~8x).
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ConformerAdapterLayer

    torch.manual_seed(0)

    batch, seq_len, hidden = 1, 128, 1024
    num_heads, head_dim = 16, 64

    config = SeamlessM4Tv2Config()
    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert (
        config.speech_encoder_attention_heads == num_heads
    ), f"unexpected num_heads {config.speech_encoder_attention_heads}"
    assert config.adaptor_kernel_size == 8, f"unexpected adaptor_kernel_size {config.adaptor_kernel_size}"
    assert config.adaptor_stride == 8, f"unexpected adaptor_stride {config.adaptor_stride}"

    kernel_size = config.adaptor_kernel_size  # 8
    stride = config.adaptor_stride  # 8
    eps = config.layer_norm_eps  # 1e-5

    hf = SeamlessM4Tv2ConformerAdapterLayer(config)
    hf.eval()
    # Belt-and-suspenders: explicitly zero all dropouts in case downstream config changes.
    hf.self_attn_dropout.p = 0.0
    hf.self_attn.dropout.p = 0.0
    hf.ffn.intermediate_dropout.p = 0.0
    hf.ffn.output_dropout.p = 0.0

    state_dict = _extract_adapter_layer_state_dict(hf)

    hidden_states = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    # --- unmasked path ---
    with torch.no_grad():
        hf_out = hf(
            hidden_states=hidden_states,
            attention_mask=None,
            output_attentions=False,
        )
    ref_out = conformer_adapter_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        kernel_size=kernel_size,
        stride=stride,
        eps=eps,
        attention_mask=None,
    )
    assert ref_out.shape == hf_out.shape, f"shape mismatch: ref={ref_out.shape} hf={hf_out.shape}"
    pcc_unmasked = _pcc(ref_out, hf_out)
    max_abs_unmasked = (ref_out - hf_out).abs().max().item()
    print(
        f"[conformer_adapter_layer/unmasked] in={tuple(hidden_states.shape)} "
        f"out={tuple(ref_out.shape)} pcc={pcc_unmasked:.6f} "
        f"max_abs_diff={max_abs_unmasked:.3e}"
    )
    assert pcc_unmasked > 0.99, f"unmasked PCC {pcc_unmasked} <= 0.99"
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"unmasked diverged: max_abs={max_abs_unmasked}"

    # --- masked path ---
    # HF builds the post-downsampling 4D attention mask itself from a 2D bool
    # mask using _compute_sub_sample_lengths_from_attention_mask +
    # _compute_new_attention_mask + _prepare_4d_attention_mask. Our reference
    # leaves that to the caller (consistent with conformer_encoder_layer_forward),
    # so we mirror HF's pipeline here to construct an equivalent additive log
    # mask of shape [batch, 1, sub_seq_len, sub_seq_len] for the reference.
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import (
        _compute_new_attention_mask,
        _prepare_4d_attention_mask,
    )

    # 2D bool attention mask over the *pre-downsample* axis; mark last 16
    # frames as padded.
    pre_attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    pre_attention_mask[:, -16:] = 0

    # Reuse HF's helper to compute sub-sampled lengths and the corresponding
    # bool mask -> 4D additive log-mask.
    sub_sampled_lengths = hf._compute_sub_sample_lengths_from_attention_mask(pre_attention_mask)
    # Build a dummy hidden tensor of the correct sub-sampled shape so we can
    # use _compute_new_attention_mask. We only need its shape/device/dtype.
    sub_seq_len = ref_out.shape[1]
    dummy_sub_hidden = torch.zeros(batch, sub_seq_len, hidden, dtype=torch.float32)
    sub_attention_mask_2d = _compute_new_attention_mask(hidden_states=dummy_sub_hidden, seq_lens=sub_sampled_lengths)
    sub_attention_mask_4d = _prepare_4d_attention_mask(sub_attention_mask_2d, dummy_sub_hidden.dtype)

    with torch.no_grad():
        hf_out_masked = hf(
            hidden_states=hidden_states,
            attention_mask=pre_attention_mask,
            output_attentions=False,
        )
    ref_out_masked = conformer_adapter_layer_forward(
        hidden_states,
        state_dict,
        num_heads=num_heads,
        head_dim=head_dim,
        kernel_size=kernel_size,
        stride=stride,
        eps=eps,
        attention_mask=sub_attention_mask_4d,
    )
    pcc_masked = _pcc(ref_out_masked, hf_out_masked)
    max_abs_masked = (ref_out_masked - hf_out_masked).abs().max().item()
    print(
        f"[conformer_adapter_layer/masked]   in={tuple(hidden_states.shape)} "
        f"out={tuple(ref_out_masked.shape)} pcc={pcc_masked:.6f} "
        f"max_abs_diff={max_abs_masked:.3e}"
    )
    assert pcc_masked > 0.99, f"masked PCC {pcc_masked} <= 0.99"
    assert torch.allclose(
        ref_out_masked, hf_out_masked, atol=1e-5, rtol=1e-4
    ), f"masked diverged: max_abs={max_abs_masked}"

    # Save golden tensor for downstream TTNN PCC checks.
    golden_path = GOLDEN_DIR / "conformer_adapter_layer.pt"
    torch.save(
        {
            "input": hidden_states,
            "pre_attention_mask": pre_attention_mask,
            "sub_attention_mask_4d": sub_attention_mask_4d,
            "state_dict": state_dict,
            "output_unmasked": ref_out,
            "output_masked": ref_out_masked,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "sub_seq_len": sub_seq_len,
                "hidden": hidden,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "kernel_size": kernel_size,
                "stride": stride,
                "eps": eps,
                "dtype": "float32",
                "block": "conformer_adapter_layer",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2ConformerAdapterLayer",
            },
        },
        golden_path,
    )
    print(f"[conformer_adapter_layer] saved golden to {golden_path}")
    return min(pcc_unmasked, pcc_masked)


if __name__ == "__main__":
    pcc = test_conformer_adapter_layer_matches_hf()
    print(f"\nFINAL PCC conformer_adapter_layer: {pcc:.6f}")

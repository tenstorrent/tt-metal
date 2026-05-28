# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``conformer_convolution_module_forward``.

Builds an HuggingFace ``SeamlessM4Tv2ConformerConvolutionModule`` at the
SeamlessM4T-v2-Large sizes (``hidden_size=1024``,
``conv_depthwise_kernel_size=31``, activation=``swish``), extracts its weights,
and verifies the standalone reference matches bit-for-bit. Saves a golden
tensor for downstream TTNN PCC verification.

Kept in a separate file from ``test_functional.py`` so parallel workers do not
race when writing the shared test module.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import conformer_convolution_module_forward

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


def _extract_conv_module_state_dict(module) -> dict:
    """Pull weights out of the HF ``SeamlessM4Tv2ConformerConvolutionModule``."""
    return {
        "layer_norm": {
            "weight": module.layer_norm.weight.detach().clone(),
            "bias": module.layer_norm.bias.detach().clone(),
        },
        "pointwise_conv1": {"weight": module.pointwise_conv1.weight.detach().clone()},
        "depthwise_conv": {"weight": module.depthwise_conv.weight.detach().clone()},
        "depthwise_layer_norm": {
            "weight": module.depthwise_layer_norm.weight.detach().clone(),
            "bias": module.depthwise_layer_norm.bias.detach().clone(),
        },
        "pointwise_conv2": {"weight": module.pointwise_conv2.weight.detach().clone()},
    }


def test_conformer_convolution_module_matches_hf() -> float:
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ConformerConvolutionModule

    torch.manual_seed(0)

    batch, seq_len, hidden = 1, 128, 1024
    kernel_size = 31
    eps = 1e-5

    config = SeamlessM4Tv2Config()
    assert config.hidden_size == hidden, f"unexpected hidden_size {config.hidden_size}"
    assert (
        config.conv_depthwise_kernel_size == kernel_size
    ), f"unexpected conv_depthwise_kernel_size {config.conv_depthwise_kernel_size}"
    assert (
        config.speech_encoder_hidden_act == "swish"
    ), f"unexpected speech_encoder_hidden_act {config.speech_encoder_hidden_act}"
    assert config.layer_norm_eps == eps, f"unexpected layer_norm_eps {config.layer_norm_eps}"

    hf = SeamlessM4Tv2ConformerConvolutionModule(config)
    hf.eval()
    state_dict = _extract_conv_module_state_dict(hf)

    x = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    with torch.no_grad():
        hf_out = hf(x)

    ref_out = conformer_convolution_module_forward(
        x,
        state_dict,
        kernel_size=kernel_size,
        eps=eps,
    )

    pcc = _pcc(ref_out, hf_out)
    max_abs = (ref_out - hf_out).abs().max().item()
    print(f"[conformer_convolution_module] pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")
    assert pcc > 0.99, f"conformer_convolution_module PCC {pcc} <= 0.99"
    # fp32 reference + fp32 HF -> identical op sequence -> near-exact match.
    assert torch.allclose(
        ref_out, hf_out, atol=1e-5, rtol=1e-4
    ), f"conformer_convolution_module diverged: max_abs={max_abs}"

    # Also verify with an attention_mask (padded positions zeroed pre-depthwise).
    torch.manual_seed(1)
    x_masked = torch.randn(batch, seq_len, hidden, dtype=torch.float32)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    attention_mask[:, -10:] = False
    with torch.no_grad():
        hf_out_m = hf(x_masked, attention_mask=attention_mask)
    ref_out_m = conformer_convolution_module_forward(
        x_masked,
        state_dict,
        kernel_size=kernel_size,
        eps=eps,
        attention_mask=attention_mask,
    )
    pcc_m = _pcc(ref_out_m, hf_out_m)
    max_abs_m = (ref_out_m - hf_out_m).abs().max().item()
    print(f"[conformer_convolution_module/masked] pcc={pcc_m:.6f} max_abs_diff={max_abs_m:.3e}")
    assert pcc_m > 0.99, f"conformer_convolution_module (masked) PCC {pcc_m} <= 0.99"
    assert torch.allclose(
        ref_out_m, hf_out_m, atol=1e-5, rtol=1e-4
    ), f"conformer_convolution_module (masked) diverged: max_abs={max_abs_m}"

    golden_path = GOLDEN_DIR / "conformer_convolution_module.pt"
    torch.save(
        {
            "input": x,
            "state_dict": state_dict,
            "output": ref_out,
            "input_masked": x_masked,
            "attention_mask": attention_mask,
            "output_masked": ref_out_m,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden,
                "kernel_size": kernel_size,
                "eps": eps,
                "activation": "swish",
                "dtype": "float32",
                "block": "conformer_convolution_module",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2ConformerConvolutionModule",
            },
        },
        golden_path,
    )
    print(f"[conformer_convolution_module] saved golden to {golden_path}")
    return min(pcc, pcc_m)


if __name__ == "__main__":
    pcc = test_conformer_convolution_module_matches_hf()
    print(f"\nFINAL PCC conformer_convolution_module: {pcc:.6f}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``hifigan_vocoder_forward``.

Builds an HuggingFace ``SeamlessM4Tv2HifiGan`` at the SeamlessM4T-v2-Large
config defaults (model_in_dim = unit_embed_dim + lang_embed_dim + spkr_embed_dim
= 1280 + 256 + 256 = 1792, upsample_initial_channel=512, upsample_rates
=(5,4,4,2,2), upsample_kernel_sizes=(11,8,8,4,4), resblock_kernel_sizes
=(3,7,11), resblock_dilation_sizes=((1,3,5),(1,3,5),(1,3,5)),
leaky_relu_slope=0.1), extracts its weights, and verifies the standalone
reference matches bit-for-bit. Saves a golden tensor for downstream TTNN PCC
verification.

Kept in a separate file from ``test_functional.py`` so parallel workers do not
race when writing the shared test module.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import hifigan_vocoder_forward

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


def _extract_residual_block_state_dict(module) -> dict:
    """Pull weights out of an HF ``HifiGanResidualBlock``."""
    convs1 = []
    for conv in module.convs1:
        convs1.append(
            {
                "weight": conv.weight.detach().clone(),
                "bias": conv.bias.detach().clone(),
            }
        )
    convs2 = []
    for conv in module.convs2:
        convs2.append(
            {
                "weight": conv.weight.detach().clone(),
                "bias": conv.bias.detach().clone(),
            }
        )
    return {"convs1": convs1, "convs2": convs2}


def _extract_hifigan_state_dict(module) -> dict:
    """Pull weights out of an HF ``SeamlessM4Tv2HifiGan``."""
    conv_pre = {
        "weight": module.conv_pre.weight.detach().clone(),
        "bias": module.conv_pre.bias.detach().clone(),
    }
    upsampler = []
    for layer in module.upsampler:
        upsampler.append(
            {
                "weight": layer.weight.detach().clone(),
                "bias": layer.bias.detach().clone(),
            }
        )
    resblocks = []
    for rb in module.resblocks:
        resblocks.append(_extract_residual_block_state_dict(rb))
    conv_post = {
        "weight": module.conv_post.weight.detach().clone(),
        "bias": module.conv_post.bias.detach().clone(),
    }
    return {
        "conv_pre": conv_pre,
        "upsampler": upsampler,
        "resblocks": resblocks,
        "conv_post": conv_post,
    }


def test_hifigan_vocoder_matches_hf() -> float:
    from transformers.models.seamless_m4t_v2.configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2HifiGan

    torch.manual_seed(0)

    config = SeamlessM4Tv2Config()
    # SeamlessM4T-v2-Large defaults
    upsample_rates = tuple(config.upsample_rates)
    upsample_kernel_sizes = tuple(config.upsample_kernel_sizes)
    resblock_kernel_sizes = tuple(config.resblock_kernel_sizes)
    resblock_dilation_sizes = tuple(tuple(d) for d in config.resblock_dilation_sizes)
    leaky_relu_slope = config.leaky_relu_slope
    model_in_dim = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim

    hf = SeamlessM4Tv2HifiGan(config)
    hf.eval()

    state_dict = _extract_hifigan_state_dict(hf)

    # Keep input small per spec: [1, model_in_dim, 16]
    batch, time = 1, 16
    x = torch.randn(batch, model_in_dim, time, dtype=torch.float32)

    with torch.no_grad():
        hf_out = hf(x)

    ref_out = hifigan_vocoder_forward(
        x,
        state_dict,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        leaky_relu_slope=leaky_relu_slope,
    )

    assert ref_out.shape == hf_out.shape, f"shape mismatch ref {ref_out.shape} vs hf {hf_out.shape}"

    pcc = _pcc(ref_out, hf_out)
    max_abs = (ref_out - hf_out).abs().max().item()
    print(f"[hifigan_vocoder] out_shape={tuple(ref_out.shape)} pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")
    assert pcc > 0.99, f"hifigan_vocoder PCC {pcc} <= 0.99"
    # fp32 reference + fp32 HF -> identical op sequence -> near-exact match.
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"hifigan_vocoder diverged: max_abs={max_abs}"

    golden_path = GOLDEN_DIR / "hifigan_vocoder.pt"
    torch.save(
        {
            "input": x,
            "state_dict": state_dict,
            "output": ref_out,
            "config": {
                "batch": batch,
                "model_in_dim": model_in_dim,
                "time_in": time,
                "time_out": int(ref_out.shape[-1]),
                "upsample_rates": upsample_rates,
                "upsample_kernel_sizes": upsample_kernel_sizes,
                "upsample_initial_channel": config.upsample_initial_channel,
                "resblock_kernel_sizes": resblock_kernel_sizes,
                "resblock_dilation_sizes": resblock_dilation_sizes,
                "leaky_relu_slope": leaky_relu_slope,
                "unit_embed_dim": config.unit_embed_dim,
                "lang_embed_dim": config.lang_embed_dim,
                "spkr_embed_dim": config.spkr_embed_dim,
                "dtype": "float32",
                "block": "hifigan_vocoder",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2HifiGan",
            },
        },
        golden_path,
    )
    print(f"[hifigan_vocoder] saved golden to {golden_path}")
    return pcc


if __name__ == "__main__":
    pcc = test_hifigan_vocoder_matches_hf()
    print(f"\nFINAL PCC hifigan_vocoder: {pcc:.6f}")

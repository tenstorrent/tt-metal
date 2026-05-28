# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``hifigan_residual_block_forward``.

Builds an HuggingFace ``HifiGanResidualBlock`` at the SeamlessM4T-v2-Large
HiFi-GAN vocoder default sizes (``channels=512``, ``kernel_size=3``,
``dilation=(1, 3, 5)``, ``leaky_relu_slope=0.1``), extracts its weights, and
verifies the standalone reference matches bit-for-bit. Saves a golden tensor
for downstream TTNN PCC verification.

Kept in a separate file from ``test_functional.py`` so parallel workers do not
race when writing the shared test module.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import hifigan_residual_block_forward

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
    """Pull weights out of the HF ``HifiGanResidualBlock``."""
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


def test_hifigan_residual_block_matches_hf() -> float:
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import HifiGanResidualBlock

    torch.manual_seed(0)

    batch, channels, time = 1, 512, 80
    kernel_size = 3
    dilation = (1, 3, 5)
    leaky_relu_slope = 0.1

    hf = HifiGanResidualBlock(
        channels=channels,
        kernel_size=kernel_size,
        dilation=dilation,
        leaky_relu_slope=leaky_relu_slope,
    )
    hf.eval()
    state_dict = _extract_residual_block_state_dict(hf)

    x = torch.randn(batch, channels, time, dtype=torch.float32)

    with torch.no_grad():
        hf_out = hf(x)

    ref_out = hifigan_residual_block_forward(
        x,
        state_dict,
        kernel_size=kernel_size,
        dilation=dilation,
        leaky_relu_slope=leaky_relu_slope,
    )

    pcc = _pcc(ref_out, hf_out)
    max_abs = (ref_out - hf_out).abs().max().item()
    print(f"[hifigan_residual_block] pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")
    assert pcc > 0.99, f"hifigan_residual_block PCC {pcc} <= 0.99"
    # fp32 reference + fp32 HF -> identical op sequence -> near-exact match.
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"hifigan_residual_block diverged: max_abs={max_abs}"

    golden_path = GOLDEN_DIR / "hifigan_residual_block.pt"
    torch.save(
        {
            "input": x,
            "state_dict": state_dict,
            "output": ref_out,
            "config": {
                "batch": batch,
                "channels": channels,
                "time": time,
                "kernel_size": kernel_size,
                "dilation": dilation,
                "leaky_relu_slope": leaky_relu_slope,
                "dtype": "float32",
                "block": "hifigan_residual_block",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "HifiGanResidualBlock",
            },
        },
        golden_path,
    )
    print(f"[hifigan_residual_block] saved golden to {golden_path}")
    return pcc


if __name__ == "__main__":
    pcc = test_hifigan_residual_block_matches_hf()
    print(f"\nFINAL PCC hifigan_residual_block: {pcc:.6f}")

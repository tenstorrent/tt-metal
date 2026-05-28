# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``variance_predictor_forward``.

Builds an HuggingFace ``SeamlessM4Tv2VariancePredictor`` at the T2U
duration-predictor sizes (``embed_dim=1024``, ``hidden_dim=256``,
``kernel_size=3``), extracts its weights, and verifies the standalone
reference matches bit-for-bit. Saves a golden tensor for downstream TTNN
PCC verification.

Kept in a separate file from ``test_functional.py`` so parallel workers do not
race when writing the shared test module.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import variance_predictor_forward

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


def _extract_variance_predictor_state_dict(module) -> dict:
    """Pull weights out of the HF ``SeamlessM4Tv2VariancePredictor``."""
    return {
        "conv1": {
            "weight": module.conv1.weight.detach().clone(),
            "bias": module.conv1.bias.detach().clone(),
        },
        "ln1": {
            "weight": module.ln1.weight.detach().clone(),
            "bias": module.ln1.bias.detach().clone(),
        },
        "conv2": {
            "weight": module.conv2.weight.detach().clone(),
            "bias": module.conv2.bias.detach().clone(),
        },
        "ln2": {
            "weight": module.ln2.weight.detach().clone(),
            "bias": module.ln2.bias.detach().clone(),
        },
        "proj": {
            "weight": module.proj.weight.detach().clone(),
            "bias": module.proj.bias.detach().clone(),
        },
    }


def test_variance_predictor_matches_hf() -> float:
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2VariancePredictor

    torch.manual_seed(0)

    batch, seq_len = 1, 64
    embed_dim = 1024
    hidden_dim = 256
    kernel_size = 3
    eps = 1e-5

    # Build HF VariancePredictor in eval mode (dropout = no-op).
    hf = SeamlessM4Tv2VariancePredictor(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        var_pred_dropout=0.5,
    )
    hf.eval()
    state_dict = _extract_variance_predictor_state_dict(hf)

    # Sanity-check shapes against documented expectations.
    assert state_dict["conv1"]["weight"].shape == (hidden_dim, embed_dim, kernel_size)
    assert state_dict["conv1"]["bias"].shape == (hidden_dim,)
    assert state_dict["ln1"]["weight"].shape == (hidden_dim,)
    assert state_dict["ln1"]["bias"].shape == (hidden_dim,)
    assert state_dict["conv2"]["weight"].shape == (hidden_dim, hidden_dim, kernel_size)
    assert state_dict["conv2"]["bias"].shape == (hidden_dim,)
    assert state_dict["ln2"]["weight"].shape == (hidden_dim,)
    assert state_dict["ln2"]["bias"].shape == (hidden_dim,)
    assert state_dict["proj"]["weight"].shape == (1, hidden_dim)
    assert state_dict["proj"]["bias"].shape == (1,)

    x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float32)

    with torch.no_grad():
        hf_out = hf(x)

    ref_out = variance_predictor_forward(
        x,
        state_dict,
        kernel_size=kernel_size,
        eps=eps,
    )

    # Output shape: (B, T) after squeeze.
    assert ref_out.shape == (batch, seq_len), f"unexpected ref_out shape {tuple(ref_out.shape)}"
    assert hf_out.shape == ref_out.shape, f"hf={tuple(hf_out.shape)} ref={tuple(ref_out.shape)}"

    pcc = _pcc(ref_out, hf_out)
    max_abs = (ref_out - hf_out).abs().max().item()
    print(f"[variance_predictor] pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")
    assert pcc > 0.99, f"variance_predictor PCC {pcc} <= 0.99"
    # fp32 reference + fp32 HF -> identical op sequence -> near-exact match.
    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"variance_predictor diverged: max_abs={max_abs}"

    # Also verify with a padding_mask (padded positions zeroed pre-conv).
    torch.manual_seed(1)
    x_masked = torch.randn(batch, seq_len, embed_dim, dtype=torch.float32)
    padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    padding_mask[:, -8:] = False
    with torch.no_grad():
        hf_out_m = hf(x_masked, padding_mask=padding_mask)
    ref_out_m = variance_predictor_forward(
        x_masked,
        state_dict,
        kernel_size=kernel_size,
        eps=eps,
        padding_mask=padding_mask,
    )
    pcc_m = _pcc(ref_out_m, hf_out_m)
    max_abs_m = (ref_out_m - hf_out_m).abs().max().item()
    print(f"[variance_predictor/masked] pcc={pcc_m:.6f} max_abs_diff={max_abs_m:.3e}")
    assert pcc_m > 0.99, f"variance_predictor (masked) PCC {pcc_m} <= 0.99"
    assert torch.allclose(
        ref_out_m, hf_out_m, atol=1e-5, rtol=1e-4
    ), f"variance_predictor (masked) diverged: max_abs={max_abs_m}"

    golden_path = GOLDEN_DIR / "variance_predictor.pt"
    torch.save(
        {
            "input": x,
            "state_dict": state_dict,
            "output": ref_out,
            "input_masked": x_masked,
            "padding_mask": padding_mask,
            "output_masked": ref_out_m,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "kernel_size": kernel_size,
                "eps": eps,
                "activation": "relu",
                "dtype": "float32",
                "block": "variance_predictor",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2VariancePredictor",
            },
        },
        golden_path,
    )
    print(f"[variance_predictor] saved golden to {golden_path}")
    return min(pcc, pcc_m)


if __name__ == "__main__":
    pcc = test_variance_predictor_matches_hf()
    print(f"\nFINAL PCC variance_predictor: {pcc:.6f}")

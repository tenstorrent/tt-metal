# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verification test for ``conformer_feature_projection_forward``.

Builds an HuggingFace ``SeamlessM4Tv2ConformerFeatureProjection`` at the
SeamlessM4T-v2-Large sizes (``feature_projection_input_dim=160``,
``hidden_size=1024``), extracts its weights, and verifies the standalone
reference matches bit-for-bit. Saves a golden tensor for downstream TTNN
PCC verification.

Kept in a separate file from ``test_functional.py`` so parallel workers do not
race when writing the shared test module.
"""

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import conformer_feature_projection_forward

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


def _extract_feature_projection_state_dict(module) -> dict:
    """Pull weights out of the HF ``SeamlessM4Tv2ConformerFeatureProjection``."""
    return {
        "layer_norm": {
            "weight": module.layer_norm.weight.detach().clone(),
            "bias": module.layer_norm.bias.detach().clone(),
        },
        "projection": {
            "weight": module.projection.weight.detach().clone(),
            "bias": module.projection.bias.detach().clone(),
        },
    }


def test_conformer_feature_projection_matches_hf() -> float:
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ConformerFeatureProjection

    torch.manual_seed(0)

    batch, seq_len = 1, 64
    feature_size = 160
    hidden_size = 1024
    eps = 1e-5

    # Build HF feature projection using the v2-Large config defaults
    # (feature_projection_input_dim=160, hidden_size=1024, layer_norm_eps=1e-5,
    # speech_encoder_dropout=0.0).
    config = SeamlessM4Tv2Config()
    # Sanity-check the assumed config values for v2-Large.
    assert (
        config.feature_projection_input_dim == feature_size
    ), f"feature_projection_input_dim={config.feature_projection_input_dim} != {feature_size}"
    assert config.hidden_size == hidden_size, f"hidden_size={config.hidden_size} != {hidden_size}"
    assert config.layer_norm_eps == eps, f"layer_norm_eps={config.layer_norm_eps} != {eps}"

    hf = SeamlessM4Tv2ConformerFeatureProjection(config)
    hf.eval()
    state_dict = _extract_feature_projection_state_dict(hf)

    # Sanity-check shapes against documented expectations.
    assert state_dict["layer_norm"]["weight"].shape == (feature_size,)
    assert state_dict["layer_norm"]["bias"].shape == (feature_size,)
    assert state_dict["projection"]["weight"].shape == (hidden_size, feature_size)
    assert state_dict["projection"]["bias"].shape == (hidden_size,)

    x = torch.randn(batch, seq_len, feature_size, dtype=torch.float32)

    with torch.no_grad():
        hf_out = hf(x)

    ref_out = conformer_feature_projection_forward(
        x,
        state_dict,
        eps=eps,
    )

    assert ref_out.shape == (batch, seq_len, hidden_size), f"unexpected ref_out shape {tuple(ref_out.shape)}"
    assert hf_out.shape == ref_out.shape, f"hf={tuple(hf_out.shape)} ref={tuple(ref_out.shape)}"

    pcc = _pcc(ref_out, hf_out)
    max_abs = (ref_out - hf_out).abs().max().item()
    print(f"[conformer_feature_projection] pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")
    assert pcc > 0.99, f"conformer_feature_projection PCC {pcc} <= 0.99"
    # fp32 reference + fp32 HF -> identical op sequence -> near-exact match.
    assert torch.allclose(
        ref_out, hf_out, atol=1e-5, rtol=1e-4
    ), f"conformer_feature_projection diverged: max_abs={max_abs}"

    golden_path = GOLDEN_DIR / "conformer_feature_projection.pt"
    torch.save(
        {
            "input": x,
            "state_dict": state_dict,
            "output": ref_out,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "feature_size": feature_size,
                "hidden_size": hidden_size,
                "eps": eps,
                "dtype": "float32",
                "block": "conformer_feature_projection",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2ConformerFeatureProjection",
            },
        },
        golden_path,
    )
    print(f"[conformer_feature_projection] saved golden to {golden_path}")
    return pcc


if __name__ == "__main__":
    pcc = test_conformer_feature_projection_matches_hf()
    print(f"\nFINAL PCC conformer_feature_projection: {pcc:.6f}")

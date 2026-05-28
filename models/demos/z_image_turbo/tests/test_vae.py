# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VAE decoder correctness test — compare TTNN forward pass against
the PyTorch reference (diffusers AutoencoderKL decoder).
"""

import pytest

import ttnn
from models.demos.z_image_turbo.tt.vae.model_pt import VaeDecoderPT, get_input
from models.demos.z_image_turbo.tt.vae.model_ttnn import VaeDecoderTTNN


def pcc(a, b):
    a_flat = a.flatten().double()
    b_flat = b.flatten().double()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    den = a_centered.norm() * b_centered.norm()
    return (num / den).item() if den > 0 else 0.0


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_vae_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()

    latent = get_input()  # [1, 16, 64, 64] float32, seed=42

    # --- PyTorch reference ---
    pt_vae = VaeDecoderPT()
    pt_result = pt_vae.forward(latent)  # [1, 3, 512, 512]
    del pt_vae

    # --- TTNN ---
    tt_vae = VaeDecoderTTNN(mesh_device)
    tt_result = tt_vae(latent)  # [1, 3, 512, 512]

    correlation = pcc(pt_result, tt_result)
    print(f"\nVAE PyTorch vs TTNN: PCC={correlation:.6f}")
    print(f"  PT  output: shape={pt_result.shape}, range=[{pt_result.min():.4f}, {pt_result.max():.4f}]")
    print(f"  TT  output: shape={tt_result.shape}, range=[{tt_result.min():.4f}, {tt_result.max():.4f}]")
    assert correlation > 0.998, f"PyTorch vs TTNN PCC too low: {correlation:.6f}"

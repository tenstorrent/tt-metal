# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""VAE decoder test — deterministic input, compare output against reference."""

import os

import pytest
import torch

from models.demos.z_image_turbo.tt.vae.model_ttnn import VaeDecoderTTNN

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference_outputs", "vae_test_output.pt")


@pytest.fixture(scope="function")
def device_params(request):
    return {"l1_small_size": 1 << 15, "trace_region_size": 50_000_000}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_vae_decoder(mesh_device):
    mesh_device.enable_program_cache()

    torch.manual_seed(42)
    latent = torch.randn(1, 16, 64, 64, dtype=torch.float32)

    vae = VaeDecoderTTNN(mesh_device)
    output = vae(latent)

    assert output.shape[0] == 1, f"Expected batch size 1, got {output.shape[0]}"
    assert output.ndim == 4, f"Expected 4D tensor, got {output.ndim}D"

    if os.path.exists(REFERENCE_PATH):
        ref = torch.load(REFERENCE_PATH, weights_only=True)
        assert ref.shape == output.shape, f"Shape mismatch: output {output.shape} vs reference {ref.shape}"
        pcc = torch.corrcoef(torch.stack([output.flatten(), ref.flatten()]))[0, 1].item()
        assert pcc > 0.99, f"PCC too low: {pcc:.6f}"

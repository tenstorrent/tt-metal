# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test: Hunyuan VAE conv_in PyTorch ref vs TTNN (replicated mesh)."""

import pytest
import torch

import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.vae_decoder import get_input, load_conv_in
from models.experimental.hunyuan_image_3_0.tt.vae.conv_in import ConvInTTNN
from models.tt_dit.utils.check import assert_quality


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_conv_in_vs_pytorch(mesh_device):
    """Phase 1: replicated (1,4) mesh — conv_in only."""
    mesh_device.enable_program_cache()

    z = get_input()  # [1, 32, 1, 64, 64]

    with torch.no_grad():
        pt_out = load_conv_in()(z)

    tt_conv_in = ConvInTTNN(mesh_device)
    tt_out = tt_conv_in(z)

    assert pt_out.shape == tt_out.shape == (1, 1024, 1, 64, 64)
    assert_quality(pt_out, tt_out, pcc=0.998)

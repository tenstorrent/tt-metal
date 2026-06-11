# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test: Hunyuan VAE mid block PyTorch ref vs TTNN."""

import pytest
import torch

import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.vae_decoder import get_mid_input, load_mid
from models.experimental.hunyuan_image_3_0.tt.vae.mid import MidBlockTTNN
from models.tt_dit.utils.check import assert_quality


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_mid_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()

    x = get_mid_input()

    with torch.no_grad():
        pt_out = load_mid()(x)

    tt_mid = MidBlockTTNN(mesh_device)
    tt_out = tt_mid(x)

    assert pt_out.shape == tt_out.shape == (1, 1024, 1, 64, 64)
    assert_quality(pt_out, tt_out, pcc=0.998)

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


### SDPA comparison
import os

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.siglip.compare import comp_pcc
from models.demos.siglip.reference.functional import siglip_layer_norm
from models.demos.siglip.tests.common import convert_state_dict
from models.demos.siglip.tt.layer_norm import siglip_layer_norm_ttnn

# Global device variable to be set from outside
global mesh_device


@pytest.mark.parametrize("layer_norm_func", [siglip_layer_norm, siglip_layer_norm_ttnn])
def test_layer_norm(layer_norm_func):
    config = AutoConfig.from_pretrained(os.getenv("HF_MODEL"))
    assert hasattr(
        config, "vision_config"
    ), f"Unexpected model config provided. Expected a vision_config field to be present in: {config}"
    config = config.vision_config

    reference_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    batch = 1
    seq_len = 4096
    expected_max_input_scale = 250
    expected_min_input_scale = -100

    random_inputs = (
        torch.rand(batch, seq_len, config.hidden_size, dtype=reference_layer_norm.state_dict()["weight"].dtype)
        * (expected_max_input_scale - expected_min_input_scale)
    ) - (
        (expected_max_input_scale - expected_min_input_scale) / 2
    )  # Random inputs scaled to range of first attention inputs

    reference_output = reference_layer_norm(
        random_inputs,
    )
    state_dict = convert_state_dict(reference_layer_norm.state_dict())
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device()
    result = layer_norm_func(
        mesh_device=mesh_device,
        hidden_states=random_inputs,
        state_dict=state_dict,
        dim=config.hidden_size,
        eps=config.layer_norm_eps,
        state_dict_prefix="",
        weight_cache_path=None,
    )
    ttnn.close_mesh_device(mesh_device)
    result, pcc = comp_pcc(reference_output, result)

    if result:
        print(f"✅ Siglip SDPA attention passes with PCC: {pcc}")
    else:
        print(f"❌ Siglip SDPA attention fails with PCC: {pcc}")
        assert False

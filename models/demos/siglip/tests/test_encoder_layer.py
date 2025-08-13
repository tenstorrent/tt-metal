# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


### SDPA comparison
import os

import pytest
import torch
from transformers import AutoConfig
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

import ttnn
from models.demos.siglip.compare import comp_pcc
from models.demos.siglip.reference.functional import siglip_encoder_layer
from models.demos.siglip.tests.common import convert_state_dict
from models.demos.siglip.tt.encoder_layer import siglip_encoder_layer_ttnn

# Global device variable to be set from outside
global mesh_device


@pytest.mark.parametrize("encoder_layer_func", [siglip_encoder_layer, siglip_encoder_layer_ttnn])
def test_encoder_layer(encoder_layer_func):
    config = AutoConfig.from_pretrained(os.getenv("HF_MODEL"))
    assert hasattr(
        config, "vision_config"
    ), f"Unexpected model config provided. Expected a vision_config field to be present in: {config}"
    config = config.vision_config

    reference_encoder_layer = SiglipEncoderLayer(config=config)

    batch = 1
    seq_len = 4096
    expected_max_input_scale = 250
    expected_min_input_scale = -100

    random_inputs = (
        torch.rand(
            batch,
            seq_len,
            config.hidden_size,
            dtype=reference_encoder_layer.state_dict()["self_attn.k_proj.weight"].dtype,
        )
        * (expected_max_input_scale - expected_min_input_scale)
    ) - (
        (expected_max_input_scale - expected_min_input_scale) / 2
    )  # Random inputs scaled to range of first attention inputs
    reference_output = reference_encoder_layer(
        hidden_states=random_inputs,
        attention_mask=None,
        output_attentions=False,
    )
    state_dict = convert_state_dict(reference_encoder_layer.state_dict())
    # state_dict = reference_encoder_layer.state_dict()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device()
    result = encoder_layer_func(
        mesh_device=mesh_device,
        hidden_states=random_inputs,
        state_dict=state_dict,
        state_dict_prefix="",
        weight_cache_path=None,
        vision_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        attention_dropout=0.0,
        attention_mask=None,
        hidden_act="gelu_pytorch_tanh",
    )
    ttnn.close_mesh_device(mesh_device)
    result, pcc = comp_pcc(reference_output[0], result)

    if result:
        print(f"✅ Siglip SDPA attention passes with PCC: {pcc}")
    else:
        print(f"❌ Siglip SDPA attention fails with PCC: {pcc}")
        assert False

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


### SDPA comparison
import os

import pytest
import torch
from transformers import AutoConfig
from transformers.models.siglip.modeling_siglip import SiglipSdpaAttention

import ttnn
from models.demos.siglip.compare import comp_pcc
from models.demos.siglip.reference.functional import siglip_attention
from models.demos.siglip.tt.attention import siglip_attention_ttnn


@pytest.mark.parametrize("attention_func", [siglip_attention, siglip_attention_ttnn])
def test_attention(attention_func):
    config = AutoConfig.from_pretrained(os.getenv("HF_MODEL"))
    assert hasattr(
        config, "vision_config"
    ), f"Unexpected model config provided. Expected a vision_config field to be present in: {config}"
    config = config.vision_config

    reference_attention = SiglipSdpaAttention(config=config)

    batch = 1
    seq_len = 4096
    expected_max_input_scale = 15
    expected_min_input_scale = -15

    random_inputs = (
        torch.rand(batch, seq_len, config.hidden_size, dtype=reference_attention.state_dict()["q_proj.weight"].dtype)
        * (expected_max_input_scale - expected_min_input_scale)
    ) - (
        (expected_max_input_scale - expected_min_input_scale) / 2
    )  # Random inputs scaled to range of first attention inputs
    reference_output = reference_attention(
        hidden_states=random_inputs,
        attention_mask=None,
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device()
    result = attention_func(
        mesh_device=mesh_device,
        hidden_states=random_inputs,
        state_dict=reference_attention.state_dict(),
        state_dict_prefix="",
        weight_cache_path=None,
        vision_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dropout=0.0,
        attention_mask=None,
    )
    ttnn.close_mesh_device(mesh_device)
    result, pcc = comp_pcc(reference_output[0], result[0])

    if result:
        print(f"✅ Siglip SDPA attention passes with PCC: {pcc}")
    else:
        print(f"❌ Siglip SDPA attention fails with PCC: {pcc}")
        assert False

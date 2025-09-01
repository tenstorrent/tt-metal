# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


### SDPA comparison
import os

import pytest
import torch
from transformers import AutoConfig
from transformers.models.siglip.modeling_siglip import SiglipAttention

import ttnn
from models.demos.siglip.compare import comp_pcc
from models.demos.siglip.reference.functional import siglip_attention
from models.demos.siglip.tests.common import convert_state_dict
from models.demos.siglip.tt.attention import siglip_attention_ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("attention_func", [siglip_attention, siglip_attention_ttnn])
def test_attention(mesh_device, attention_func, model_location_generator, is_ci_env):
    config = AutoConfig.from_pretrained(
        model_location_generator(model_version=os.getenv("HF_MODEL"), download_if_ci_v2=True)
        if is_ci_env
        else os.getenv("HF_MODEL")
    )
    assert hasattr(
        config, "vision_config"
    ), f"Unexpected model config provided. Expected a vision_config field to be present in: {config}"
    config = config.vision_config

    reference_attention = SiglipAttention(config=config)

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

    state_dict = convert_state_dict(reference_attention.state_dict())
    result = attention_func(
        mesh_device=mesh_device,
        hidden_states=random_inputs,
        state_dict=state_dict,
        state_dict_prefix="",
        weight_cache_path=None,
        vision_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dropout=0.0,
        attention_mask=None,
    )

    result, pcc = comp_pcc(reference_output[0], result[0])

    if result:
        print(f"✅ Siglip SDPA attention passes with PCC: {pcc}")
    else:
        print(f"❌ Siglip SDPA attention fails with PCC: {pcc}")
        assert False

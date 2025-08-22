# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.siglip.modeling_siglip import SiglipMLP

import ttnn
from models.demos.siglip.compare import comp_pcc
from models.demos.siglip.reference.functional import siglip_mlp
from models.demos.siglip.tests.common import convert_state_dict
from models.demos.siglip.tt.mlp import siglip_mlp_ttnn


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
@pytest.mark.parametrize("mlp_func", [siglip_mlp, siglip_mlp_ttnn])
def test_mlp(mesh_device, mlp_func, model_location_generator):
    config = AutoConfig.from_pretrained(
        model_location_generator(model_version=os.getenv("HF_MODEL"), download_if_ci_v2=True)
    )
    assert hasattr(
        config, "vision_config"
    ), f"Unexpected model config provided. Expected a vision_config field to be present in: {config}"
    config = config.vision_config

    reference_mlp = SiglipMLP(config=config)

    batch = 1
    seq_len = 4096
    expected_max_input_scale = 15
    expected_min_input_scale = -15

    random_inputs = (
        torch.rand(batch, seq_len, config.hidden_size, dtype=reference_mlp.state_dict()["fc1.weight"].dtype)
        * (expected_max_input_scale - expected_min_input_scale)
    ) - ((expected_max_input_scale - expected_min_input_scale) / 2)

    reference_output = reference_mlp(hidden_states=random_inputs)

    state_dict = convert_state_dict(reference_mlp.state_dict())
    result = mlp_func(
        mesh_device=mesh_device,
        hidden_states=random_inputs,
        state_dict=state_dict,
        state_dict_prefix="",
        weight_cache_path=None,
        vision_dim=config.hidden_size,
        vision_mlp_ratio=config.intermediate_size / config.hidden_size,
    )

    result, pcc = comp_pcc(reference_output, result)

    if result:
        logger.info(f"✅ Siglip MLP passes with PCC: {pcc}")
    else:
        logger.error(f"❌ Siglip MLP fails with PCC: {pcc}")
        assert False

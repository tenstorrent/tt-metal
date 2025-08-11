# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.experimental.mistral_24b.tt.pipeline.mistral_vision_tower import MistralVisionTower
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_mistral_vision_tower(mesh_device, use_program_cache, reset_seeds):
    pcc_required = 0.9999
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "vision_tower."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    B, C, H, W = 1, 3, model_args.vision_chunk_size, model_args.vision_chunk_size
    input_tensor = torch.randn((B, C, H, W))

    ##### Reference model output (Torch) #####
    reference_model = model_args.reference_vision_model()
    reference_model.load_state_dict(partial_state_dict)
    reference_output = reference_model(input_tensor, image_sizes=[(H, W)])

    reference_output = reference_output.last_hidden_state
    ##### TT Model: MistralVisionTower #####
    vision_model = MistralVisionTower(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=dtype,
        configuration=model_args,
    )

    submodule_partial_state_dict = {}
    reference_submodule = model_args.reference_vision_rot_emb()
    reference_submodule.load_state_dict(submodule_partial_state_dict)

    tt_output = vision_model(input_tensor)  # [0]
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output).squeeze(0)
    passing, pcc_message = comp_pcc(reference_output, tt_output)

    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}. {pcc_message}"

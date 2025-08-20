# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs
from models.experimental.mistral_24b.tt.pipeline.vision_model import TtMistralVisionTransformer
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


def get_image_features(vision_tower, projector, input_tensor, image_sizes):
    """
    Get image features from the vision tower and projector.
    """
    vision_token = vision_tower(input_tensor, image_sizes).last_hidden_state
    image_features = projector(vision_token.squeeze(0), image_sizes)
    return image_features


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
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_mistral_vision_model(mesh_device, reset_seeds):
    pcc_required = 0.97
    dtype = ttnn.bfloat8_b

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "vision_tower."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    ##### Reference model output (Torch) #####
    reference_model = model_args.reference_vision_model()
    reference_model.load_state_dict(partial_state_dict)

    mmp_first_layer_prefix = "multi_modal_projector."

    mmp_partial_state_dict = {
        k[len(mmp_first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(mmp_first_layer_prefix))
    }

    reference_mmp = model_args.reference_vision_multi_modal()
    reference_mmp.load_state_dict(mmp_partial_state_dict)

    B, C, H, W = 1, 3, model_args.vision_chunk_size, model_args.vision_chunk_size
    input_tensor = torch.rand((B, C, H, W), dtype=torch.bfloat16)

    reference_output = get_image_features(reference_model, reference_mmp, input_tensor, image_sizes=[(H, W)])

    # ##### TT Model: TtMistralVisionTransformer #####
    tt_ccl = TT_CCL(mesh_device=mesh_device)
    vision_model = TtMistralVisionTransformer(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=dtype,
        model_args=model_args,
    )

    tt_output = vision_model(input_tensor, image_sizes=[(H, W)])  # [0]
    tt_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[
        :, : tt_output.shape[-1]
    ]

    non_zero_indices = tt_output.ne(0).nonzero(as_tuple=True)
    tt_output = tt_output[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    passing, pcc_message = comp_pcc(reference_output, tt_output, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}. {pcc_message}"

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.gemma.gemma_vision_model import TtSiglipGemmaVisionModel
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
@pytest.mark.parametrize("bsz", [1])
def test_gemma_vision(
    mesh_device,
    reset_seeds,
    bsz,
):
    pcc_required = 0.94
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "model.vision_tower.vision_model."
    # partial_state_dict = {
    #     k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    # }

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    input_tensor = torch.rand((bsz, in_channels, image_size, image_size))

    reference_model = model_args.reference_vision_model()
    # reference_model.load_state_dict(partial_state_dict)
    reference_output = reference_model(input_tensor).last_hidden_state

    test_gemma_vision = TtSiglipGemmaVisionModel(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )

    test_output = test_gemma_vision(input_tensor)

    logger.info("Checking outputs")
    out = ttnn.from_device(test_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).squeeze(0)
    print("reference_output ", reference_output.shape)
    print("tt_output_torch ", tt_output_torch.shape)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

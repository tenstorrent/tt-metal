# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma3.tt.model_config import ModelArgs
from models.demos.gemma3.tt.siglip_vision_embedding import TtSiglipVisionEmbeddings
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from ttnn import ConcatMeshToTensor


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
def test_vision_embedding_integration(
    mesh_device,
    reset_seeds,
    bsz,
):
    pcc_required = 0.9999
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "model.vision_tower.vision_model.embeddings."

    image_size = model_args.vision_chunk_size
    patch_size = model_args.vision_patch_size
    hidden_dim = model_args.vision_dim
    dim = model_args.vision_dim
    in_channels = 3

    input_tensor = torch.randn((bsz, in_channels, image_size, image_size))

    reference_model = model_args.reference_vision_embedding()
    reference_output = reference_model(input_tensor)

    vision_embed = TtSiglipVisionEmbeddings(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=dtype,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=in_channels,
        hidden_dim=hidden_dim,
        bias=True,
    )

    embeddings = vision_embed(input_tensor)
    ##### Check the outputs #####
    logger.info("Checking outputs")
    out = ttnn.from_device(embeddings)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    # Only select output from one device
    tt_output_torch = tt_output_torch[..., :dim]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    # To get RTOL values
    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

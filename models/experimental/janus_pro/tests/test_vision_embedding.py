# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.experimental.janus_pro.tt.janus_pro_vision_embedding import TtJanusProVisionEmbeddings
from ttnn import ConcatMeshToTensor


@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
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

    first_layer_prefix = "model.vision_model.embeddings."

    image_size = model_args.vision_chunk_size
    patch_size = model_args.vision_patch_size
    hidden_dim = model_args.vision_dim
    in_channels = model_args.vision_in_channels

    input_tensor = torch.randn((bsz, in_channels, image_size, image_size))

    reference_model = model_args.reference_vision_embedding()
    reference_output = reference_model(input_tensor)

    vision_embed = TtJanusProVisionEmbeddings(
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

    # Weights are replicated across devices, so ConcatMeshToTensor(dim=-1) stacks an identical
    # per-device copy along the last dim; keep only the first (device 0) copy.
    tt_output_torch = tt_output_torch[..., :hidden_dim]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    # To get RTOL values
    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.embedding import VisionEmbedding
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
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
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize("interpolate_pos_encoding", [False, True])
def test_vision_embedding(max_seq_len, batch_size, mesh_device, interpolate_pos_encoding):
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()

    dtype = "bfloat16"
    if dtype == "bfloat16":
        pytorch_dtype = torch.bfloat16
        tt_dtype = ttnn.bfloat16

    pytorch_input = torch.empty(1, 3, 896, 896).uniform_(-1, 1)
    pytorch_input = pytorch_input.to(pytorch_dtype)

    pytorch_whole_model = model_args.cached_hf_model
    pytorch_vision_embeddings = pytorch_whole_model.model.vision_tower.vision_model.embeddings
    reference_output = pytorch_vision_embeddings(pytorch_input, interpolate_pos_encoding=interpolate_pos_encoding)

    tt_vision_embedding = VisionEmbedding(
        state_dict=state_dict,
        dtype=dtype,
        state_dict_prefix=None,
        image_size=None,
        patch_size=None,
        num_channels=None,
        bias=None,
        hidden_dim=None,
        mesh_device=None,
    )

    logger.info(f"reference_output: {reference_output.shape}")

    tt_input = ttnn.from_torch(
        pytorch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_output = tt_vision_embedding(tt_input)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=model_args.cluster_shape),
    )[:32].view(reference_output.shape)

    logger.info(f"tt_output_torch: {tt_output_torch.shape}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("embedding Passed!")
    else:
        logger.warning("embedding Failed!")

    assert passing, f"embedding output does not meet PCC requirement {0.99}."

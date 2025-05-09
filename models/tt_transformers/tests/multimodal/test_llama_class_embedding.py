# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_class_embedding import TtLlamaClassEmbedding
from models.utility_functions import comp_allclose, comp_pcc, nearest_32, skip_for_grayskull
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh


##### Torch op #####
class ClassEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

    def forward(self, x):
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        return x


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
    "bsz, num_concurrent_media, num_chunks",
    [
        ((1, 1, 4)),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
def test_class_embedding_inference(
    mesh_device,
    use_program_cache,
    reset_seeds,
    # Input params
    bsz,
    num_concurrent_media,
    num_chunks,
    layout,
    ensure_gc,
):
    dtype = ttnn.bfloat16
    pcc_required = 0.9999

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()
    first_layer_prefix = "vision_model.vision_encoder."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    ntok = nearest_32(model_args.vision_chunk_ntok - 1)
    dim = model_args.vision_dim

    ##### Prepare inputs #####
    input_tensor = torch.randn(bsz * num_concurrent_media * num_chunks, ntok, dim)
    logger.info(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    tt_input_tensor = ttnn.as_tensor(
        input_tensor.view(1, bsz * num_concurrent_media * num_chunks, ntok, dim),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    logger.info(f"TT Input tensor shape: {tt_input_tensor.shape}, dtype: {tt_input_tensor.dtype}")

    ##### Perform the torch ops #####
    reference_model = ClassEmbedding(
        width=dim,
    )
    reference_model.load_state_dict(partial_state_dict, strict=False)
    reference_output = reference_model(input_tensor)

    ##### Perform the TT ops #####
    tt_model = TtLlamaClassEmbedding(
        mesh_device,
        state_dict,
        first_layer_prefix,
        None,
        dtype,
        model_args,
    )
    tt_output = tt_model(tt_input_tensor)

    ##### Check the outputs #####
    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    # Only select output from one device
    tt_output_torch = tt_output_torch[..., :dim].view(reference_output.shape)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

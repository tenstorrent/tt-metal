# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

##### Python imports #####
import math
import pytest
from loguru import logger
import os
import itertools

##### PyTorch imports #####
import torch
import torch.nn.functional as F
import torch.nn as nn

##### TTNN imports #####
import ttnn
from ttnn import experimental as ttl
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh
from models.utility_functions import skip_for_grayskull
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import (
    nearest_32,
)
from models.demos.llama3.tt.llama_class_embedding import (
    TtLlamaClassEmbedding,
)
from models.demos.llama3.tt.model_config import TtModelArgs


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
        {"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_shape",
    [
        ((1, 4, 4, 1024, 1280)),
        ((1, 4, 4, 1024 + 1, 1280)),
        ((1, 4, 4, 1032, 1280)),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
def test_llama_class_embedding_inference(
    mesh_device,
    use_program_cache,
    reset_seeds,
    # Input params
    input_shape,
    layout,
    ensure_gc,
):
    dtype = ttnn.bfloat16
    pcc = 0.9999

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))
    first_layer_prefix = "vision_model.vision_encoder."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    (
        bsz,
        num_concurrent_media,
        num_chunks,
        ntok,
        dim,
    ) = input_shape

    ##### Prepare inputs #####
    input_tensor = torch.randn(bsz * num_concurrent_media * num_chunks, ntok, dim)
    logger.info(f"Input tensor shape: {input_tensor.shape}")

    tt_input_tensor = ttnn.as_tensor(
        input_tensor.view(1, bsz * num_concurrent_media * num_chunks, ntok, dim),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    logger.info(f"TT Input tensor shape: {tt_input_tensor.shape}")

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

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info(f"Llama_ClassEmbedding Passed!")
    else:
        logger.warning(f"Llama_ClassEmbedding Failed!")
        assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"

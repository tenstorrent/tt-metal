"""Gemma-3-4b-it Test for Vision MLP"""


# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs
from models.experimental.gemma3_4b.tt.gemma_image_mlp import TtGemmaImageFeedForward
from models.experimental.gemma3_4b.tests.references import reference_vision_mlp
from models.common.utility_functions import comp_allclose, comp_pcc, nearest_32, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch, num_chunks",
    ((1, 4),),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_inference(batch, num_chunks, mesh_device, reset_seeds):
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0, is_vision=True)

    # Add "encoder." after "visual." in first_layer_prefix
    if first_layer_prefix.startswith("visual."):
        first_layer_prefix = "visual.encoder." + first_layer_prefix[len("visual.") :]

    model_args.WEIGHTS_DTYPE = dtype

    dim = model_args.vision_dim
    seq_len = nearest_32(model_args.image_size) * num_chunks
    reference_model = reference_vision_mlp(model_args)
    # reference_model.load_state_dict(partial_state_dict)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtGemmaImageFeedForward(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
    )
    torch_input = torch.randn(1, batch, seq_len, dim)
    reference_output = reference_model(torch_input).squeeze()
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
        :, :1, :, :
    ].squeeze()

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import llama_models.llama3.reference_impl.multimodal.model as llama_reference_mod
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_image_mlp import TtLlamaImageFeedForward
from models.utility_functions import comp_allclose, comp_pcc, nearest_32, skip_for_grayskull


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
def test_mlp_inference(batch, num_chunks, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device)
    # state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder.transformer.resblocks.31.mlp."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    model_args.WEIGHTS_DTYPE = dtype

    dim = model_args.vision_dim
    seq_len = nearest_32(model_args.vision_chunk_ntok) * num_chunks
    mlp_ratio = model_args.vision_mlp_ratio
    act_layer = torch.nn.GELU
    dropout = 0.0
    reference_model = llama_reference_mod.ImageFeedForward(
        dim=dim,
        hidden_dim=int(mlp_ratio * dim),
        dropout=dropout,
        act_layer=act_layer,
    )
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtLlamaImageFeedForward(
        mesh_device=mesh_device,
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

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import torch
from loguru import logger
from transformers import AutoModelForVision2Seq
from transformers.models.mllama.image_processing_mllama import (
    convert_aspect_ratios_to_ids,
    get_all_supported_aspect_ratios,
)
from transformers.models.mllama.modeling_mllama import MllamaPrecomputedAspectRatioEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, nearest_32
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_tile_position_embedding import TtLlamaTilePositionEmbedding
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh


def load_partial_weights(weights_path, embedding_layer_prefix):
    partial_state_dict = {}
    model = AutoModelForVision2Seq.from_pretrained(weights_path, torch_dtype="auto", local_files_only=True)
    weights = model.state_dict()
    keys = weights.keys()
    for key in keys:
        if embedding_layer_prefix in key:
            key_name = "embedding.weight" if "weight" in key else "gate"
            partial_state_dict.update({key_name: weights[key]})
    return partial_state_dict


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
        (1, 1, 4),
        (1, 4, 4),
    ],
)
@pytest.mark.parametrize("pre_embed", [False, True])
def test_tile_position_emb_inference(
    mesh_device,
    reset_seeds,
    # Input params
    bsz,
    num_concurrent_media,
    num_chunks,
    pre_embed,
    ensure_gc,
):
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    gated = True
    pcc_required = 0.9999

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # TT models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder." + (
        "pre_tile_pos_embed." if pre_embed else "post_tile_pos_embed."
    )
    embedding_layer_prefix = "pre_tile_positional_embedding" if pre_embed else "post_tile_positional_embedding"

    ntok = nearest_32(model_args.vision_chunk_ntok - (0 if pre_embed else 1))
    dim = model_args.vision_dim
    max_num_tiles = model_args.vision_max_num_tiles

    ##### Check parms #####
    assert num_chunks == max_num_tiles, "num_chunks must be the same value as max_num_tiles!"

    ##### Prepare inputs #####
    input_tensor = torch.randn(bsz * num_concurrent_media, num_chunks, ntok, dim)
    logger.info(f"Input tensor shape: {input_tensor.shape}")

    tt_input_tensor = ttnn.as_tensor(
        input_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    logger.info(f"TT Input tensor shape: {tt_input_tensor.shape}")

    # Generate all possible aspect ratios (H * W must be less than or equal to max_num_tiles)
    aspect_ratios = list(itertools.product(range(1, max_num_tiles + 1), repeat=2))
    aspect_ratios = [x for x in aspect_ratios if x[0] * x[1] <= max_num_tiles]

    # Repeat the aspect ratios to match the batch size
    if len(aspect_ratios) < bsz * num_concurrent_media:
        aspect_ratios = aspect_ratios * (bsz * num_concurrent_media // len(aspect_ratios) + 1)

    aspect_ratios = torch.tensor(aspect_ratios[: bsz * num_concurrent_media], dtype=torch.int64)
    logger.info(f"Aspects ratios shape: {aspect_ratios.shape}")

    tt_aspect_ratios = aspect_ratios.tolist()

    ##### Perform the torch ops #####
    supported_aspect_ratios = get_all_supported_aspect_ratios(max_num_tiles)

    # subclass MllamaPrecomputedAspectRatioEmbedding expects parameters in the following format
    class Config:
        def __init__(
            self,
            max_num_tiles=max_num_tiles,
            hidden_size=dim,
            max_aspect_ratio_id=len(supported_aspect_ratios),
            is_gated=gated,
        ):
            self.max_num_tiles = max_num_tiles
            self.hidden_size = hidden_size
            self.max_aspect_ratio_id = max_aspect_ratio_id
            self.is_gated = is_gated

    # partial loading of HF safetensors to match model graph expected dimensionality of the loaded weights
    partial_state_dict = load_partial_weights(os.getenv("HF_MODEL"), embedding_layer_prefix)
    reference_model = MllamaPrecomputedAspectRatioEmbedding(Config())
    reference_model.load_state_dict(partial_state_dict)
    # HF tricky part the aspect ratios are mapped to integer values and these are used to draw the correct embedding vector
    aspect_ratios_id = torch.from_numpy(convert_aspect_ratios_to_ids(aspect_ratios.unsqueeze(0), max_num_tiles))
    reference_output = reference_model(input_tensor, aspect_ratios_id)

    ##### Perform the TT ops #####
    tt_model = TtLlamaTilePositionEmbedding(
        mesh_device,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=dtype,
        num_tiles=max_num_tiles,
        width=dim,
        gated=gated,
    )
    tt_output = tt_model(tt_input_tensor, tt_aspect_ratios)

    ##### Check the outputs #####
    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    # Only select output from one device
    tt_output_torch = tt_output_torch[..., :dim]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

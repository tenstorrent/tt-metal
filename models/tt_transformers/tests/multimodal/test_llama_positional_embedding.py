# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForVision2Seq
from transformers.models.mllama.image_processing_mllama import convert_aspect_ratios_to_ids
from transformers.models.mllama.modeling_mllama import MllamaPrecomputedPositionEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_positional_embedding import TtLlamaPositionalEmbedding
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh


def load_partial_weights(weights_path, embedding_layer_prefix):
    partial_state_dict = {}
    model = AutoModelForVision2Seq.from_pretrained(
        weights_path, torch_dtype="auto", local_files_only=os.getenv("CI") == "true"
    )
    weights = model.state_dict()
    keys = weights.keys()
    for key in keys:
        if embedding_layer_prefix in key:
            # Caution it may cause potential failures. In future versions and different formats the below prefix may change
            key_name = key[len("model.vision_model.gated_positional_embedding.") :]
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
    [(1, 4, 4)],
)
def test_positional_embedding_inference(
    mesh_device,
    reset_seeds,
    # Input params
    bsz,
    num_concurrent_media,
    num_chunks,
    ensure_gc,
):
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    pcc_required = 0.9999

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()
    first_layer_prefix = "vision_model.vision_encoder."

    ntok = model_args.vision_chunk_ntok
    dim = model_args.vision_dim

    ##### Check parms #####
    max_num_tiles = model_args.vision_max_num_chunks
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

    tt_input_tensor = ttnn.to_layout(tt_input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    tt_input_tensor = ttnn.reshape(tt_input_tensor, (bsz * num_concurrent_media, num_chunks, ntok, dim))
    tt_input_tensor = ttnn.to_layout(tt_input_tensor, ttnn.TILE_LAYOUT)
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

    # config contains paramters for the whole multimodal network the subeset of vision branch is chosen instead
    config = AutoConfig.from_pretrained(os.getenv("HF_MODEL"))
    reference_model = MllamaPrecomputedPositionEmbedding(config.vision_config)
    # partial loading of HF safetensors to match model graph expected dimensionality of the loaded weights
    partial_state_dict = load_partial_weights(os.getenv("HF_MODEL"), "gated_positional")
    reference_model.load_state_dict(partial_state_dict)
    # HF tricky part the aspect ratios are mapped to integer values and these are used to draw the correct embedding vector
    aspect_ratios_id = torch.from_numpy(convert_aspect_ratios_to_ids(aspect_ratios.unsqueeze(0), max_num_tiles))
    reference_output = reference_model(input_tensor, aspect_ratios_id)

    ##### Perform the TT ops #####
    tt_model = TtLlamaPositionalEmbedding(
        mesh_device,
        state_dict,
        first_layer_prefix,
        None,
        dtype,
        model_args,
    )
    tt_output = tt_model(tt_input_tensor, tt_aspect_ratios)

    ##### Check the outputs #####
    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    # Only select output from one device
    tt_output_torch = tt_output_torch[..., :dim]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

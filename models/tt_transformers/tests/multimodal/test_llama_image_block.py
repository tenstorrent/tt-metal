# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoConfig, AutoModelForVision2Seq
from transformers.models.mllama.modeling_mllama import MllamaVisionEncoderLayer

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import build_encoder_attention_mask
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_image_block import TtLlamaImageTransformerBlock
from models.tt_transformers.tt.multimodal.llama_vision_encoder import mask_tile_padding, pad_seq_one_tile


def load_partial_weights(weights_path, layer_prefix):
    partial_state_dict = {}
    model = AutoModelForVision2Seq.from_pretrained(
        weights_path, torch_dtype="auto", local_files_only=os.getenv("CI") == "true"
    )
    weights = model.state_dict()
    keys = weights.keys()
    for key in keys:
        if layer_prefix in key:
            # Caution it may cause potential failures. In future versions and different formats the below prefix may change
            key_name = key[len(layer_prefix) :]
            partial_state_dict.update({key_name: weights[key]})
    return partial_state_dict


def expand_num_tokens_to_mult8(tensor):
    num_padding_patches = (8 - (tensor.shape[-2] % 8)) % 8
    # Compute padding tuple for pad function
    padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
    # Pad the tensor
    tensor = F.pad(tensor, padding, mode="constant", value=0)
    slice_index = -num_padding_patches if num_padding_patches > 0 else 0
    return tensor, slice_index


def contract_num_tokens_from_mult8(tensor, slice_index):
    if slice_index == 0:
        return tensor
    return tensor[:, :, :slice_index, :]


@pytest.mark.parametrize(
    "batch, num_chunks",
    ((1, 4),),
)
@pytest.mark.parametrize(
    "gated",
    (True, False),
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
def test_block_inference(batch, num_chunks, mesh_device, gated, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    if gated:
        first_layer_prefix = "vision_model.vision_encoder.global_transformer.resblocks.0."
        hf_layer_prefix = "model.vision_model.global_transformer.layers.0."
    else:
        first_layer_prefix = "vision_model.vision_encoder.transformer.resblocks.31."
        hf_layer_prefix = "model.vision_model.transformer.layers.31."

    dim = model_args.vision_dim
    ntok = model_args.vision_chunk_ntok

    model_repo_name = os.getenv("HF_MODEL")
    # config contains paramters for the whole multimodal network the subeset of vision branch is chosen instead
    config = AutoConfig.from_pretrained(model_repo_name)
    config.vision_config._attn_implementation = "sdpa"
    reference_model = MllamaVisionEncoderLayer(config.vision_config, is_gated=gated)
    # partial loading of HF safetensors to match model graph expected dimensionality of the loaded weights
    partial_state_dict = load_partial_weights(model_repo_name, hf_layer_prefix)
    reference_model.load_state_dict(partial_state_dict)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtLlamaImageTransformerBlock(
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        gated=gated,
    )

    # Create PT input
    ar = torch.tensor([[1, 2]])
    pt_block_input = (torch.rand(batch, num_chunks, ntok, dim) * 2) - 1
    tt_attention_input = pt_block_input.clone()
    # Do PT padding
    pt_block_input, slice_index = expand_num_tokens_to_mult8(pt_block_input)
    # Create PT attention mask
    mask = build_encoder_attention_mask(pt_block_input, ar, ntok, num_chunks, 1)
    pt_block_input = pt_block_input.reshape(batch, -1, dim)

    attention_input = model_args.prepare_residual_tensor_prefill(
        tt_attention_input.view(num_chunks, ntok, dim),
        force_replicated=True,
    )
    # Pad TT input to multipple of 32
    attention_input, npadtt = pad_seq_one_tile(attention_input, mesh_device)
    # Create attention mask, assuming padding of 32
    fake_x = torch.zeros(
        attention_input.shape[0], attention_input.shape[1], attention_input.shape[2], attention_input.shape[3]
    )
    tt_attn_mask = build_encoder_attention_mask(fake_x, ar, ntok, num_chunks, 1)
    tt_attn_mask = mask_tile_padding(tt_attn_mask, ntok, npadtt, num_chunks)
    attention_input = attention_input.reshape(1, batch, -1, dim)

    tt_mask = ttnn.from_torch(
        tt_attn_mask,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_model(attention_input, mask=tt_mask)
    tt_out = tt_out.reshape(batch, num_chunks, ntok + npadtt, dim)
    tt_out = ttnn.slice(tt_out, (0, 0, 0, 0), (batch, num_chunks, ntok, dim))
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :, :]

    reference_output = reference_model(pt_block_input, attention_mask=mask)[0]
    reference_output = reference_output.reshape(batch, num_chunks, ntok - slice_index, dim)
    reference_output = contract_num_tokens_from_mult8(reference_output, slice_index)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

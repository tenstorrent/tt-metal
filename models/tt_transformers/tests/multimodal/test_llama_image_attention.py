# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import llama_models.llama3.reference_impl.multimodal.model as llama_reference_mod
import pytest
import torch
from llama_models.llama3.reference_impl.multimodal import encoder_utils
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_image_attention import TtLlamaImageAttention
from models.tt_transformers.tt.multimodal.llama_vision_encoder import mask_tile_padding, pad_seq_one_tile
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


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
def test_attention_inference(batch, num_chunks, mesh_device, use_program_cache, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder.transformer.resblocks.0.attn."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.vision_dim
    heads = model_args.vision_attn_n_heads
    ntok = model_args.vision_chunk_ntok
    reference_model = llama_reference_mod.ImageAttention(dim=dim, head_dim=dim // heads, n_heads=heads)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtLlamaImageAttention(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
    )

    # Create PT input
    ar = torch.tensor([[1, 2]])
    pt_block_input = (torch.rand(batch, num_chunks, ntok, dim) * 2) - 1
    tt_attention_input = pt_block_input.clone()
    # Do PT padding
    pt_block_input, npad = encoder_utils.expand_num_tokens_to_mult8(pt_block_input)
    # Create PT attention mask
    mask = encoder_utils.build_encoder_attention_mask(pt_block_input, ar, ntok, num_chunks, 1)
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
    tt_attn_mask = encoder_utils.build_encoder_attention_mask(fake_x, ar, ntok, num_chunks, 1)
    # Make striped attention mask to mask out our padding between 8 and 32
    # Striped mask doesn't affect PCC on first layer but is necessary for later layers
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

    # Doing contract in tt is correct!!
    tt_out = tt_out.reshape(batch, num_chunks, ntok + npadtt, dim)
    tt_out = ttnn.slice(tt_out, (0, 0, 0, 0), (batch, num_chunks, ntok, dim))
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :, :]

    reference_output = reference_model(pt_block_input, mask=mask)
    reference_output = reference_output.reshape(batch, num_chunks, ntok + npad, dim)
    reference_output = encoder_utils.contract_num_tokens_from_mult8(reference_output, npad)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

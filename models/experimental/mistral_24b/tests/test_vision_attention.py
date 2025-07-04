# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull

from models.tt_transformers.tt.multimodal.llama_image_attention import TtLlamaImageAttention


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
    "seq_len",
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_vision_attention(mesh_device, seq_len, batch_size):
    logger.info(f"seq_len: {seq_len}, batch_size: {batch_size}")
    dtype = ttnn.bfloat8_b

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "vision_tower.transformer.layers.0.attention."

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = model_args.reference_vision_attention()
    reference_model.load_state_dict(partial_state_dict)

    hidden_size = model_args.vision_dim
    n_heads = model_args.vision_attn_n_heads
    head_dim = hidden_size // n_heads

    tt_model = TtLlamaImageAttention(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
    )

    dim = model_args.vision_dim
    pt_attention_input = torch.randn(batch_size, seq_len, dim)

    B, T, D = pt_attention_input.shape
    cos = torch.ones((1, T, head_dim))
    sin = torch.zeros((1, T, head_dim))

    attention_input = model_args.prepare_residual_tensor_prefill(
        pt_attention_input,
        force_replicated=True,
    )

    tt_out = tt_model(attention_input)
    tt_output_torch = ttnn.to_torch(tt_out, device=mesh_device)[0, :, :, :]

    reference_output = reference_model(pt_attention_input, position_embeddings=(cos, sin))[0]
    pcc_required = 0.99

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

"""Gemma-3-4b-it Test for Vision Attention"""


# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import (  # convert_vision_hf_to_meta,
    convert_hf_qkv_to_meta_format,
    convert_vision_hf_to_meta,
)
from models.tt_transformers.tt.model_config import ModelArgs

from models.experimental.gemma3_4b.tt.gemma_image_attention import TtGemmaImageAttention
from models.experimental.gemma3_4b.tests.references import reference_vision_attention
from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


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
def test_attention_inference(batch, num_chunks, mesh_device, reset_seeds):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "visual.encoder.layers.0.attn."
    # partial_state_dict = {
    #     k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    # }

    dim = model_args.vision_dim

    reference_model = reference_vision_attention(model_args)
    # reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    hidden_size = model_args.vision_dim
    n_heads = model_args.vision_attn_n_heads
    head_dim = hidden_size // n_heads
    seq_len = model_args.image_size

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtGemmaImageAttention(
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
    )

    pt_attention_input = torch.randn(batch, seq_len, dim)

    attention_input = model_args.prepare_residual_tensor_prefill(
        pt_attention_input,
        force_replicated=True,
    )

    tt_out = tt_model(attention_input)

    # Doing contract in tt is correct!!
    tt_output_torch = ttnn.to_torch(tt_out, device=mesh_device)[0, :, :, :]

    reference_output = reference_model(pt_attention_input)[0]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

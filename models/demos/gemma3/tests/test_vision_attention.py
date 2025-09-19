# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from models.demos.gemma3.tt.gemma_image_attention import TtGemmaImageAttention
from models.demos.gemma3.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


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
    first_layer_prefix = "model.vision_tower.vision_model.encoder.layers.0.attn."

    dim = model_args.vision_dim

    reference_model = model_args.reference_vision_attention()
    reference_model.eval()

    hidden_size = model_args.vision_dim
    n_heads = model_args.vision_attn_n_heads
    head_dim = hidden_size // n_heads
    seq_len = model_args.vision_chunk_ntok

    pt_attention_input = torch.rand(batch, seq_len, dim)

    # Create the attention model
    tt_attention = TtGemmaImageAttention(
        mesh_device=mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=None,
        dtype=dtype,
        configuration=model_args,
    )

    reference_output = reference_model(pt_attention_input, attention_mask=None)

    # Extract tensor from tuple if needed (reference model returns (tensor, None))
    if isinstance(reference_output, tuple):
        reference_output = reference_output[0]

    tt_output = tt_attention(pt_attention_input, attention_mask=None)

    tt_output_torch = tt_output

    # Trim to match sequence length if needed because of padding
    if tt_output_torch.shape[1] > seq_len:
        tt_output_torch = tt_output_torch[:, :seq_len, :]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma3.tt.gemma_image_block import TtGemmaImageTransformerBlock
from models.demos.gemma3.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
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
def test_block_inference(batch, num_chunks, mesh_device, reset_seeds, gated):
    dtype = ttnn.bfloat16
    pcc_required = 0.99
    gated = False

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    if gated:
        first_layer_prefix = "model.vision_tower.vision_model.encoder.layers.0."
    else:
        first_layer_prefix = "model.vision_tower.vision_model.encoder.layers.0."

    dim = model_args.vision_dim
    heads = model_args.vision_attn_n_heads
    seq_len = model_args.vision_chunk_ntok - 1

    reference_model = model_args.reference_vision_encoder_block()
    reference_model.eval()

    tt_model = TtGemmaImageTransformerBlock(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        gated=gated,
    )

    pt_attention_input = torch.randn(batch, seq_len, dim)
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len)

    attention_input = model_args.prepare_residual_tensor_prefill(
        pt_attention_input,
        force_replicated=True,
    )
    tt_mask = ttnn.from_torch(
        attention_mask,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_model(attention_input, mask=tt_mask)
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :, :]

    reference_output = reference_model(pt_attention_input, attention_mask=attention_mask)[0]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma3.tt.model_config import ModelArgs
from models.demos.gemma3.tt.multi_modal_projector import TtGemma3MultiModalProjector
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("device"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (1152,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_multi_modal_inference(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    reference_model = tt_model_args.reference_vision_multi_modal()

    # create input tensor for multi_modal_projector layer
    patches_per_image = 64
    num_patches = patches_per_image * patches_per_image
    input = torch.randn((batch_size, num_patches, seq_len))
    reference_output = reference_model(input)

    # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
    tt_input = ttnn.from_torch(
        input,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(None, -1), mesh_shape=tt_model_args.cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_model = TtGemma3MultiModalProjector(
        mesh_device=device,
        state_dict=state_dict,
        state_dict_prefix="model.multi_modal_projector",
        image_size=tt_model_args.vision_chunk_size,
        patch_size=tt_model_args.vision_patch_size,
        hidden_size=tt_model_args.vision_hidden_dim,
        mm_tokens_per_image=tt_model_args.mm_tokens_per_image,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        layer_norm_eps=1e-06,  # layer_norm_eps
        dtype=dtype,
        configuration=tt_model_args,
    )
    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    pcc_required = 0.9999
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

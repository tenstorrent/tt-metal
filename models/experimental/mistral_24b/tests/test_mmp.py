# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull

from models.experimental.mistral_24b.tt.vision_mmp import TTMistral3MultiModalProjector


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
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_multi_modal_inference(seq_len, batch_size, reset_seeds, device):
    print("device:", device)
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
    # print(reference_model)
    first_layer_prefix = "multi_modal_projector."

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model.load_state_dict(partial_state_dict)
    # create input tensor for multi_modal_projector layer
    batch_size = 1
    seq_length = 1152
    patches_per_image = 64
    num_patches = patches_per_image * patches_per_image
    input = torch.randn((1656, 1024))  # image_features: torch.Size([1656, 1024])

    image_size = torch.tensor([[504, 644]], dtype=torch.int32)

    reference_output = reference_model(input, image_size)
    print("reference_output:", reference_output.shape)

    # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
    tt_input = ttnn.from_torch(
        input,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(None, -1), mesh_shape=tt_model_args.cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_image_size = ttnn.from_torch(
        image_size,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print("state_dict ", state_dict.keys())
    tt_model = TTMistral3MultiModalProjector(
        mesh_device=device,
        args=tt_model_args,
        state_dict=state_dict,
        state_dict_prefix="multi_modal_projector.",
        dtype=dtype,
        eps=1e-06,  # layer_norm_eps
    )

    # print("tt_input:", tt_input.memory_config())

    tt_output = tt_model(tt_input, tt_image_size)

    output_torch = ttnn.to_torch(tt_output)

    print("output_torch:", output_torch.shape)
    # # transpose output from NHWC to NCHW
    # output_torch = output_torch.permute(0, 2, 1)
    passing, pcc_message = comp_pcc(reference_output, output_torch)
    pcc_required = 0.999
    logger.info(comp_allclose(reference_output, output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from models.demos.gemma3.tt.gemma_conv2d_patch import TtGemmaConv2dPatch
from models.demos.gemma3.tt.model_config import ModelArgs
from ttnn import ConcatMeshToTensor


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
def test_conv2d_inference(
    mesh_device,
    reset_seeds,
):
    pcc_required = 0.9999
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    tt_layer_prefix = "model.vision_tower.vision_model.embeddings.patch_embedding."
    first_layer_prefix = "model.vision_tower.vision_model.embeddings.patch_embedding._linear."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    num_devices = model_args.num_devices

    B, NCH, H, W = (1, 3, model_args.vision_chunk_size, model_args.vision_chunk_size)
    in_channels, out_channels, kernel_size, stride, bias = (
        3,
        model_args.vision_dim,
        model_args.vision_patch_size,
        model_args.vision_patch_size,
        True,
    )

    assert NCH == in_channels, "Number of channels in input tensor should match in_channels for the Conv2d patch."
    assert type(kernel_size) == int, "Only symmetric kernel_size is currently supported."
    assert kernel_size == stride, "Only same kernel_size and stride are currently supported."

    assert H % kernel_size == 0, "Height should be divisible by kernel_size."
    assert W % kernel_size == 0, "Width should be divisible by kernel_size."

    input_tensor = torch.randn((B, NCH, H, W))

    ##### Perform the torch ops #####
    reference_model = model_args.reference_siglip_patch_embed()
    # reference_model.load_state_dict(partial_state_dict)
    reference_output = reference_model(input_tensor)
    del reference_model

    tt_model = TtGemmaConv2dPatch(
        mesh_device,
        state_dict,
        tt_layer_prefix,
        dtype,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias,
    )
    tt_output = tt_model(input_tensor)

    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    tt_output_torch = tt_output_torch[0, ..., :out_channels]

    logger.info(f"Reference output shape: {reference_output.shape}")
    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # TT output: [B, HW, C]
    B, HW, C = tt_output_torch.shape
    H = W = int(HW**0.5)
    assert H * W == HW, "HW is not a perfect square — can't reshape"
    tt_output_torch = tt_output_torch.permute(0, 2, 1)
    tt_output_torch = tt_output_torch.reshape(B, C, H, W)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

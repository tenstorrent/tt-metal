# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.experimental.mistral_24b.tt.vision_conv2d import TtMistralConv2dPatch
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
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
    use_program_cache,
    reset_seeds,
):
    pcc_required = 0.9999
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_tower.patch_conv."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    num_devices = model_args.num_devices

    ##### Create input tensor for the all gather #####
    B, NCH, H, W = (1, 3, model_args.vision_chunk_size, model_args.vision_chunk_size)
    in_channels, out_channels, kernel_size, stride, bias = (
        3,
        model_args.vision_dim,
        model_args.vision_patch_size,
        model_args.vision_patch_size,
        False,
    )

    assert NCH == in_channels, "Number of channels in input tensor should match in_channels for the Conv2d patch."
    assert type(kernel_size) == int, "Only symmetric kernel_size is currently supported."
    assert kernel_size == stride, "Only same kernel_size and stride are currently supported."

    assert H % kernel_size == 0, "Height should be divisible by kernel_size."
    assert W % kernel_size == 0, "Width should be divisible by kernel_size."

    ##### Prepare inputs #####
    input_tensor = torch.randn((B, NCH, H, W))
    logger.info(f"Input tensor shape: {input_tensor.shape}")

    ##### Perform the torch ops #####
    # reference_model = llama_reference_mod.ColumnParallelConv2dPatch(
    #     in_channels=in_channels,
    #     out_channels=out_channels,
    #     kernel_size=kernel_size,
    #     stride=stride,
    #     bias=bias,
    # )
    reference_model = model_args.reference_conv2d_patch()
    reference_model.load_state_dict(partial_state_dict)
    reference_output = reference_model(input_tensor)

    tt_model = TtMistralConv2dPatch(
        mesh_device,
        state_dict,
        first_layer_prefix,
        dtype,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias,
    )
    tt_output = tt_model(input_tensor)

    ##### Check the outputs #####
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=2))

    # Only select output from one device
    tt_output_torch = tt_output_torch[0, ..., :out_channels]

    # 1. Restore batch dim
    tt_output_torch = tt_output_torch.unsqueeze(0)
    # 1 1024 4096
    # 2. Permute to match Conv2D output: (N, C_out, H_out, W_out)
    tt_output_torch = tt_output_torch.permute(0, 2, 1).reshape(1, 1024, 64, 64)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

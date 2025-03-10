# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

##### Python imports #####
import pytest
from loguru import logger
import os

##### PyTorch imports #####
import torch
import torch.nn.functional as F

##### TTNN imports #####
import ttnn
from ttnn import experimental as ttl
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh
from models.utility_functions import skip_for_grayskull
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.tt_transformers.tt.multimodal.llama_conv2d_patch import (
    TtLlamaConv2dPatch,
)

from models.tt_transformers.tt.model_config import ModelArgs
import llama_models.llama3.reference_impl.multimodal.model as llama_reference_mod


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
    ensure_gc,
):
    pcc_required = 0.9999
    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)

    model_args = ModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder.conv1."
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
    reference_model = llama_reference_mod.ColumnParallelConv2dPatch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
    )
    reference_model.load_state_dict(partial_state_dict)
    reference_output = reference_model(input_tensor)

    tt_model = TtLlamaConv2dPatch(
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
    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    # Only select output from one device
    tt_output_torch = tt_output_torch[0, ..., :out_channels]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"

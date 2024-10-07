# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

##### Python imports #####
import math
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
from models.utility_functions import (
    nearest_32,
)
from models.demos.wormhole.llama31_8b_N300.tt.llama_conv2d_patch import (
    TtLlamaConv2dPatch,
)


##### Torch op #####
class Conv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = torch.nn.Linear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x = F.linear(x, self._linear.weight)
        return x


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_shape, in_channels, out_channels, kernel_size, stride, bias",
    [
        ((1, 3, 32 * 32, 32 * 32), 3, 512, 32, 32, True),
        ((1, 3, 32 * 32, 32 * 32), 3, 512, 32, 32, False),
        ((1, 3, 14 * 32, 14 * 32), 3, 1280, 14, 14, True),  # Llama3.2 case
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
def test_llama_conv2d_inference(
    mesh_device,
    use_program_cache,
    reset_seeds,
    # Input params
    input_shape,
    input_dtype,
    layout,
    # Conv2d patch params
    in_channels,
    out_channels,
    kernel_size,
    stride,
    bias,
):
    pcc = 0.9999

    devices = mesh_device.get_devices()
    num_devices = len(devices)

    ##### Create input tensor for the all gather #####
    B, NCH, H, W = input_shape

    assert NCH == in_channels, "Number of channels in input tensor should match in_channels for the Conv2d patch."
    assert type(kernel_size) == int, "Only symmetric kernel_size is currently supported."
    assert kernel_size == stride, "Only same kernel_size and stride are currently supported."

    assert H % kernel_size == 0, "Height should be divisible by kernel_size."
    assert W % kernel_size == 0, "Width should be divisible by kernel_size."

    ##### Prepare inputs #####
    input_tensor = torch.randn(input_shape)
    logger.info(f"Input tensor shape: {input_tensor.shape}")

    ##### Perform the torch ops #####
    reference_model = Conv2dPatch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
    )
    reference_output = reference_model(input_tensor)

    ##### Perform the TT ops #####
    _input_config = {
        "dtype": input_dtype,
        "layout": layout,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "mesh_mapper": ttnn.ReplicateTensorToMesh,
    }

    _linear_config = {
        "dtype": ttnn.bfloat8_b,
        "layout": layout,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "mesh_mapper": ttnn.ReplicateTensorToMesh,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    }

    tt_model = TtLlamaConv2dPatch(
        mesh_device,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        reference_model._linear.weight,
        reference_model._linear.bias,
        input_config=_input_config,
        linear_config=_linear_config,
    )
    tt_output = tt_model(input_tensor)

    ##### Check the outputs #####
    print("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    # Only select output from one device
    tt_output_torch = tt_output_torch[0, ..., :out_channels]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info(f"Llama_Conv2dPatch Passed!")
    else:
        logger.warning(f"Llama_Conv2dPatch Failed!")
        assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"

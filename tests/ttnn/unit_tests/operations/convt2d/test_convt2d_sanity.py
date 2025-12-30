# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import math


class DeviceGetter:
    _instance = None
    _mesh_shape = None
    l1_small_size = 1 << 15

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    def __del__(self):
        if self._instance is not None:
            ttnn.close_mesh_device(self._instance)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    @classmethod
    def get_device(cls, mesh_shape):
        if cls._instance == None:
            if (
                not isinstance(mesh_shape, (list, tuple))
                or len(mesh_shape) == 0
                or not all(isinstance(x, int) and x > 0 for x in mesh_shape)
            ):
                raise ValueError(f"mesh_shape must be a non-empty list or tuple of positive integers, got {mesh_shape}")
            cls._mesh_shape = mesh_shape

            if math.prod(mesh_shape) >= 2:
                ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            cls._instance = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(mesh_shape),
                l1_small_size=cls.l1_small_size,
            )
            print(f"Device: {cls._instance}")

        # Compare requested mesh_shape with _mesh_shape used to initialize the device
        if tuple(cls._mesh_shape) != tuple(mesh_shape):
            raise ValueError(
                f"Device already initialized with mesh_shape={cls._mesh_shape}, but got mesh_shape={mesh_shape}"
            )

        return cls._instance


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("in_channels", [128])
@pytest.mark.parametrize("out_channels", [128])
@pytest.mark.parametrize("input_height", [124])
@pytest.mark.parametrize("input_width", [108])
@pytest.mark.parametrize("kernel_size", [[2, 2]])
@pytest.mark.parametrize("stride", [[2, 2]])
@pytest.mark.parametrize("padding", [[0, 0]])
@pytest.mark.parametrize("output_padding", [[0, 0]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("groups", [1])
def test_convt2d_sanity(
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
):
    torch.manual_seed(0)

    # Create torch input tensor: [batch_size, in_channels, input_height, input_width]
    torch_input_tensor = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.float32)

    # Create torch weight tensor: [in_channels, out_channels, kernel_h, kernel_w]
    torch_weight_tensor = torch.randn((in_channels, out_channels, kernel_size[0], kernel_size[1]), dtype=torch.float32)

    # Preprocess the input tensor
    # permute: [0, 2, 3, 1] -> [batch_size, input_height, input_width, in_channels]
    torch_input_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    # reshape: [1, 1, batch_size*input_height*input_width, in_channels]
    total_elements = batch_size * input_height * input_width
    torch_input_reshaped = torch.reshape(torch_input_permuted, (1, 1, total_elements, in_channels))

    device = DeviceGetter.get_device((1, 1))

    # Convert to ttnn tensors
    input_tensor = ttnn.from_torch(
        torch_input_reshaped, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )

    weight_tensor = ttnn.from_torch(
        torch_weight_tensor, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.ROW_MAJOR, device=None
    )

    # ttnn conv_transpose2d
    output_tensor = ttnn.conv_transpose2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        dtype=ttnn.DataType.FLOAT32,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

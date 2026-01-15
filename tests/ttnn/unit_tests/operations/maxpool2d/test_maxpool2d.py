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


@pytest.mark.parametrize("batch_size", [6])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("input_h", [464])
@pytest.mark.parametrize("input_w", [800])
@pytest.mark.parametrize("kernel_size", [[3, 3]])
@pytest.mark.parametrize("stride", [[2, 2]])
@pytest.mark.parametrize("padding", [[1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("ceil_mode", [False])
def test_maxpool2d_sanity(
    batch_size, channels, input_h, input_w, kernel_size, stride, padding, dilation, ceil_mode
):
    torch.manual_seed(0)
    device = DeviceGetter.get_device((1, 1))

    # Create torch input tensor
    torch_input_tensor = torch.randn((batch_size, channels, input_h, input_w), dtype=torch.float32)

    # Preprocess the tensor
    # permute: [0, 2, 3, 1] -> [batch_size, input_h, input_w, channels]
    torch_input_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    # reshape: [1, 1, batch_size*input_h*input_w, channels]
    total_elements = batch_size * input_h * input_w
    torch_input_reshaped = torch.reshape(torch_input_permuted, (1, 1, total_elements, channels))

    # Convert to ttnn tensor
    input_tensor = ttnn.from_torch(
        torch_input_reshaped, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=device
    )

    # ttnn max_pool2d
    output_tensor = ttnn.max_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        applied_shard_scheme=None,
        reallocate_halo_output=False,
    )

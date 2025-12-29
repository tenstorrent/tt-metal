# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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


@pytest.mark.parametrize(
    "shape",
    [
        [1, 256, 7, 25281],
    ],
)
def test_typecast(shape):
    fill_value = 0

    # Create input tensor using torch with INT32 dtype
    torch_input = torch.full(shape, fill_value, dtype=torch.int32)

    # Convert torch tensor to ttnn tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=DeviceGetter.get_device((1, 1)),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # Perform typecast from INT32 to FLOAT32
    output_tensor = ttnn.typecast(
        input_tensor,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

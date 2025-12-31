# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


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


def test_ttnn_concat():
    device = DeviceGetter.get_device((1, 1))

    torch_input_tensor_0 = torch.rand((4096, 4096), dtype=torch.bfloat16)
    torch_input_tensor_1 = torch.rand((4096, 4096), dtype=torch.bfloat16)
    torch_input_tensor_2 = torch.rand((4096, 4096), dtype=torch.bfloat16)

    input_tensor_0 = ttnn.from_torch(torch_input_tensor_0, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_1 = ttnn.from_torch(torch_input_tensor_1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_2 = ttnn.from_torch(torch_input_tensor_2, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.concat(
        [input_tensor_0, input_tensor_1, input_tensor_2],
        1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

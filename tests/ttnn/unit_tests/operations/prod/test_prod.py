# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
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
                raise ValueError(
                    f"mesh_shape must be a non-empty list or tuple of positive integers, got {mesh_shape}"
                )
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


@pytest.mark.parametrize("input_values", [
    [224, 224],
    [55, 55],
])
def test_prod(input_values):
    device = DeviceGetter.get_device((1, 1))

    cpu_input = torch.tensor(input_values, dtype=torch.bfloat16)
    cpu_result = torch.prod(cpu_input)

    input_tensor = ttnn.from_torch(
        cpu_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    result = ttnn.prod(input_tensor, dim=0)
    tt_result = ttnn.to_torch(result)

    assert torch.equal(tt_result, cpu_result), (
        f"ttnn.prod returned {tt_result.item()}, expected {cpu_result.item()}"
    )

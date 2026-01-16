# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal, maybe_trace

DEEPSEEK_REC_SEND_SHAPE_DTYPE_MEM = [
    ((1, 1), (0, 1), (1, 1, 32, 896), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
    ((2, 1), (1, 1), (1, 1, 32, 896), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
    ((1, 1), (0, 1), (1, 1, 32, 7168), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
]


def _linear_coord(coord, mesh_shape):
    return coord[0] * mesh_shape[1] + coord[1]


@pytest.mark.requires_device(["TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
@pytest.mark.parametrize("test_config", DEEPSEEK_REC_SEND_SHAPE_DTYPE_MEM)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
def test_point_to_point_deepseek(mesh_device, test_config, layout, enable_trace):
    # send, sreceive
    coord0, coord1, shape, dtype, memory_config = test_config

    devices = prod(list(mesh_device.shape))
    multi_device_shape = tuple(s * (devices if i == 0 else 1) for i, s in enumerate(shape))

    lcoord0, lcoord1 = (_linear_coord(c, list(mesh_device.shape)) for c in (coord0, coord1))
    coord0, coord1 = (ttnn.MeshCoordinate(c) for c in (coord0, coord1))

    idx_start0, idx_end0 = lcoord0 * shape[0], (lcoord0 + 1) * shape[0]
    idx_start1, idx_end1 = lcoord1 * shape[0], (lcoord1 + 1) * shape[0]

    input_tensor_torch = torch.zeros(multi_device_shape).bfloat16()
    input_tensor_torch[idx_start0:idx_end0, :, :, :] = torch.linspace(1, prod(shape), prod(shape)).reshape(shape)
    input_tensor = ttnn.from_torch(
        input_tensor_torch,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        dtype=dtype,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    def run_op():
        return ttnn.point_to_point(
            input_tensor,
            coord0,
            coord1,
            topology=ttnn.Topology.Linear,
        )

    sent_tensor = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)

    sent_tensor_torch = ttnn.to_torch(sent_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert_equal(input_tensor_torch[idx_start0:idx_end0, :, :, :], sent_tensor_torch[idx_start1:idx_end1, :, :, :])

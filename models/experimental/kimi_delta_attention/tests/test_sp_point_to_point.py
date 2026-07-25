# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fabric-level building block for KDA sequence-parallel state handoff.

The KDA recurrence is causal across a sequence partition.  Before integrating
an SP topology into the layer, prove that a 1x8 Blackhole mesh can relay the
two pieces of boundary state without materializing either on the host:

* the FP32 recurrent matrix, and
* the three previous projected Q/K/V samples required by the kernel-size-four
  causal short convolution.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole


_MESH_SHAPE = (1, 8)
_TP4_LOCAL_HEADS = 8
_HEAD_KEY_DIM = 128
_HEAD_VALUE_DIM = 128
_CONV_HISTORY = 3


def _relay_left_to_right(tensor: ttnn.Tensor) -> None:
    """Overwrite each next SP rank with the preceding rank's boundary tensor."""
    for source_rank in range(_MESH_SHAPE[1] - 1):
        ttnn.point_to_point(
            tensor,
            ttnn.MeshCoordinate(0, source_rank),
            ttnn.MeshCoordinate(0, source_rank + 1),
            topology=ttnn.Topology.Linear,
            output_tensor=tensor,
        )


pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


def test_sequence_parallel_boundary_state_relay(mesh_device: ttnn.MeshDevice) -> None:
    """Relay KDA's two causal carries across all eight sequence ranks.

    The recurrent tensor models one local TP=4 rank (eight heads); the same
    operation is issued independently for every TP rank in the eventual
    `SP=2, TP=4` and `SP=8, TP=4` layouts.
    """
    generator = torch.Generator().manual_seed(2971)
    # Dim 0 is the mesh-shard dimension. Each device initially has unique data
    # so a successful relay is distinguishable from a replicated input.
    recurrent_host = torch.randn(
        _MESH_SHAPE[1],
        _TP4_LOCAL_HEADS,
        _HEAD_KEY_DIM,
        _HEAD_VALUE_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    convolution_host = torch.randn(
        _MESH_SHAPE[1], _CONV_HISTORY, 3 * _HEAD_KEY_DIM, generator=generator, dtype=torch.bfloat16
    )
    expected_recurrent = recurrent_host[:1].expand_as(recurrent_host)
    expected_convolution = convolution_host[:1].expand_as(convolution_host)

    recurrent_tt = ttnn.from_torch(
        recurrent_host,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    convolution_tt = ttnn.from_torch(
        convolution_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    _relay_left_to_right(recurrent_tt)
    _relay_left_to_right(convolution_tt)
    ttnn.synchronize_device(mesh_device)

    actual_recurrent = ttnn.to_torch(recurrent_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    actual_convolution = ttnn.to_torch(convolution_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    torch.testing.assert_close(actual_recurrent, expected_recurrent, rtol=0, atol=0)
    torch.testing.assert_close(actual_convolution, expected_convolution, rtol=0, atol=0)

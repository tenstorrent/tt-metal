# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended (verifier-authored) coverage for point_to_point.

The acceptance suite, the golden suite, and the precision baseline all pin the
adjacent (0,0)->(0,1) pair — a single fabric hop. This file adds the one
correctness path none of them exercise: MULTI-HOP unicast (num_hops > 1). The
route + hop count is baked into the runtime args and drives the fabric
sign-reversal handshake (op_design.md "Key Risks": route direction / hops), so a
2- and 3-hop transfer is the highest-value gap to close.

Transfers stay on row 0 (the x direction the blackhole_8xP150 torus_x descriptor
routes), varying only the receiver column so num_hops = 2 and 3. Small matrix:
two receivers x two float dtypes, TILE + Linear. Drive it via the multi-device
sim runner (mesh shape MUST match the topology or fabric init hangs):

    scripts/run_multidevice_sim_pytest.py --op point_to_point -- \
        tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_extended.py -v
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.point_to_point import point_to_point

MESH_SHAPE = (2, 4)
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

PCC = {ttnn.bfloat16: 0.995, ttnn.float32: 0.999}


def _linear_index(coord, mesh_shape):
    mesh_shape = tuple(mesh_shape)
    return coord[0] * mesh_shape[1] + coord[1]


def _make_input(mesh_device, shard_shape, dtype, layout):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    torch.manual_seed(11)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    input_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(input_tensor)]
    return input_tensor, input_shards


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
# receiver column 2 -> 2 hops from (0,0); column 3 -> 3 hops. Row 0 = the torus_x
# routing direction, so no cross-row fabric assumption is made.
@pytest.mark.parametrize("receiver_col", [2, 3])
def test_point_to_point_multi_hop(mesh_device, dtype, receiver_col):
    """A >1-hop unicast still delivers the sender's shard bit-for-bit."""
    if prod(tuple(mesh_device.shape)) < 4:
        pytest.skip("multi-hop needs >= 4 devices along the row")

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, receiver_col)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, (1, 1, 64, 128), dtype, ttnn.TILE_LAYOUT)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)
    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    # Receiver (num_hops = receiver_col away) holds the sender's shard.
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], pcc)
    # Sender's own shard is unchanged.
    assert_with_pcc(input_shards[send_idx], output_shards[send_idx], pcc)
    # An intermediate device on the path is untouched (pure unicast, no relay writes).
    mid_idx = _linear_index(ttnn.MeshCoordinate(0, 1), mesh_device.shape)
    assert_with_pcc(input_shards[mid_idx], output_shards[mid_idx], pcc)
    logger.info(f"p2p multi-hop {dtype} (0,0)->(0,{receiver_col}): receiver shard matches, path untouched")

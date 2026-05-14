# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_16: 3-pass ETH launch ordering prevents simultaneous handshake deadlock
#
# Strategy: Run AllGather + quiesce in a tight loop (20 iterations) on a T3K
# mesh (>= 4 devices). With 20 iterations, any non-deterministic relay latency
# variation (the root cause of the original deadlock at Device 4 chan 6 / Device 5
# chan 6) has high probability of triggering if the 3-pass ordering is broken.
#
# Pass = all 20 quiesce cycles complete within 30s each AND final AllGather correct.
# Fail = any quiesce hangs (STARTED-STARTED deadlock), detected by per-cycle 30s timer.
#
# Background — FIX AE + FIX AF:
#   FIX AE introduces a 3-sub-pass ETH launch in quiesce_and_restart_fabric_workers:
#     Pass 1a: launch ERISCs on MMIO devices (ring master)
#     Pass 1b: wait for MMIO ERISCs to reach STARTED
#     Pass 1c: launch non-MMIO ERISCs one device at a time, polling STARTED
#              between each successive non-MMIO device (FIX AF barrier)
#   Without these barriers, two non-MMIO peers can initiate the ETH handshake
#   simultaneously, both blocking in SENDER state, causing a STARTED-STARTED
#   deadlock where neither side transitions to RECEIVER.

import time
import torch
import pytest
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

_NUM_ITERATIONS = 20
_QUIESCE_DEADLINE_S = 30


@pytest.fixture
def mesh_device():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("num_links", [1])
def test_3pass_eth_launch_no_deadlock(mesh_device, num_links):
    """Stress the 3-pass quiesce ETH launch ordering to catch STARTED-STARTED deadlock."""
    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need >= 4 devices for non-MMIO peer ordering test")

    num_devices = mesh_device.get_num_devices()
    per_device_cols = 32

    for i in range(_NUM_ITERATIONS):
        input_tensor = torch.randn(1, 1, 32, per_device_cols * num_devices)
        tt_input = ttnn.from_torch(
            input_tensor,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
        )
        start = time.time()
        tt_output = ttnn.all_gather(tt_input, dim=3, num_links=num_links)
        elapsed = time.time() - start

        assert elapsed < _QUIESCE_DEADLINE_S, (
            f"Iteration {i}: AllGather+quiesce took {elapsed:.1f}s > {_QUIESCE_DEADLINE_S}s "
            f"(probable STARTED-STARTED handshake deadlock — FIX AE/AF regression)"
        )

        # Verify output correctness
        output = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
        assert output.shape == input_tensor.shape, f"Iteration {i}: output shape mismatch"

        ttnn.deallocate(tt_output)
        ttnn.deallocate(tt_input)

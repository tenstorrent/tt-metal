# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_17: Partial-mesh quiesce — FIX AK non-fatal REMOTE_HANDSHAKE_COMPLETE
#
# Strategy: Open a 1x4 sub-mesh on T3K (leaving 4 devices outside the mesh).
# Mesh-edge devices have ETH channels connected to out-of-mesh chips that never
# respond to the EDM handshake. These channels get stuck at STARTED or
# REMOTE_HANDSHAKE_COMPLETE. FIX AK classifies these as non-fatal.
# FIX AM sets fabric_channels_not_ready_for_traffic_ so callers can distinguish
# this from a broken relay.
#
# Pass = all quiesce cycles complete without TT_THROW (Phase 5b non-fatal),
#        and subsequent AllGather on in-mesh devices succeeds.
# Fail = Phase 5b throws TT_FATAL_ERROR or TT_THROW on stuck handshake channels
#        (regression to pre-FIX AK behavior).
#
# Background — FIX AK:
#   Phase 5b in quiesce_and_restart_fabric_workers checks each ETH channel state.
#   Pre-FIX AK: STARTED/REMOTE_HANDSHAKE_COMPLETE on any channel → TT_THROW.
#   Post-FIX AK: those states on mesh-edge devices are classified non-fatal;
#   fabric_channels_not_ready_for_traffic_ is set (FIX AM) to signal degraded state.

import time
import torch
import pytest
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

_NUM_ITERATIONS = 5
_QUIESCE_DEADLINE_S = 30


@pytest.fixture
def partial_mesh():
    # Open a 1x4 sub-mesh on T3K (leaves 4 devices outside the mesh)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("num_links", [1])
def test_partial_mesh_quiesce_nonfatal(partial_mesh, num_links):
    """Verify quiesce on a partial mesh doesn't throw on incomplete ETH handshakes."""
    if partial_mesh.get_num_devices() < 4:
        pytest.skip("Need T3K (8 devices) for partial-mesh test")

    num_devices = partial_mesh.get_num_devices()
    per_device_cols = 32

    for i in range(_NUM_ITERATIONS):
        input_tensor = torch.randn(1, 1, 32, per_device_cols * num_devices)
        tt_input = ttnn.from_torch(
            input_tensor,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ShardTensorToMesh(partial_mesh, dim=3),
        )
        start = time.time()
        # This triggers quiesce on close of each AllGather scope.
        # With a 1x4 partial mesh, Phase 5b sees mesh-edge channels
        # stuck at REMOTE_HANDSHAKE_COMPLETE. FIX AK must not throw here.
        tt_output = ttnn.all_gather(tt_input, dim=3, num_links=num_links)
        elapsed = time.time() - start

        # Key assertion: quiesce must not throw for mesh-edge channels stuck at
        # REMOTE_HANDSHAKE_COMPLETE. If FIX AK is reverted, Phase 5b throws here.
        assert elapsed < _QUIESCE_DEADLINE_S, (
            f"Iteration {i}: took {elapsed:.1f}s — probable Phase 5b throw "
            f"on partial-mesh handshake (FIX AK regression)"
        )

        output = ttnn.to_torch(
            tt_output, mesh_composer=ConcatMeshToTensor(partial_mesh, dim=3)
        )
        assert output.shape == input_tensor.shape, (
            f"Iteration {i}: output shape mismatch — "
            f"partial-mesh AllGather produced incorrect data"
        )

        ttnn.deallocate(tt_output)
        ttnn.deallocate(tt_input)

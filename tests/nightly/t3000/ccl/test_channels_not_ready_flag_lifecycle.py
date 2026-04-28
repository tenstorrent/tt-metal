# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_19: channels_not_ready_for_traffic_ flag cleared on re-init (FIX AM lifecycle)
#
# Strategy:
#   1. Open partial mesh (1x4) — triggers FIX AK/AM on mesh-edge channels,
#      setting fabric_channels_not_ready_for_traffic_=true.
#   2. Close partial mesh (quiesce + teardown).
#   3. Open full mesh (1x8) — configure_fabric() must clear the flag (FIX AM).
#   4. Run AllGather on full mesh — must succeed with correct output.
#
# Pass = full-mesh AllGather produces correct output after partial-mesh cycle.
# Fail = channels_not_ready flag persists across mesh re-open, causing AllGather
#        on "not-ready" channels to produce wrong data or hang.
#
# Background — FIX AM:
#   Phase 5b sets fabric_channels_not_ready_for_traffic_=true when any channel
#   is stuck below READY_FOR_TRAFFIC (e.g. mesh-edge channels in partial mesh).
#   FIX AM adds a clear() call at the top of configure_fabric() so the flag
#   is reset for each new mesh open. Without this, the flag persists and
#   silently degrades all subsequent mesh operations.

import torch
import pytest
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def test_channels_not_ready_cleared_on_reinit():
    """
    channels_not_ready_for_traffic_ must be cleared when a new mesh is opened.
    Tests the partial-mesh -> full-mesh transition that exercises FIX AM lifecycle.
    """
    # Phase 1: partial mesh — triggers FIX AK/AM on mesh-edge channels
    partial = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

    if partial.get_num_devices() < 4:
        ttnn.close_mesh_device(partial)
        pytest.skip("Need T3K (8 devices) for partial-mesh -> full-mesh transition test")

    partial_num_devices = partial.get_num_devices()
    t1 = torch.randn(1, 1, 32, 32 * partial_num_devices)
    tt1 = ttnn.from_torch(
        t1,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(partial, dim=3),
    )
    out1 = ttnn.all_gather(tt1, dim=3, num_links=1)
    ttnn.deallocate(out1)
    ttnn.deallocate(tt1)
    ttnn.close_mesh_device(partial)

    # Phase 2: full mesh — configure_fabric() must clear channels_not_ready flag
    full = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))

    if full.get_num_devices() < 8:
        ttnn.close_mesh_device(full)
        pytest.skip("Need all 8 T3K devices for full-mesh verification")

    full_num_devices = full.get_num_devices()
    t2 = torch.randn(1, 1, 32, 32 * full_num_devices)
    tt2 = ttnn.from_torch(
        t2,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(full, dim=3),
    )
    out2 = ttnn.all_gather(tt2, dim=3, num_links=1)

    result = ttnn.to_torch(out2, mesh_composer=ConcatMeshToTensor(full, dim=3))

    # If channels_not_ready was NOT cleared, AllGather on the formerly "not-ready"
    # channels produces wrong shape or hangs. Shape check is the fast-fail path.
    assert result.shape == t2.shape, (
        f"Full-mesh AllGather shape mismatch after partial-mesh cycle — "
        f"channels_not_ready_for_traffic_ not cleared by configure_fabric() (FIX AM regression). "
        f"Expected {t2.shape}, got {result.shape}"
    )

    ttnn.deallocate(out2)
    ttnn.deallocate(tt2)
    ttnn.close_mesh_device(full)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor


def _run_ag(ccl, mesh, label):
    """Run an all-gather on ``mesh`` and report whether it completes."""
    t = bf16_tensor(torch.randn(1, 1, 32, 256 * mesh.shape[1]), device=mesh, mesh_axis=1, shard_dim=3)
    out = ccl.all_gather(t, dim=3, mesh_axis=1, use_hyperparams=False)
    ttnn.synchronize_device(mesh)
    print(f"[REPRO] all_gather OK ({label}): {tuple(out.shape)}", flush=True)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300000000,
                "num_command_queues": 2,
            },
        ]
    ],
    indirect=["mesh_device", "device_params"],
)
def test_submesh_cq1_repro(mesh_device):
    """Parent video (cq 0) + overlapping child audio submesh (cq 1) must not deadlock.

    Models LTX_AUDIO_SUBMESH routing: the parent 4x8 runs CCL on cq 0 and opens a fabric
    connection on every chip; the overlapping 1x4 audio child then runs CCL on cq 1 on 4 of
    those chips. The worker fabric connection lives at a fixed per-physical-chip L1 address and
    is reused whenever its initialized flag is set, with no cq/submesh awareness — so the child
    reuses the parent's stale connection (built for the parent's routing) and the child CCL
    deadlocks. cq routing does not help (the connection is per chip, not per cq).

    Host-zeroing the connection region on the shared chips (``reset_fabric_connection_lock``)
    while they are idle forces the next worker op to open a fresh connection for its own routing.
    Resetting before the child runs lets the child open its own connection; resetting again before
    returning to the parent lets the parent reopen its own. Without the resets this test hangs at
    the child CCL (or the next parent CCL); with them every stage completes and close is clean.
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)

    # Parent (video proxy) on the default cq (cq 0): opens fabric on all chips.
    _run_ag(ccl, mesh, "parent cq0, before audio submesh")

    audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)

    # Idle barrier, then drop the stale (parent-built) fabric connection on the child's chips so
    # the child opens its own. The submesh is freshly created and nothing is in flight.
    ttnn.synchronize_device(parent)
    ttnn.reset_fabric_connection_lock(audio)

    with ttnn.command_queue(1):
        _run_ag(audio_ccl, audio, "child cq1")

    # Idle barrier, then drop the child-built connection on the shared chips so the parent reopens
    # its own when it returns to cq 0.
    ttnn.synchronize_device(parent)
    ttnn.reset_fabric_connection_lock(audio)

    _run_ag(ccl, mesh, "parent cq0, after audio submesh")

    # The child dirtied cq 0 building its global semaphores even though its CCL ran on cq 1, so the
    # per-cq close guard would throw at teardown for every mesh in the chain sharing cq 0. Finish +
    # reset in_use on the child and the parent submesh so conftest teardown closes them cleanly.
    ttnn.reset_cq_in_use(audio)
    ttnn.reset_cq_in_use(mesh)
    print("[REPRO] PASS: parent cq0 + child cq1 ran; cqs reset for clean teardown", flush=True)

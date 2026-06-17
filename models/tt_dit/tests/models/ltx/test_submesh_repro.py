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
    """Parent video (cq 0) + child audio submesh (cq 1) must not deadlock or throw at close.

    Models the LTX_AUDIO_SUBMESH=1x4 routing: the parent 4x8 keeps cq 0, the overlapping 1x4
    audio child runs its CCL on cq 1. cq routing alone is not enough — fabric/EDM connection
    and event state are per physical chip, not per cq, so after the child runs CCL on cq 1 the
    parent's next cq-0 CCL on the 4 shared chips deadlocks against the child's stale fabric
    connection. ``synchronize_device`` only drains the submesh it is called on, so it does not
    fix this. ``quiesce_devices`` on the common parent recursively drains every submesh's CQs
    and resets in-use/event state across the shared chips — the intended cross-submesh barrier
    between phases that touch overlapping submeshes — so the parent op can re-open fabric cleanly.
    The same drain leaves the shared cq idle, so the per-cq close guard (mesh_device.cpp) also
    does not throw at conftest teardown.
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)

    # Parent (video proxy) on the default cq (cq 0).
    _run_ag(ccl, mesh, "parent cq0, before audio submesh")

    # Child 1x4 audio submesh, built and exercised on cq 1.
    with ttnn.command_queue(1):
        audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
        audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
        _run_ag(audio_ccl, audio, "child cq1")

    # Cross-submesh barrier on the common parent before returning to cq 0 on the shared chips.
    parent.quiesce_devices()
    print("[REPRO] quiesce_devices() returned", flush=True)

    # Parent again on cq 0 after the child ran on cq 1.
    _run_ag(ccl, mesh, "parent cq0, after audio submesh")
    print("[REPRO] PASS: parent cq0 + child cq1 ran; close must not throw at teardown", flush=True)

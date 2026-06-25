# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Up-front precompile across a multi-device mesh (MeshWorkload path).

Drives ops on a MeshDevice through collect -> parallel compile -> warm forward,
covering the multi-chip MeshWorkload path. Both cases here are homogeneous (every
device runs the same program on its shard), so the collector dedups to a small
unique set regardless of mesh size, and one compile warms every chip.

Parametrized for a 1x2 mesh; the mesh_device fixture skips on machines with fewer
devices. Mesh dispatch goes over fabric, so run under flock / run_safe_pytest.sh.
"""

import pytest
import torch

import ttnn


def _chain(x):
    """exp(x) + x, times exp(x) — a few distinct elementwise programs."""
    a = ttnn.exp(x)
    return ttnn.multiply(ttnn.add(a, x), a)


def _torch_chain(t):
    a = torch.exp(t.float())
    return (a + t.float()) * a


def _collect_and_compile(mesh_device, run_body):
    """Collect run_body() under NO_DISPATCH, parallel-compile, assert the mechanics."""
    ttnn.graph.up_front_clear()
    ttnn.graph.up_front_begin_collect()
    try:
        run_body()
    finally:
        ttnn.graph.up_front_end_collect()
    n_unique = ttnn.graph.up_front_num_unique()
    assert ttnn.graph.up_front_num_collected() >= 1, "collect captured nothing on the mesh"
    num_programs, num_errors, _, _ = ttnn.graph.up_front_compile(mesh_device, 4)
    assert num_errors == 0, "parallel compile reported errors"
    assert num_programs == n_unique, f"compiled {num_programs} programs, expected the {n_unique} unique collected"


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_mesh_replicated_up_front_compile(mesh_device):
    """Same tensor replicated to every device; warm output on each device must match."""
    torch.manual_seed(0)
    t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    _collect_and_compile(mesh_device, lambda: _chain(x))

    # Warm run, then bring back one output per device (concat on dim 0 -> [num_devices, 1, 64, 64]).
    out = _chain(x)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    expected = _torch_chain(t)[0].flatten()  # [1, 64, 64] -> flat

    assert got.shape[0] == mesh_device.get_num_devices(), f"expected one replica per device, got {got.shape}"
    for i in range(got.shape[0]):
        pcc = torch.corrcoef(torch.stack([got[i].flatten(), expected]))[0, 1].item()
        assert pcc > 0.999, f"device {i} warm output incorrect: PCC {pcc}"


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_mesh_sharded_up_front_compile(mesh_device):
    """Tensor sharded row-wise across the mesh; recomposed warm output must match the whole."""
    num_dev = mesh_device.get_num_devices()
    torch.manual_seed(0)
    t = torch.randn(1, 1, 32 * num_dev, 64, dtype=torch.bfloat16)  # 32 tile-aligned rows per device
    x = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    _collect_and_compile(mesh_device, lambda: _chain(x))

    out = _chain(x)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)).float().flatten()
    expected = _torch_chain(t).flatten()
    pcc = torch.corrcoef(torch.stack([got, expected]))[0, 1].item()
    assert pcc > 0.999, f"sharded warm output incorrect: PCC {pcc}"

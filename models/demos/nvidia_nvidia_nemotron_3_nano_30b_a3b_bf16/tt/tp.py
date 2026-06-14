# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 helpers for NemotronH-30B on QB (4× Blackhole).

All component forwards accept/return CPU torch tensors and handle
device sharding internally via these helpers.

Topology: FABRIC_1D_RING + Topology.Linear (4-chip linear chain on QB).
"""

import ttnn

TP = 4
FABRIC = ttnn.FabricConfig.FABRIC_1D_RING
TOPOLOGY = ttnn.Topology.Linear

_R = ttnn.ReplicateTensorToMesh
_S = ttnn.ShardTensorToMesh
_C = ttnn.ConcatMeshToTensor


def open_device_tp4() -> ttnn.MeshDevice:
    """Open the 4-chip QB mesh with FABRIC_1D_RING fabric."""
    ttnn.set_fabric_config(FABRIC)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, TP), physical_device_ids=list(range(TP)))


def close_device_tp4(mesh: ttnn.MeshDevice) -> None:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# from_torch helpers
# ---------------------------------------------------------------------------


def _rep(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Load tensor replicated on all TP devices."""
    return ttnn.from_torch(
        t.bfloat16() if dtype == ttnn.bfloat16 else t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=_R(mesh),
    )


def _col(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Column-parallel: shard weight along output-feature dim (dim=0)."""
    return ttnn.from_torch(
        t.bfloat16(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=_S(mesh, dim=0),
    )


def _row(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Row-parallel: shard weight along input-feature dim (dim=1)."""
    return ttnn.from_torch(
        t.bfloat16(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=_S(mesh, dim=1),
    )


def _shard_act(t, mesh, dim, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Shard an activation tensor along `dim` across TP devices."""
    return ttnn.from_torch(
        t.bfloat16(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=_S(mesh, dim=dim),
    )


# ---------------------------------------------------------------------------
# to_torch helpers
# ---------------------------------------------------------------------------


def _host_rep(t_tt, mesh, n):
    """Bring a replicated device tensor to host, returning first `n` rows (dim-0 slice).

    With MeshShape(1,4) and ReplicateTensorToMesh, ConcatMeshToTensor(dim=0)
    concatenates the 4 identical copies along dim=0 → [4n, ...].  Take [:n].
    """
    full = ttnn.to_torch(t_tt, mesh_composer=_C(mesh, dim=0))
    return full[:n].bfloat16()


def _host_sharded(t_tt, mesh, concat_dim):
    """Bring a column-parallel (sharded) device tensor to host by concatenating along `concat_dim`."""
    return ttnn.to_torch(t_tt, mesh_composer=_C(mesh, dim=concat_dim)).bfloat16()


# ---------------------------------------------------------------------------
# CCL wrappers
# ---------------------------------------------------------------------------


def all_reduce(t_tt):
    """Element-wise sum across all TP devices; result identical on all devices."""
    return ttnn.all_reduce(t_tt, topology=TOPOLOGY)


def all_gather(t_tt, dim):
    """Gather sharded tensors from all TP devices along `dim`."""
    return ttnn.all_gather(t_tt, dim=dim, topology=TOPOLOGY)

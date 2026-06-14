# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 helpers for NemotronH-30B on QB (4× Blackhole).

All component forwards accept/return CPU torch tensors and handle
device sharding internally via these helpers.

Topology: FABRIC_1D + Topology.Linear (4-chip linear chain on QB).
"""

import weakref

import ttnn

TP = 4
FABRIC = ttnn.FabricConfig.FABRIC_1D
TOPOLOGY = ttnn.Topology.Linear

_R = ttnn.ReplicateTensorToMesh
_S = ttnn.ShardTensorToMesh
_C = ttnn.ConcatMeshToTensor

# ---------------------------------------------------------------------------
# Device weight cache — avoids re-uploading the same weight on every forward.
#
# Key: (id(source_tensor), shard_dim_or_None, layout, dtype, id(mesh))
# Value: (weakref.ref(source_tensor), device_tensor)
#
# Correctness: the weakref tracks the SOURCE tensor's lifetime.
#   - Weights (held by WeightCache._shards): their weakref stays alive → cache hits.
#   - Activations (transient): freed after the forward → weakref() returns None
#     on the next call → stale entry evicted → re-upload. Safe even if Python
#     reuses the same id() for a new tensor after GC.
# ---------------------------------------------------------------------------
_DEVICE_CACHE: dict = {}


def clear_device_weight_cache() -> None:
    _DEVICE_CACHE.clear()


def open_device_tp4() -> ttnn.MeshDevice:
    """Open the 4-chip QB mesh with FABRIC_1D fabric."""
    ttnn.set_fabric_config(FABRIC)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, TP), physical_device_ids=list(range(TP)))


def close_device_tp4(mesh: ttnn.MeshDevice) -> None:
    clear_device_weight_cache()
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# from_torch helpers
# ---------------------------------------------------------------------------


def _upload(t, mesh, shard_dim, layout, dtype):
    """Upload tensor to mesh, caching by source-tensor lifetime.

    A cache entry is valid only while the source tensor is alive.  Weight
    tensors (held by WeightCache) live for the model's lifetime → always hit
    after the first call.  Activation tensors die after each forward → their
    weakref() returns None → stale entry evicted → safe re-upload.
    """
    key = (id(t), shard_dim, layout, dtype, id(mesh))
    if key in _DEVICE_CACHE:
        weak_src, dev_tensor = _DEVICE_CACHE[key]
        if weak_src() is not None:  # source still alive → valid hit
            return dev_tensor
        del _DEVICE_CACHE[key]  # source GC'd → evict stale entry

    mapper = _R(mesh) if shard_dim is None else _S(mesh, dim=shard_dim)
    dev_tensor = ttnn.from_torch(
        t.bfloat16() if dtype == ttnn.bfloat16 else t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=mapper,
    )
    try:
        _DEVICE_CACHE[key] = (weakref.ref(t), dev_tensor)
    except TypeError:
        pass  # not all types support weakrefs; skip caching for this tensor
    return dev_tensor


def _rep(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Load tensor replicated on all TP devices (cached after first upload)."""
    return _upload(t, mesh, None, layout, dtype)


def _col(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Column-parallel: shard weight along output-feature dim (dim=0), cached."""
    return _upload(t, mesh, 0, layout, dtype)


def _row(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Row-parallel: shard weight along input-feature dim (dim=1), cached."""
    return _upload(t, mesh, 1, layout, dtype)


def _shard_act(t, mesh, dim, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Shard an activation tensor along `dim` — NOT cached (activations change)."""
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

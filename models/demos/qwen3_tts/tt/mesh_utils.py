# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Mesh / tensor-parallel helpers for Qwen3-TTS.

TP=2 (and beyond) is driven by the mesh shape passed at device-open time.
Modules read ``get_tp_size(device)`` once at construction and switch their
weight layout + forward path accordingly. TP=1 (plain Device or 1x1 mesh)
is the legacy single-chip path.
"""

from __future__ import annotations

import ttnn


def is_mesh_device(device) -> bool:
    return device.__class__.__name__ == "MeshDevice"


def get_mesh_shape(device):
    """Return (rows, cols) for a MeshDevice, or (1, 1) for a plain Device."""
    if not is_mesh_device(device):
        return (1, 1)
    shape = list(device.shape)
    if len(shape) == 1:
        return (1, shape[0])
    return (shape[0], shape[1])


def get_tp_size(device) -> int:
    """Tensor-parallel size = number of devices along the column axis of the mesh.

    For (1, N) meshes (N150=1, N300=2, T3K=8) this is N. For multi-row meshes
    we only TP along the column axis for now.
    """
    rows, cols = get_mesh_shape(device)
    return max(rows, cols) if min(rows, cols) == 1 else cols


def is_n300(device) -> bool:
    """True only for a Wormhole 2-chip mesh, i.e. an N300 card opened as (1,2)/(2,1).

    Gate for N300-specific fast paths: the shard grids and CCL trade-offs below are
    picked for wormhole's 8x8 compute grid / 12 DRAM banks at tp_size=2, so N150
    (1 chip), T3K (8 chips) and Blackhole all keep the generic path.
    """
    if not is_mesh_device(device):
        return False
    try:
        if device.get_num_devices() != 2:
            return False
        if device.arch() != ttnn._ttnn.device.Arch.WORMHOLE_B0:
            return False
    except Exception:
        return False
    rows, cols = get_mesh_shape(device)
    return min(rows, cols) == 1 and max(rows, cols) == 2


def to_torch(t: ttnn.Tensor, device=None, **kwargs) -> "torch.Tensor":
    """Drop-in for ttnn.to_torch that handles multi-device meshes.

    On a (1, N) mesh all chips hold the same data after all_reduce, so we
    extract chip-0's view via ConcatMeshToTensor and take the first slice.
    On a plain Device or (1,1) mesh the call passes through unchanged.

    The optional ``device`` argument is looked up on the tensor if not provided:
    ``t.device()`` works for both Device and MeshDevice, but for older
    codebases that pass the tensor only, we fall back to ``ttnn.to_torch(t)``.
    """
    import torch as _torch  # noqa — local import to avoid circular dependency

    # Determine whether this is a multi-device tensor.
    dev = device
    if dev is None:
        try:
            dev = t.device()
        except Exception:
            return ttnn.to_torch(t, **kwargs)

    if dev.__class__.__name__ == "MeshDevice" and dev.get_num_devices() > 1:
        stacked = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0), **kwargs)
        return stacked[0:1]
    return ttnn.to_torch(t, **kwargs)


def tp_all_reduce(tensor: ttnn.Tensor, device, memory_config=None) -> ttnn.Tensor:
    """All-reduce ``tensor`` across the TP axis. No-op when tp_size==1.

    Uses ``ttnn.all_reduce`` (handles semaphore + topology internally) on a
    1-D mesh; cluster_axis is inferred from the mesh shape.
    """
    if get_tp_size(device) == 1:
        return tensor
    rows, cols = get_mesh_shape(device)
    # For (1, N) or (N, 1) meshes pick the non-singleton axis.
    cluster_axis = 1 if rows == 1 else 0
    kwargs = {"cluster_axis": cluster_axis}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    return ttnn.all_reduce(tensor, **kwargs)


def tp_all_reduce_2chip(tensor: ttnn.Tensor, device, memory_config=None) -> ttnn.Tensor:
    """All-reduce across exactly 2 chips using one CCL op instead of two.

    ``ttnn.all_reduce`` lowers to reduce_scatter + all_gather, and on N300 both are
    dominated by fixed fabric setup rather than payload — a 1-tile CP activation pays
    ~51 us to reduce 64 KB. With two chips we can instead all-gather the two partial
    sums and add the halves locally: 34 us of CCL plus ~4 us of slice/add.

    Two measured details, both worth keeping:
      * Gather on the last dim. Width is tile-aligned for every CP/Talker activation,
        whereas a size-1 outer dim or a 1-2-row height inside a 32-row tile pushes
        all_gather onto its composite all-broadcast fallback (78 us vs 34 us).
      * Leave ``num_links`` on auto. Forcing 2 links doubled the gather to 69 us: the
        payload is far too small to amortise a second link's setup.
    """
    rows, cols = get_mesh_shape(device)
    cluster_axis = 1 if rows == 1 else 0
    mc = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
    shape = list(tensor.shape)
    w = shape[-1]

    gathered = ttnn.all_gather(tensor, dim=-1, cluster_axis=cluster_axis, memory_config=mc)
    lo = ttnn.slice(gathered, [0, 0, 0, 0], shape[:-1] + [w], memory_config=mc)
    hi = ttnn.slice(gathered, [0, 0, 0, w], shape[:-1] + [2 * w], memory_config=mc)
    ttnn.deallocate(gathered)
    out = ttnn.add(lo, hi, memory_config=mc)
    ttnn.deallocate(lo)
    ttnn.deallocate(hi)
    return out

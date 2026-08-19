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

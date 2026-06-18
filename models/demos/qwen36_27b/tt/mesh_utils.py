# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh helpers for the Qwen3.6-27B TP (tensor-parallel) vLLM path.

Single-device code uses ``ttnn.to_torch`` directly; the TP path runs on a 1xN
line mesh where some tensors are sharded across devices and some are replicated.
``to_torch`` here reads a (possibly multi-device) tensor back to host as a single
torch tensor, transparently composing shards or de-duplicating replicas.

This mirrors the ``mesh_utils.to_torch`` the coder_next template imports.
"""

import torch
import ttnn


def _num_devices(tensor) -> int:
    dev = getattr(tensor, "device", None)
    try:
        d = dev() if callable(dev) else dev
        n = d.get_num_devices() if hasattr(d, "get_num_devices") else None
        if n:
            return int(n)
    except Exception:
        pass
    return 1


def to_torch(tensor, dim=None, dedup=True):
    """Read a ttnn tensor (single- or multi-device) back to a host torch tensor.

    Args:
        tensor: a ttnn.Tensor (on device or host).
        dim: if given, compose multi-device shards by concatenating along this
            dim (ConcatMeshToTensor). If None, the tensor is assumed REPLICATED
            and the first shard is returned (dedup=True), or all shards are
            concatenated on a new leading dim when dedup=False.
    Returns:
        torch.Tensor
    """
    if not isinstance(tensor, ttnn.Tensor):
        return tensor

    n = _num_devices(tensor)
    if n <= 1:
        return ttnn.to_torch(tensor)

    dev = tensor.device() if callable(getattr(tensor, "device", None)) else tensor.device

    if dim is not None:
        composer = ttnn.ConcatMeshToTensor(dev, dim=dim)
        return ttnn.to_torch(tensor, mesh_composer=composer)

    # Replicated tensor: every device holds the same data. Concatenate on a new
    # leading axis then take shard 0 (dedup) or keep all (dedup=False).
    composer = ttnn.ConcatMeshToTensor(dev, dim=0)
    full = ttnn.to_torch(tensor, mesh_composer=composer)
    if dedup:
        per = full.shape[0] // n
        return full[:per]
    return full

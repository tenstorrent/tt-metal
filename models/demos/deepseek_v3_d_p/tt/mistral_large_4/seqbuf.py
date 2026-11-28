# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device tensors sized by max_seq that are the same for every layer, kept once per mesh.

``SEQ_BUFFERS=shared`` gives every layer the same copy of: the attention's SP-gather scratch kv_full
[1, 1, max_seq, 576] bfp8 (replicated), the indexer's k_full [1, 1, max_seq, 128], the RoPE tables (one inverse
frequency for every RoPE layer: YaRN x a global LeRoPE alpha), the indexer's fp32 RoPE and position RowTables, and
the zero sin table of the softmax-scaling tables. The scratches are overwritten by every layer before they are read,
and layers run one after another on one command queue (eager or traced), so one copy serves all. ``per_layer``
(default) keeps a copy per layer.

``get(mesh, key, build)`` returns the mesh's copy of ``key`` (built on first use) or, per_layer, a new one.
``release(mesh)`` deallocates the mesh's shared copies (TtModel.resize).
"""

from __future__ import annotations

import hashlib

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

_SHARED: dict = {}  # (id(mesh), *key) -> object (ttnn.Tensor, IndexedTables, RowTables or a tuple of them)


def shared() -> bool:
    return settings.get("SEQ_BUFFERS") == "shared"


def tensor_key(t: torch.Tensor) -> str:
    """A key for a host tensor's values (an inverse frequency, a table's inputs)."""
    return hashlib.sha1(t.detach().float().contiguous().numpy().tobytes()).hexdigest()


def get(mesh, key: tuple, build):
    if not shared():
        return build()
    k = (id(mesh),) + tuple(key)
    if k not in _SHARED:
        _SHARED[k] = build()
    return _SHARED[k]


def is_shared(obj) -> bool:
    return any(v is obj for v in _SHARED.values())


def device_tensors(obj):
    """The ttnn tensors an object holds (a tensor, IndexedTables, RowTables or a tuple / list of them)."""
    if obj is None:
        return []
    if isinstance(obj, ttnn.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for o in obj for t in device_tensors(o)]
    return [v for v in vars(obj).values() if isinstance(v, ttnn.Tensor)]


def free(obj) -> None:
    """Deallocates a per-layer object; a shared one stays (release frees it)."""
    if obj is None or is_shared(obj):
        return
    keep = {id(t) for v in _SHARED.values() for t in device_tensors(v)}  # a shared part (the zero sin table)
    for t in device_tensors(obj):
        if id(t) not in keep and t.is_allocated():
            ttnn.deallocate(t)


def release(mesh) -> None:
    for k in [k for k in _SHARED if k[0] == id(mesh)]:
        for t in device_tensors(_SHARED.pop(k)):
            if t.is_allocated():
                ttnn.deallocate(t)

# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module-test helpers.

This file is a Phase 3 extension to ``util/`` (the implementer is permitted
to extend ``util/*.py``). It exposes:

* :func:`prepare_module` — runs the 3-step TTNNModule lifecycle
  (``to_device`` -> ``preprocess_weights`` -> ``move_weights_to_device``).
* :func:`replicated_from_torch` — :func:`ttnn.from_torch` shorthand that
  installs a :class:`ttnn.ReplicateTensorToMesh` mapper so the same per-device
  tensor is uploaded on every chip.
* :func:`gather_replicated_first` — read a replicated TTNN tensor's
  device-0 copy back to torch.

These are the same idioms that ``ops/`` tests built ad-hoc; collecting them
here keeps the Phase 3 module tests short and consistent.

Why "replicate, then gather first replica" instead of full DP-sharded
inputs? For correctness module tests we want the reference and the TT
forward to see the same numerical activations. Replicating the input gives
each chip the *full logical batch* (not a per-DP-shard slice), which means
the TT output's per-device tensor is also the full logical output and we
can compare it directly against the PyTorch reference. The production
batch-sharding pattern is exercised by the e2e + integration tests, not
by the unit module tests.
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn


def _recursive_set_device(tt_module, mesh_device, bypass: bool):
    """Walk a TTNNModule tree, setting ``device`` on every node.

    Production code uses
    ``models.experimental.tt_symbiote.utils.device_management.set_device``
    which performs the same recursive walk plus PyTorch ``forward`` timing
    hooks; we use a simpler version here that doesn't depend on
    ``draw_model_graph`` (which requires graphviz). The walk inspects each
    attribute, plus ``list`` / ``tuple`` / ``dict`` containers, mirroring
    :func:`models.experimental.tt_symbiote.core.module.set_module_name_recursively`.
    """
    from models.experimental.tt_symbiote.core.module import TTNNModule

    visited: set = set()

    def _walk(obj):
        if not isinstance(obj, TTNNModule):
            return
        if id(obj) in visited:
            return
        visited.add(id(obj))
        obj.to_device(mesh_device)
        obj._bypass_tensor_wrapping = bool(bypass)
        for name, child in obj.__dict__.items():
            if name == "_fallback_torch_layer":
                continue
            if isinstance(child, TTNNModule):
                _walk(child)
            elif isinstance(child, (list, tuple)):
                for v in child:
                    if isinstance(v, TTNNModule):
                        _walk(v)
            elif isinstance(child, dict):
                for v in child.values():
                    if isinstance(v, TTNNModule):
                        _walk(v)

    _walk(tt_module)


def prepare_module(tt_module, mesh_device, *, bypass_tensor_wrapping: bool = True):
    """Run TTNNModule's 3-step setup so it can execute on ``mesh_device``.

    Walks the module tree to set ``device`` and ``_bypass_tensor_wrapping`` on
    every child TTNNModule (parent classes do **not** propagate device to
    children automatically — production uses
    :func:`models.experimental.tt_symbiote.utils.device_management.set_device`
    for this; we replicate the walk here without the graphviz dependency).

    Production usage wraps every TTNNModule call in :class:`TorchTTNNTensor`
    via ``__torch_dispatch__``. Unit tests want to feed in / read back raw
    ``ttnn.Tensor`` instances, so we set ``_bypass_tensor_wrapping=True`` by
    default. This matches the flag the production pipeline sets on its
    children (``models/dots_ocr.py:_set_device_and_preprocess``).
    """
    _recursive_set_device(tt_module, mesh_device, bypass_tensor_wrapping)
    tt_module.preprocess_weights()
    tt_module.move_weights_to_device()
    return tt_module


def replicated_from_torch(
    t: torch.Tensor,
    *,
    mesh_device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config: Optional[ttnn.MemoryConfig] = None,
):
    """Upload ``t`` replicated across the mesh (every device gets the same copy)."""
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=mapper,
    )


def gather_replicated_first(tt_tensor, mesh_device) -> torch.Tensor:
    """Read a tensor and return one logical copy.

    Production TTNN modules may output tensors with **mixed** mesh
    distributions — width-sharded along the TP axis (mesh dim -1) and
    replicated along the DP axis (mesh dim 0). Try in order:

    1. 2D composer along ``(0, -1)`` — handles both replicate-along-DP
       and shard-along-TP at once. Most common production case.
    2. 1D :class:`ttnn.ConcatMeshToTensor` along dim 0 — for tensors
       that are pure replicas.
    3. Bare :func:`ttnn.to_torch` — single-device fallback.

    After step 1 or 2, the DP-axis replicas are identical, so we slice
    back to one per-device-row of the leading dim.
    """
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices <= 1:
        return ttnn.to_torch(tt_tensor)
    mesh_shape = tuple(int(x) for x in mesh_device.shape)
    if len(mesh_shape) == 1:
        mesh_shape = (mesh_shape[0], 1)

    # Try 2D composer first (handles row-sharded along TP and replicated along DP)
    try:
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, (0, -1))
        full = ttnn.to_torch(tt_tensor, mesh_composer=composer)
    except RuntimeError:
        # Fallback to 1D concat along dim=0 (pure-replicated tensors)
        composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        full = ttnn.to_torch(tt_tensor, mesh_composer=composer)

    dp = mesh_shape[0]
    if dp > 1 and full.shape[0] % dp == 0:
        per = full.shape[0] // dp
        full = full[:per]
    return full


def gather_col_sharded(tt_tensor, mesh_device) -> torch.Tensor:
    """Reassemble a tensor that was col-sharded along the TP axis.

    Used by the patch merger / vision tower e2e where the output is
    sharded across mesh axis -1 (TP). On a DP-only ``(8, 1)`` mesh this
    collapses to the replicated path.
    """
    mesh_shape = tuple(int(x) for x in mesh_device.shape)
    if len(mesh_shape) == 1:
        mesh_shape = (mesh_shape[0], 1)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, (0, -1))
    full = ttnn.to_torch(tt_tensor, mesh_composer=composer)
    # Concat along dim 0 produces a leading-dim multiple equal to the DP axis,
    # but since the activation was replicated across DP, slice back.
    dp = mesh_shape[0]
    if dp > 1 and full.shape[0] % dp == 0:
        per = full.shape[0] // dp
        full = full[:per]
    return full

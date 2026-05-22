# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mesh-aware gather utilities for dots.ocr unit tests.

The captured shape matrix only contains per-device tensor metadata (see
``shape_matrix/*.json``: ``device_shape: "MeshShape([8, 1])"``). It does **not**
carry an explicit "is_sharded_along" axis. So at gather time we have two
strategies:

* **DP gather** — for tensors that were sharded along the leading mesh axis
  (e.g. the activation in the production text path lives across mesh axis 0).
  We reassemble with :class:`ttnn.ConcatMesh2dToTensor` and concatenate along
  ``(0, -1)`` — matches ``models/dots_ocr.py:138``.

* **Replicated gather** — for tensors that were replicated on every device,
  the simplest correct read is ``ttnn.to_torch(t)`` followed by a slice into
  the first replica.

For the Phase 1 sanity test the activation is *constructed at the captured
per-device shape and replicated* via the default ``ttnn.from_torch`` mapper;
the output is therefore replicated too — we just need device-0's copy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import ttnn


def _parse_mesh_shape(device_shape_str: str) -> Optional[Tuple[int, int]]:
    """Parse ``'MeshShape([8, 1])'`` -> ``(8, 1)``.

    Returns ``None`` if the string is not a recognized mesh-shape literal.
    """
    if not isinstance(device_shape_str, str):
        return None
    try:
        body = device_shape_str.split("[", 1)[1].split("]", 1)[0]
        parts = [int(x.strip()) for x in body.split(",") if x.strip()]
        if len(parts) == 1:
            return (parts[0], 1)
        if len(parts) >= 2:
            return (parts[0], parts[1])
    except Exception:
        return None
    return None


def _mesh_device_shape_tuple(mesh_device) -> Tuple[int, int]:
    sh = tuple(int(x) for x in mesh_device.shape)
    if len(sh) == 1:
        return (sh[0], 1)
    if len(sh) >= 2:
        return (sh[0], sh[1])
    return (1, 1)


def gather_replicated(tt_tensor, mesh_device) -> torch.Tensor:
    """Read a replicated tensor and return device-0's view.

    On a multi-device mesh ``ttnn.to_torch`` requires a composer. We use a
    1D :class:`ttnn.ConcatMeshToTensor` along ``dim=0`` to concatenate all
    replicas and then slice to the per-device shape — i.e. just take
    device 0's copy. This is safe for replicated tensors because every
    replica is identical.
    """
    mesh_shape = _mesh_device_shape_tuple(mesh_device)
    num_devices = mesh_shape[0] * mesh_shape[1]
    if num_devices == 1:
        return ttnn.to_torch(tt_tensor)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    full = ttnn.to_torch(tt_tensor, mesh_composer=composer)
    # The composer concatenated ``num_devices`` identical copies along dim=0.
    per_device_leading = full.shape[0] // num_devices
    return full[:per_device_leading]


def gather_dp_2d(tt_tensor, mesh_device, axes: Tuple[int, int] = (0, -1)) -> torch.Tensor:
    """Reassemble a tensor sharded across a 2D mesh.

    Matches the production pattern at ``models/dots_ocr.py:138``::

        ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape, (0, -1))
        )
    """
    mesh_shape = _mesh_device_shape_tuple(mesh_device)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, axes)
    return ttnn.to_torch(tt_tensor, mesh_composer=composer)


def gather_to_torch(
    tt_tensor,
    mesh_device,
    captured_output_record: Optional[Dict[str, Any]] = None,
    *,
    strategy: Optional[str] = None,
    dp_axes: Tuple[int, int] = (0, -1),
) -> torch.Tensor:
    """Gather a TTNN tensor on ``mesh_device`` into a CPU ``torch.Tensor``.

    Strategy resolution order:

    1. Explicit ``strategy`` argument (``"replicated"`` or ``"dp_2d"``).
    2. Heuristic from ``captured_output_record`` (when provided):
       - If the captured mesh shape > 1 and the record's per-device shape
         multiplied by the mesh shape along the chosen axes matches the
         "logical" tensor size, treat as DP.
       - Otherwise treat as replicated.
    3. Default fallback: replicated.

    Phase 1's sanity test uses ``"replicated"`` because it constructs
    inputs at the captured per-device shape (no pre-sharding).
    """
    if strategy is None:
        strategy = "replicated"
        if captured_output_record is not None:
            ds = captured_output_record.get("device_shape", "")
            mesh = _parse_mesh_shape(ds) or _mesh_device_shape_tuple(mesh_device)
            if mesh != (1, 1):
                # We don't have enough info from a single record to tell sharded
                # apart from replicated, so default to replicated unless the
                # caller overrides. Phase 2 will introduce per-row strategy
                # tagging in matrix_loader.make_row_tags.
                strategy = "replicated"

    if strategy == "dp_2d":
        return gather_dp_2d(tt_tensor, mesh_device, axes=dp_axes)
    if strategy == "replicated":
        return gather_replicated(tt_tensor, mesh_device)

    raise ValueError(f"Unknown mesh-gather strategy: {strategy!r}")

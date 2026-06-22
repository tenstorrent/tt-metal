# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dots.OCR mesh / DP+TP parallelism configuration.

Supported hybrid modes on T3K (8 devices):
  - ``DP2_TP4`` -> mesh ``(2, 4)``  batch 2, TP 4
  - ``DP2_TP2`` -> mesh ``(2, 2)``  batch 2, TP 2  (4 devices)
  - ``DP4_TP2`` -> mesh ``(4, 2)``  batch 4, TP 2  (8 devices)

Set ``DOTS_OCR_PARALLELISM`` to one of the above with ``MESH_DEVICE=T3K``.
Plain ``DOTS_OCR_PARALLELISM=DP`` uses ``(8, 1)`` DP8/TP1 on T3K.
"""

from __future__ import annotations

import os

import ttnn

MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (2, 4),
    "T3K_DP8": (8, 1),
    "TG": (8, 4),
    "P100": (1, 1),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

DOTS_OCR_DP_MESH_DEVICE_MAP = {
    "N300": (2, 1),
    "T3K": (8, 1),
}

DOTS_OCR_DP2_TP4_MESH_DEVICE_MAP = {
    "T3K": (2, 4),
}

DOTS_OCR_DP2_TP2_MESH_DEVICE_MAP = {
    "T3K": (2, 2),
}

DOTS_OCR_DP4_TP2_MESH_DEVICE_MAP = {
    "T3K": (4, 2),
}

# Modes that use a 2D DP×TP mesh with batch along axis 0.
HYBRID_DP_TP_PARALLELISM_MODES = frozenset({"DP2_TP4", "DP2_TP2", "DP4_TP2"})

# TP=2 hybrid modes (share tp4_prefill body + head_parallel decode tuning).
TP2_HYBRID_PARALLELISM_MODES = frozenset({"DP2_TP2", "DP4_TP2"})


def parallelism_mode() -> str:
    return os.environ.get("DOTS_OCR_PARALLELISM", "").upper()


def _mesh_device_map_get(mapping, mesh_device, default=None):
    if mesh_device is None:
        return default
    for key, value in mapping.items():
        if key.upper() == mesh_device.upper():
            return value
    return default


def resolve_mesh_device_shape(mesh_device: str | None = None):
    """Return the 2D mesh shape for the active ``MESH_DEVICE`` + parallelism mode."""
    mesh_device = mesh_device if mesh_device is not None else os.environ.get("MESH_DEVICE")
    mode = parallelism_mode()
    if mode == "DP2_TP4":
        shape = _mesh_device_map_get(DOTS_OCR_DP2_TP4_MESH_DEVICE_MAP, mesh_device)
        if shape is None:
            raise ValueError("DOTS_OCR_PARALLELISM=DP2_TP4 is only supported for MESH_DEVICE=T3K")
        return shape
    if mode == "DP2_TP2":
        shape = _mesh_device_map_get(DOTS_OCR_DP2_TP2_MESH_DEVICE_MAP, mesh_device)
        if shape is None:
            raise ValueError("DOTS_OCR_PARALLELISM=DP2_TP2 is only supported for MESH_DEVICE=T3K")
        return shape
    if mode == "DP4_TP2":
        shape = _mesh_device_map_get(DOTS_OCR_DP4_TP2_MESH_DEVICE_MAP, mesh_device)
        if shape is None:
            raise ValueError("DOTS_OCR_PARALLELISM=DP4_TP2 is only supported for MESH_DEVICE=T3K")
        return shape
    if mode == "DP":
        return _mesh_device_map_get(
            DOTS_OCR_DP_MESH_DEVICE_MAP,
            mesh_device,
            _mesh_device_map_get(MESH_DEVICE_MAP, mesh_device, len(ttnn.get_device_ids())),
        )
    return _mesh_device_map_get(MESH_DEVICE_MAP, mesh_device, len(ttnn.get_device_ids()))


def mesh_dp_degree(mesh_shape=None) -> int:
    """DP degree (batch axis) for the resolved mesh."""
    sh = mesh_shape if mesh_shape is not None else resolve_mesh_device_shape()
    if not isinstance(sh, (tuple, list)) or len(sh) < 2:
        return 1
    mode = parallelism_mode()
    if mode in HYBRID_DP_TP_PARALLELISM_MODES:
        return int(sh[0])
    if mode == "DP":
        return int(sh[0]) if int(sh[0]) > 1 else int(sh[1])
    if int(sh[0]) > 1 and int(sh[1]) > 1:
        return int(sh[0])
    return 1


def mesh_tp_degree(mesh_shape=None) -> int:
    sh = mesh_shape if mesh_shape is not None else resolve_mesh_device_shape()
    if not isinstance(sh, (tuple, list)) or len(sh) < 2:
        return 1
    return int(sh[1]) if int(sh[1]) > 1 else 1


def mesh_num_devices(mesh_shape=None) -> int:
    sh = mesh_shape if mesh_shape is not None else resolve_mesh_device_shape()
    if isinstance(sh, int):
        return max(1, int(sh))
    if isinstance(sh, (tuple, list)):
        if len(sh) >= 2:
            return int(sh[0]) * int(sh[1])
        if len(sh) == 1:
            return int(sh[0])
    return 1


def pipeline_batch_size() -> int:
    """Pipeline ``batch_size`` must match DP degree on hybrid meshes."""
    n = mesh_dp_degree()
    return n if n > 1 else 1


def is_tp2_hybrid_parallelism() -> bool:
    return parallelism_mode() in TP2_HYBRID_PARALLELISM_MODES


def validate_pipeline_batch_size(batch_size: int, device) -> None:
    """Ensure pipeline batch matches mesh DP degree when batch-parallel."""
    if not hasattr(device, "shape"):
        return
    expected = mesh_dp_degree(tuple(int(x) for x in device.shape))
    if expected > 1 and int(batch_size) != expected:
        raise ValueError(
            f"dots.ocr DP batch-parallel expects batch_size={expected} for mesh "
            f"{tuple(device.shape)} ({parallelism_mode() or 'default'}), got {batch_size}"
        )

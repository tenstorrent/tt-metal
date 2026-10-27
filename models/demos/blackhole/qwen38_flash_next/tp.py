# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import os

TP_SIZES = (4, 8)
# The MESH_DEVICE names (as tt_transformers and vLLM read them) of the meshes the model opens, as (rows, columns).
MESH_DEVICE_SHAPES = {"P150x4": (1, 4), "P150x8": (1, 8), "P300x2": (1, 4)}


def mesh_device_shape(value):
    """The mesh a ``MESH_DEVICE`` value names: a name in ``MESH_DEVICE_SHAPES`` or a ``(rows, columns)`` literal;
    unset or empty, the 1x4 mesh."""

    if not value:
        return (1, 4)
    if value in MESH_DEVICE_SHAPES:
        return MESH_DEVICE_SHAPES[value]
    try:
        shape = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        shape = None
    if not (isinstance(shape, tuple) and len(shape) == 2 and all(type(extent) is int for extent in shape)):
        raise ValueError(
            f"MESH_DEVICE must be one of {sorted(MESH_DEVICE_SHAPES)} or a (rows, columns) tuple, got {value!r}"
        )
    return shape


def _tp_size() -> int:
    value = os.environ.get("MESH_DEVICE")
    rows, columns = mesh_device_shape(value)
    if rows != 1 or columns not in TP_SIZES:
        raise ValueError(f"MESH_DEVICE must name a 1x4 or 1x8 mesh, got {value!r}")
    return columns


TP_SIZE = _tp_size()

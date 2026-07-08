# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Submesh carving for the standalone pi0.5 streamed-denoise port.

``carve_n_submeshes(parent, n)`` carves ``n`` 1x1 submeshes from ``parent`` (row-major,
matching the proven ``carve_four_submeshes`` / tt_bh_glx ``mesh_setup._carve_per_chip``
form), with a LOUD error when more submeshes are requested than the parent has devices
(the SC2/SC6 reconciliation: the Galaxy denoise_submesh is (6,1)=6 chips and supports at
most 4-way; P150x8 requires a fresh >=8-chip PARENT). ZERO tt_symbiote imports.
"""
from __future__ import annotations

import ttnn

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"


def carve_n_submeshes(parent, n):
    rows, cols = (int(d) for d in parent.shape)
    if rows * cols < n:
        raise ValueError(
            f"parent mesh {tuple(parent.shape)} has {rows * cols} devices, cannot carve {n} submeshes. "
            f"P150x8 (n=8) requires a >=8-chip PARENT mesh; the Galaxy denoise_submesh is (6,1)=6 "
            f"chips and supports at most 4-way. Open a fresh >=8-chip mesh for 8-way (see plan §7.0)."
        )
    coords = [(r, c) for r in range(rows) for c in range(cols)][:n]  # row-major (matches carve_four)
    return tuple(parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c)) for (r, c) in coords)


def carve_four_submeshes(parent):
    return carve_n_submeshes(parent, 4)

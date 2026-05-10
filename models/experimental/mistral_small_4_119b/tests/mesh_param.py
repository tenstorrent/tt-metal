# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mesh device request parameter for Mistral-Small-4-119B tests.

Returns the mesh shape tuple used by the ``mesh_device`` pytest fixture.
The default is (1, 2) — a single-row, two-column BlackHole (P300 × 2) mesh.

Override via env var ``MESH_DEVICE``:
    P300x2   → (1, 2)   (default; 2 × BlackHole P300)
    P150x4   → (1, 4)   (4 × BlackHole P150)
    P150x2   → (1, 2)
    N300     → (1, 2)   (2 × WH N300)
    T3K      → (1, 4)   (WH T3000, 4-chip rack half)
    single   → (1, 1)   (single chip; E2E smoke uses disabled fabric for 1×1 — no Ethernet mesh required)
"""

import os


def mesh_device_request_param() -> tuple[int, int]:
    """Return the (rows, cols) mesh shape for the ``mesh_device`` fixture."""
    env = os.environ.get("MESH_DEVICE", "P300x2").strip().upper()

    _MAP = {
        "P300X2": (1, 2),
        "P300": (1, 2),
        "P150X4": (1, 4),
        "P150X2": (1, 2),
        "N300": (1, 2),
        "T3K": (1, 4),
        "SINGLE": (1, 1),
    }
    return _MAP.get(env, (1, 2))

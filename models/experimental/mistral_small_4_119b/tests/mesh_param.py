# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mesh device request parameter for Mistral-Small-4-119B tests.

Returns the mesh shape tuple used by the ``mesh_device`` pytest fixture.
This port targets the 1×8 BlackHole P150 mesh (P150x8) exclusively, so the
shape is always (1, 8). Set ``MESH_DEVICE=P150x8`` when running.
"""


def mesh_device_request_param() -> tuple[int, int]:
    """Return the (rows, cols) mesh shape for the ``mesh_device`` fixture (P150x8)."""
    return (1, 8)

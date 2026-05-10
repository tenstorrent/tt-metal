# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared ``MESH_DEVICE`` → root ``device`` / ``mesh_device`` fixture ``request.param``."""

import os

import ttnn


def mesh_device_request_param():
    """Tuple mesh shape or 1×N device count for ``@pytest.mark.parametrize(..., indirect=True)``."""
    mapping = {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
        "P150x8": (1, 8),
    }
    name = os.environ.get("MESH_DEVICE")
    if name in mapping:
        return mapping[name]
    return len(ttnn.get_device_ids())

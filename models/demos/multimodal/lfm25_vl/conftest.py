# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import ttnn


@pytest.fixture
def device_params(request, galaxy_type):
    # Get param dict passed in from test parametrize (or default to empty dict)
    params = getattr(request, "param", {}).copy()

    mesh_device = {
        "N150": (1, 1),
        "N300": (1, 2),
        "P150": (1, 1),
        "P300": (1, 2),
    }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] == True:
            params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D

    return params

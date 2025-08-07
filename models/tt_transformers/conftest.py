# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest

import ttnn
from models.utility_functions import is_blackhole


@pytest.fixture
def device_params(request, galaxy_type):
    # Get param dict passed in from test parametrize (or default to empty dict)
    params = getattr(request, "param", {}).copy()

    mesh_device = os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        # TODO: 26411
        # Remove this blackhole condition once fabric CCLs are working on blackhole
        if is_blackhole() or is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] == True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )

    return params

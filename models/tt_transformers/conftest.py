# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest

import ttnn


@pytest.fixture
def device_params(request, galaxy_type):
    # Get param dict passed in from test parametrize (or default to empty dict)
    params = getattr(request, "param", {}).copy()

    mesh_device_env = os.environ.get("MESH_DEVICE")
    if mesh_device_env == "TG":
        raise ValueError(
            "MESH_DEVICE=TG corresponds to a 32-device mesh (8x4), which is not supported. "
            "Please select a supported mesh configuration (e.g., N150, N300, N150x4, or T3K)."
        )

    mesh_device = {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "T3K": (1, 8),
    }.get(mesh_device_env, len(ttnn.get_device_ids()))
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] == True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )

    return params

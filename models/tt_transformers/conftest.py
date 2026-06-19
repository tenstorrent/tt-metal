# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest

import ttnn
from models.demos.utils.trace_region_sizes import apply_trace_model_key


@pytest.fixture
def device_params(request, galaxy_type):
    # Get param dict passed in from test parametrize (or default to empty dict).
    # apply_trace_model_key resolves TRACE_MODEL_KEY_PARAM -> trace_region_size from
    # the centralized YAML (no-op when the key is absent).
    params = apply_trace_model_key(getattr(request, "param", {}).copy())

    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4), "P150x8": (1, 8)}.get(
        os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    )
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] == True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )

    return params

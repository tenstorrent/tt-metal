# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest

import ttnn


@pytest.fixture
def device_params(request, galaxy_type):
    # Get param dict passed in from test parametrize (or default to empty dict).
    # Any TRACE_MODEL_KEY_PARAM is left in place; the mesh_device fixture resolves it
    # to trace_region_size using the logical submesh SKU.
    params = getattr(request, "param", {}).copy()

    # Keep the Blackhole SKUs in step with the map in demo/simple_text_demo.py. A name
    # missing here falls through to the physical device count, so a single-chip request
    # such as MESH_DEVICE=P150 on a multi-chip host is read as multi-device and fabric
    # is started across the whole cluster for a 1x1 mesh, which then times out in
    # fabric router sync.
    mesh_device = {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
        "P150": (1, 1),
        "P300": (1, 2),
        "P150x4": (1, 4),
        "P150x8": (1, 8),
        "BHGLX": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] == True:
            cluster_type = ttnn.cluster.get_cluster_type()
            if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
                # The 8x4 decode path uses Ring collectives along both mesh axes.
                params["fabric_config"] = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
            else:
                params["fabric_config"] = (
                    ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
                )

    return params

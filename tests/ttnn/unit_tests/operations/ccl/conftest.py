# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Galaxy-style fabric resolution for CCL tests (matches models/demos/llama3_70b_galaxy/conftest.py)."""

import os

import pytest
import ttnn


def resolve_galaxy_fabric_config(galaxy_type=None):
    """
    WH 6U Galaxy → FABRIC_1D_RING. BH 6U UBB (BLACKHOLE_GALAXY) and TG/loudbox 4U → FABRIC_1D.
    BH auto-discovered eth graphs are mesh (mixed degree 3/4), not a uniform ring graph.
    """
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
    except (IndexError, KeyError, RuntimeError):
        cluster_type = None
    if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
        return ttnn.FabricConfig.FABRIC_1D
    if galaxy_type == "6U" or cluster_type == ttnn.cluster.ClusterType.GALAXY:
        return ttnn.FabricConfig.FABRIC_1D_RING
    return ttnn.FabricConfig.FABRIC_1D


@pytest.fixture
def galaxy_type():
    """
    Galaxy form factor for model mesh layout. Prefer GALAXY_TYPE env.
    BH UBB and WH Galaxy are 6U; TG / loudbox-style hosts are 4U.
    """
    env = os.environ.get("GALAXY_TYPE")
    if env:
        return env
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
    except (IndexError, KeyError, RuntimeError):
        return None
    if cluster_type in (ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.BLACKHOLE_GALAXY):
        return "6U"
    if cluster_type == ttnn.cluster.ClusterType.TG:
        return "4U"
    return None


@pytest.fixture
def device_params(request, galaxy_type):
    params = getattr(request, "param", {}).copy()
    if params.get("fabric_config") is True:
        params["fabric_config"] = resolve_galaxy_fabric_config(galaxy_type)
    return params

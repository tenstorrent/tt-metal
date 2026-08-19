# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared arch-detection helpers for the Qwen3-32B galaxy tests.

The attention, MLP, decoder, prefill and e2e-accuracy tests all select their execution path
(Wormhole prefetcher vs Blackhole no-prefetcher) and fabric config from the detected architecture.
Keeping the detection here avoids the test files drifting apart.
"""

import os
import pytest
import ttnn


def is_blackhole_galaxy():
    """Return True when running on a Blackhole Galaxy, False otherwise.

    Detection order: cluster type -> ARCH_NAME / arch name.
    """
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
        if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
            return True
        if cluster_type in (ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG):
            return False
    except Exception:
        pass
    arch = os.environ.get("ARCH_NAME", "")
    if not arch:
        try:
            arch = ttnn.get_arch_name()
        except Exception:
            arch = ""
    return "blackhole" in arch.lower()


# Detected once at import so pytest parameters (fabric config, batch/seq) resolve at collection time.
IS_BLACKHOLE = is_blackhole_galaxy()

# The 8x4 Blackhole Galaxy decode path runs column-axis (cluster_axis=1) collectives on device, which
# requires a 2D-torus fabric (FABRIC_1D / FABRIC_1D_RING throw `IndexError: map::at` on the cross-column
# route). Wormhole keeps main's fabric_config=True.
DECODE_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_XY if IS_BLACKHOLE else True

# Prefill collectives need the same 2D-torus fabric on Blackhole; Wormhole keeps main's FABRIC_1D_RING.
PREFILL_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_XY if IS_BLACKHOLE else ttnn.FabricConfig.FABRIC_1D_RING

# Every Qwen3-32B galaxy suite hard-parametrizes mesh_device=(8, 4), i.e. a 32-device Galaxy mesh.
# On a machine that exposes fewer devices (e.g. a degraded Blackhole Galaxy that discovers only a
# [2, 2] mesh) open_mesh_device raises `TT_FATAL: requested_size <= system_size` and the whole leg
# fails instead of being skipped as "not applicable for this machine". Guard the suites so they skip
# cleanly when the required device count is not available.
REQUIRED_NUM_DEVICES = 32


def _available_num_devices():
    try:
        return ttnn.get_num_devices()
    except Exception:
        return 0


# Module-level marker; apply as `pytestmark = requires_galaxy_mesh` in each Qwen galaxy suite.
requires_galaxy_mesh = pytest.mark.skipif(
    _available_num_devices() < REQUIRED_NUM_DEVICES,
    reason=(
        f"Qwen3-32B galaxy tests require a {REQUIRED_NUM_DEVICES}-device Galaxy mesh (8x4); "
        f"only {_available_num_devices()} devices available. Not applicable for this machine."
    ),
)

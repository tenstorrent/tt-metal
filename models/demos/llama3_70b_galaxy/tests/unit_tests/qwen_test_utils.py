# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared arch-detection helpers for the Qwen3-32B galaxy unit tests.

Both the attention and MLP unit tests select their execution path (Wormhole prefetcher vs
Blackhole no-prefetcher) from the detected architecture. Keeping the detection here avoids the
two test files drifting apart.
"""

import os
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

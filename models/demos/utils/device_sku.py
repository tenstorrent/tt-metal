#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def cluster_type_to_sku_name(cluster_type: ttnn.cluster.ClusterType) -> str:
    """
    Convert ttnn cluster enum to canonical centralized-target SKU key.
    """
    if not isinstance(cluster_type, ttnn.cluster.ClusterType):
        raise TypeError("cluster_type must be ttnn.cluster.ClusterType, " "for example ttnn.cluster.ClusterType.N150")

    mapping: dict[ttnn.cluster.ClusterType, str] = {
        # Wormhole
        ttnn.cluster.ClusterType.N150: "wh_n150",
        ttnn.cluster.ClusterType.N300: "wh_n300",
        ttnn.cluster.ClusterType.T3K: "wh_llmbox_perf",
        ttnn.cluster.ClusterType.GALAXY: "wh_galaxy_perf",
        ttnn.cluster.ClusterType.TG: "wh_galaxy_perf",
        # Blackhole
        ttnn.cluster.ClusterType.P100: "bh_p100",
        ttnn.cluster.ClusterType.P150: "bh_p150",
        ttnn.cluster.ClusterType.P300: "bh_p300",
        ttnn.cluster.ClusterType.P300_X2: "bh_quietbox_2",
        ttnn.cluster.ClusterType.BLACKHOLE_GALAXY: "bh_galaxy_perf",
    }

    if cluster_type in mapping:
        return mapping[cluster_type]
    raise ValueError(f"Unsupported ClusterType: {cluster_type}")


def get_current_device_sku_name() -> str:
    return cluster_type_to_sku_name(ttnn.cluster.get_cluster_type())

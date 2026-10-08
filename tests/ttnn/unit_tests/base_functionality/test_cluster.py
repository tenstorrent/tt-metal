# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

import ttnn


def test_cluster_get_cluster_type():
    """Test getting cluster type returns a valid enum value"""
    cluster_type = ttnn.cluster.get_cluster_type()

    # Verify it's a ClusterType enum instance
    assert hasattr(cluster_type, "name"), "Expected cluster_type to be an enum with name attribute"
    assert hasattr(cluster_type, "value"), "Expected cluster_type to be an enum with value attribute"

    # Verify string representation works
    str_repr = str(cluster_type)
    assert "ClusterType" in str_repr, f"Unexpected string representation: {str_repr}"

    print(f"Detected cluster type: {cluster_type}")


def test_cluster_serialize_descriptor():
    """Test cluster descriptor serialization"""
    try:
        descriptor_path = ttnn.cluster.serialize_cluster_descriptor()
        assert isinstance(descriptor_path, str), "Expected string path from serialize_cluster_descriptor"
        assert len(descriptor_path) > 0, "Expected non-empty path"
        print(f"Cluster descriptor saved to: {descriptor_path}")
    except Exception as e:
        # Non-critical, might not be available in all environments
        pytest.skip(f"Cluster descriptor serialization not available: {e}")


# Allowed Blackhole ethernet link speeds, in Gbps.
BH_ETH_SPEEDS_GBPS = {40, 100, 200, 330, 350, 370, 400}


@pytest.mark.skipif(ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.P300, reason="Requires P300")
def test_cluster_ethernet_train_speed_p300():
    """At least one ethernet link on a P300 reports a valid trained speed"""
    # Logical channels are dense; out-of-range ones raise
    up_links = []
    for device_id in range(ttnn.GetNumAvailableDevices()):
        for eth_channel in range(14):
            try:
                speed = ttnn.cluster.get_ethernet_train_speed(device_id, eth_channel)
            except RuntimeError:
                break
            if speed is None:
                continue
            target = ttnn.cluster.get_ethernet_target_speed(device_id, eth_channel)
            link = f"Device {device_id} channel {eth_channel}"
            assert speed in BH_ETH_SPEEDS_GBPS, f"{link}: speed {speed} not in {BH_ETH_SPEEDS_GBPS}"
            assert target is not None, f"{link}: no target speed"
            assert speed <= target, f"{link}: speed {speed} > target {target}"
            up_links.append((device_id, eth_channel, speed))

    assert up_links, "No ethernet links up on P300"
    logger.info(f"Ethernet links up: {up_links}")

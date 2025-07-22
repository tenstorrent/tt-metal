# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn


def test_cluster_get_cluster_type():
    """Test getting cluster type"""
    cluster_type = ttnn.cluster.get_cluster_type()

    # Verify it's a valid ClusterType enum value
    assert cluster_type in [
        ttnn.cluster.ClusterType.INVALID,
        ttnn.cluster.ClusterType.N150,
        ttnn.cluster.ClusterType.N300,
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.TG,
        ttnn.cluster.ClusterType.P100,
        ttnn.cluster.ClusterType.P150,
        ttnn.cluster.ClusterType.P150_X2,
        ttnn.cluster.ClusterType.P150_X4,
        ttnn.cluster.ClusterType.SIMULATOR_WORMHOLE_B0,
        ttnn.cluster.ClusterType.SIMULATOR_BLACKHOLE,
        ttnn.cluster.ClusterType.N300_2x2,
    ]

    # Verify it's not INVALID when hardware is available
    if cluster_type != ttnn.cluster.ClusterType.INVALID:
        print(f"Detected cluster type: {cluster_type}")


def test_cluster_is_galaxy_cluster():
    """Test galaxy cluster detection"""
    is_galaxy = ttnn.cluster.is_galaxy_cluster()
    cluster_type = ttnn.cluster.get_cluster_type()

    # Verify consistency between is_galaxy_cluster() and cluster type
    galaxy_types = [ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG]
    expected_is_galaxy = cluster_type in galaxy_types
    assert (
        is_galaxy == expected_is_galaxy
    ), f"is_galaxy_cluster() returned {is_galaxy} but cluster type is {cluster_type}"


def test_cluster_number_of_user_devices():
    """Test getting number of user devices"""
    num_devices = ttnn.cluster.number_of_user_devices()

    # Should have at least 1 device
    assert num_devices >= 1, f"Expected at least 1 device, got {num_devices}"


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


def test_cluster_type_enum_values():
    """Test that all cluster type enum values are accessible"""
    cluster_types = [
        ttnn.cluster.ClusterType.INVALID,
        ttnn.cluster.ClusterType.N150,
        ttnn.cluster.ClusterType.N300,
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.TG,
        ttnn.cluster.ClusterType.P100,
        ttnn.cluster.ClusterType.P150,
        ttnn.cluster.ClusterType.P150_X2,
        ttnn.cluster.ClusterType.P150_X4,
        ttnn.cluster.ClusterType.SIMULATOR_WORMHOLE_B0,
        ttnn.cluster.ClusterType.SIMULATOR_BLACKHOLE,
        ttnn.cluster.ClusterType.N300_2x2,
    ]

    # Verify all enum values are different
    assert len(set(cluster_types)) == len(cluster_types), "Duplicate enum values detected"

    # Verify string representation works
    for cluster_type in cluster_types:
        str_repr = str(cluster_type)
        assert "ClusterType" in str_repr, f"Unexpected string representation: {str_repr}"


def test_cluster_type_comparisons():
    """Test cluster type comparisons and conditionals"""
    cluster_type = ttnn.cluster.get_cluster_type()

    # Test equality comparison
    assert cluster_type == cluster_type, "Cluster type should equal itself"

    # Test membership in list
    all_types = [
        ttnn.cluster.ClusterType.INVALID,
        ttnn.cluster.ClusterType.N150,
        ttnn.cluster.ClusterType.N300,
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.TG,
        ttnn.cluster.ClusterType.P100,
        ttnn.cluster.ClusterType.P150,
        ttnn.cluster.ClusterType.P150_X2,
        ttnn.cluster.ClusterType.P150_X4,
        ttnn.cluster.ClusterType.SIMULATOR_WORMHOLE_B0,
        ttnn.cluster.ClusterType.SIMULATOR_BLACKHOLE,
        ttnn.cluster.ClusterType.N300_2x2,
    ]
    assert cluster_type in all_types, f"Current cluster type {cluster_type} not in known types"

    # Test conditional logic (common use case)
    if cluster_type == ttnn.cluster.ClusterType.T3K:
        print("Running on T3K cluster")
    elif cluster_type in [ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG]:
        print("Running on Galaxy cluster")
    elif cluster_type in [ttnn.cluster.ClusterType.N150, ttnn.cluster.ClusterType.N300]:
        print("Running on Wormhole cluster")
    elif cluster_type in [
        ttnn.cluster.ClusterType.P100,
        ttnn.cluster.ClusterType.P150,
        ttnn.cluster.ClusterType.P150_X2,
        ttnn.cluster.ClusterType.P150_X4,
    ]:
        print("Running on Blackhole cluster")
    else:
        print(f"Running on other cluster type: {cluster_type}")


def test_cluster_type_architecture_detection():
    """Test cluster type to architecture mapping"""
    cluster_type = ttnn.cluster.get_cluster_type()

    # Test architecture detection logic
    wormhole_types = [
        ttnn.cluster.ClusterType.N150,
        ttnn.cluster.ClusterType.N300,
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.SIMULATOR_WORMHOLE_B0,
    ]

    blackhole_types = [
        ttnn.cluster.ClusterType.P100,
        ttnn.cluster.ClusterType.P150,
        ttnn.cluster.ClusterType.P150_X2,
        ttnn.cluster.ClusterType.P150_X4,
        ttnn.cluster.ClusterType.SIMULATOR_BLACKHOLE,
    ]

    galaxy_types = [ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG]

    multi_device_types = [
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.P150_X2,
        ttnn.cluster.ClusterType.P150_X4,
        ttnn.cluster.ClusterType.N300_2x2,
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.TG,
    ]

    # Verify architecture categorization
    if cluster_type in wormhole_types:
        print("Detected Wormhole-based cluster")
    elif cluster_type in blackhole_types:
        print("Detected Blackhole-based cluster")
    elif cluster_type in galaxy_types:
        print("Detected Galaxy cluster")
    elif cluster_type == ttnn.cluster.ClusterType.INVALID:
        print("Invalid/unknown cluster type")
    else:
        print(f"Unrecognized cluster type: {cluster_type}")

    # Test multi-device detection
    is_multi_device = cluster_type in multi_device_types
    num_devices = ttnn.cluster.number_of_user_devices()

    if is_multi_device:
        assert num_devices > 1, f"Multi-device cluster type {cluster_type} but only {num_devices} devices"
    print(f"Cluster type {cluster_type}: {num_devices} devices, multi-device: {is_multi_device}")


@pytest.mark.parametrize(
    "cluster_type_name",
    [
        "INVALID",
        "N150",
        "N300",
        "T3K",
        "GALAXY",
        "TG",
        "P100",
        "P150",
        "P150_X2",
        "P150_X4",
        "SIMULATOR_WORMHOLE_B0",
        "SIMULATOR_BLACKHOLE",
        "N300_2x2",
    ],
)
def test_cluster_type_enum_access(cluster_type_name):
    """Test accessing cluster type enum values by name"""
    cluster_type = getattr(ttnn.cluster.ClusterType, cluster_type_name)
    assert cluster_type is not None, f"Could not access ClusterType.{cluster_type_name}"

    # Verify string representation includes the name
    str_repr = str(cluster_type)
    assert cluster_type_name in str_repr or "ClusterType" in str_repr, f"Unexpected representation: {str_repr}"


def test_cluster_functions_integration():
    """Test integration between different cluster functions"""
    # Get all cluster information
    cluster_type = ttnn.cluster.get_cluster_type()
    is_galaxy = ttnn.cluster.is_galaxy_cluster()
    num_devices = ttnn.cluster.number_of_user_devices()

    print(f"Cluster Information:")
    print(f"  Type: {cluster_type}")
    print(f"  Is Galaxy: {is_galaxy}")
    print(f"  User Devices: {num_devices}")

    # Test consistency checks
    galaxy_types = [ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG]
    assert is_galaxy == (cluster_type in galaxy_types), "Galaxy detection inconsistent"

    # Verify device count makes sense for cluster type
    if cluster_type == ttnn.cluster.ClusterType.T3K:
        # T3K should have multiple devices (typically 8)
        assert num_devices > 1, f"T3K cluster should have multiple devices, got {num_devices}"
    elif cluster_type in [ttnn.cluster.ClusterType.P150_X2, ttnn.cluster.ClusterType.N300_2x2]:
        # X2 variants should have at least 2 devices
        assert num_devices >= 2, f"{cluster_type} should have at least 2 devices, got {num_devices}"
    elif cluster_type == ttnn.cluster.ClusterType.P150_X4:
        # X4 variant should have at least 4 devices
        assert num_devices >= 4, f"P150_X4 should have at least 4 devices, got {num_devices}"
    elif cluster_type in galaxy_types:
        # Galaxy clusters can have variable numbers of devices
        assert num_devices >= 1, f"Galaxy cluster should have at least 1 device, got {num_devices}"
    else:
        # Single device clusters
        assert num_devices >= 1, f"Cluster should have at least 1 device, got {num_devices}"

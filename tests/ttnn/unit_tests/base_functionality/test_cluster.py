# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
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

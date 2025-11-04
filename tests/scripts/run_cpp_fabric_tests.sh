#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

export TT_METAL_CLEAR_L1=1

cd $TT_METAL_HOME

#############################################
# FABRIC UNIT TESTS                         #
#############################################
echo "Running fabric unit tests now...";

# TODO (issue: #24335) disabled slow dispatch tests for now, need to re-evaluate if need to add in a different pool.
#TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

# Host side tests: Topology Mapping in Control Plane
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*LogicalToPhysicalConversionFixture*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphDescriptorTests*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphValidation*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="RoutingTableValidation*"

# Control Plane tests with mock cluster descriptors
TT_METAL_USE_MGD_1_0=1 TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*SingleGalaxy*:-*MGD2*"
TT_METAL_USE_MGD_1_0=1 TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*T3k*:-*MGD2*"
TT_METAL_USE_MGD_1_0=1 TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="T3kCustomMeshGraphControlPlaneTests*:-*MGD2*"
TT_METAL_USE_MGD_1_0=1 TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*Custom2x2*:-*MGD2*"

# MGD 2.0 Tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"

./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphValidation*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="RoutingTableValidation*"

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*SingleGalaxy*
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*T3k*
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3kCustomMeshGraphControlPlaneTests*
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*Custom2x2*

# Multi-host tests
# Dual Galaxy
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric2DSanity"

# Dual T3K Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestDual2x4ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestDual2x4Fabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestDual2x4Fabric2DSanity"

# Split 2x2 T3K Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x2_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x2_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x2_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestSplit2x2ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x2_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestSplit2x2Fabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x2_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestSplit2x2Fabric2DSanity"

# Big mesh 2x4 T3K Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestBigMesh2x4ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/cmustom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestBigMesh2x4Fabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestBigMesh2x4Fabric2DSanity"

# BHQB4x4 Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4Fabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4Fabric2DSanity"

# Closet Box Tests
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"

# Closet Box 3Pod TT-Switch tests
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchAPIs"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchFabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchFabric2DSanity"

# Topology Mapper tests
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=TopologyMapperTest.T3kMeshGraphTest
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/n300_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=TopologyMapperTest.N300MeshGraphTest
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/p100_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=TopologyMapperTest.P100MeshGraphTest
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.DualGalaxyBigMeshTest"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.BHQB4x4MeshGraphTest"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.ClosetBox3PodTTSwitchHostnameAPIs"

#############################################
# FABRIC SANITY TESTS                       #
#############################################
echo "Running fabric sanity tests now...";

./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml

./build/test/tt_metal/tt_fabric/fabric_elastic_channels_host_test 8 2 16 4352 1 4096 10000 4 4

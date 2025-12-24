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

# Host side tests that require a card: Topology Mapping in Control Plane
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphValidation*"

./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"

tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.BHQB4x4*MeshGraphTest"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.ClosetBox3PodTTSwitchHostnameAPIs"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--mca btl self,tcp --mca btl_tcp_if_include eth0 --tag-output --allow-run-as-root" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.Pinning*"

#############################################
# FABRIC SANITY TESTS                       #
#############################################
echo "Running fabric sanity tests now...";

./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml

./build/test/tt_metal/tt_fabric/fabric_elastic_channels_host_test 8 2 16 4352 1 4096 10000 4 4

#!/bin/bash
# Fabric CPU-only unit test driver (same commands as .github/workflows/fabric-cpu-only-tests-impl.yaml).
# Run from repository root, or from anywhere (script cds to root). Requires a built tree under ./build.
#
# Modes:
#   No args (sequential, default):
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh
#     Runs all 6 groups one after another. Simple, safe, readable output.
#
#   Single group:
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --group unit
#     Runs one group only (same as a single CI matrix job).
#     Groups: unit, phys-grouping, control-plane, t3k, wh-galaxy, bh-galaxy
#
#   Parallel (all groups at once):
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --parallel
#     Runs all 6 groups in parallel via self-invocation. Each group's output is
#     buffered to a temp file; logs are printed sequentially at the end.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  export TT_METAL_HOME="$REPO_ROOT"
fi

# Parse arguments
GROUP="all"
PARALLEL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      if [[ $# -lt 2 ]]; then
        echo "--group requires a value" >&2; exit 1
      fi
      GROUP="$2"; shift 2 ;;
    --parallel) PARALLEL=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

VALID_GROUPS="all unit phys-grouping control-plane t3k wh-galaxy bh-galaxy"
if ! echo "$VALID_GROUPS" | tr ' ' '\n' | grep -qx "$GROUP"; then
  echo "Invalid --group value '$GROUP'. Valid groups: $VALID_GROUPS" >&2; exit 1
fi

run_group() { [[ "$GROUP" == "all" || "$GROUP" == "$1" ]]; }

# When running all groups in parallel, self-invoke once per group in the background.
# Each group's output goes to a temp file; logs are cat'd sequentially at the end
# so parallel execution doesn't mangle the output.
if [[ "$GROUP" == "all" && "$PARALLEL" -eq 1 ]]; then
  GROUPS=(unit phys-grouping control-plane t3k wh-galaxy bh-galaxy)
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT
  pids=()
  for g in "${GROUPS[@]}"; do
    "$0" --group "$g" >"$tmpdir/$g.log" 2>&1 &
    pids+=($!)
  done
  exit_code=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "FAILED: group ${GROUPS[$i]}" >&2; exit_code=1; }
  done
  # Print each group's log sequentially so output is clean and readable
  for g in "${GROUPS[@]}"; do
    echo "════════════════════════════════════════"
    echo " Group: $g"
    echo "════════════════════════════════════════"
    cat "$tmpdir/$g.log"
  done
  exit $exit_code
fi

####################################
# Unit tests
####################################
if run_group "unit"; then

./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="FabricTopologyHelpers*"
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="RoutingTableValidation*"

./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*LogicalToPhysicalConversionFixture*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphDescriptorTests*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySolverTest.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySatEncoderTest.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="PhysicalGroupingDescriptorTests*"

fi # unit

######################################
# Physical Grouping tests
######################################
if run_group "phys-grouping"; then

# Physical Grouping Descriptor tests with real PSDs (using tt-run)
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_galaxy_sp4_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="PhysicalGroupingDescriptorSP4Tests*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/dual_t3k_ci_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="PhysicalGroupingDescriptorDualT3kTests*"

# build_physical_multi_mesh_adjacency_graph with SP4 GLX mock (16 ranks; tt-run)
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_galaxy_sp4_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.BuildPhysicalMultiMeshGraph_WithPGDAndPSD_Sp4Glx*"

# build_physical_multi_mesh_adjacency_graph with single BH galaxy (32 ASICs, no tt-run).
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/single_bh_galaxy_clus_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.BuildPhysicalMultiMeshGraph_WithPGDAndPSD_SingleBHGalaxy_*"

######################################
# Topology Mapper tests
######################################
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="T3kTopologyMapperCustomMapping/*"
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.T3kMeshGraphTest*"
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/n300_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.N300MeshGraphTest"
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/p100_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.P100MeshGraphTest"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.DualGalaxyBigMeshTest"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.BHQB4x4*MeshGraphTest"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.ClosetBox3PodTTSwitchHostnameAPIs"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.Pinning*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.ClosetBoxSuperpod*PolicyTest"

fi # phys-grouping

######################################
# Control Plane / Single Host Tests
######################################
if run_group "control-plane"; then

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*SingleGalaxy*
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*T3k*
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3kCustomMeshGraphControlPlaneTests*
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*Custom2x2*
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2xp150_disconnected_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestControlPlaneInitNoMGD
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/4xn300_disconnected_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestControlPlaneInitNoMGD
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_galaxy_xyz_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_galaxy_xyz_cluster_desc.yaml TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_galaxy_xyz_cluster_desc.yaml TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

fi # control-plane

######################################
# T3K Tests
######################################
if run_group "t3k"; then

# Dual T3K Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/dual_t3k_ci_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestDual2x4ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestDual2x4ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestDual2x4Fabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestDual2x4Fabric2DSanity"

# Split 2x2 T3K Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestSplit2x2ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestSplit2x2Fabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestSplit2x2Fabric2DSanity"

# T3K 2x2 Assign Z Direction Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_assign_z_direction_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.T3K2x2AssignZDirectionControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_assign_z_direction_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.T3K2x2AssignZDirectionFabric2DSanity"

# Big mesh 2x4 T3K Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestBigMesh2x4ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestBigMesh2x4Fabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests  --gtest_filter="MultiHost.TestBigMesh2x4Fabric2DSanity"

# BHQB4x4 Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4ControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4RelaxedControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4Fabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4Fabric2DSanity"

# Closet Box Tests
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"

# Closet Box 3Pod TT-Switch tests
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_3pod_ttswitch_mgd.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_3pod_ttswitch_mgd.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchAPIs"

fi # t3k

######################################
# WH Galaxy Tests
######################################
if run_group "wh-galaxy"; then

# Dual Galaxy
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric2DSanity"

# 6U Split Galaxy tests (8x2 and 4x4)
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_8x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test6uSplit8x2ControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x4_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test6uSplit4x4ControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"

# Quad Galaxy Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_quad_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_quad_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyFabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_quad_host_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyFabric2DSanity"

fi # wh-galaxy

######################################
# BH Galaxy Tests
######################################
if run_group "bh-galaxy"; then

# BH Galaxy 8x4 2x2 Hosts
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_6u_cluster_desc.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_8x4_2x2_hosts_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe"  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

# BH Galaxy per-host-sliced MGDs (noMGD control plane + auto rank-binding mapper).
# Each MGD slices N physical BH galaxies across an MGD host grid; tt-run new mode
# (--mesh-graph-descriptor) auto-generates the rank bindings (the path this PR fixes).
# Consolidated onto two physical mocks: the 1-pod 16x8 cluster hosts the 8x8 / 16x8 shapes (single-galaxy
# slices, dual_1x2, quad_1x2/2x2), and the SP4 GLX cluster hosts the 16x4 / 32x4 shapes (dual_2x2/4x2,
# quad_4x2). single_*/dual_1x2/quad_1x2/quad_2x2 -> 1_pod_16x8; dual_2x2/4x2/quad_4x2 -> sp4_glx.
# Single BH galaxy (32 ASICs)
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_single_pod_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_bh_galaxy_1x1_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_single_pod_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_bh_galaxy_1x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_single_pod_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_bh_galaxy_2x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_single_pod_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
# Dual BH galaxy (64 ASICs, two adjacent SP4 galaxies)
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/1_pod_16x8_bh_galaxy_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_bh_galaxy_1x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_bh_galaxy_2x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_bh_galaxy_4x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
# Quad BH galaxy (128 ASICs): 16x8 torus MGDs on the 6U quad-host torus mock; 32x4 on the SP4 GLX mock.
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/1_pod_16x8_bh_galaxy_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/quad_bh_galaxy_1x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/1_pod_16x8_bh_galaxy_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/quad_bh_galaxy_2x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/quad_bh_galaxy_4x2_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
# Dual-pod 4×16 intermesh, single-pod SP4 mock (ranks 0-3 only)
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_single_pod_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/fabric_cpu_only_blitz_dual_pod_4x16_intermesh_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

# Mock BH galaxy: sub-context 0 = single 4x4 mesh graph, sub-context 1 = dual 2x4 + intermesh (--rank-bindings-mapping).
# Runs split MPI communicators (4 ranks → two sub-contexts × 2 ranks): fabric KV exchange, subcommunicator vs job-world checks, launcher metadata / rank translation.
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml --rank-bindings-mapping tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="MpiSubContext.*"

# BH Galaxy XY Torus Dual Galaxy Multi-host (sub-torus connections)
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/dual_glx_2.5d_torus_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_bh_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*TestBHGalaxyTorusXYControlPlaneQueries*:ControlPlaneFixture.TestGalaxyCornerPinnings"

# 32x4 quad BH Galaxy MGD + SP4 GLX cluster mock (16 ranks)
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyFabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyFabric2DSanity"

# BH Blitz Pipeline Multi-host 48 stage
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineControlPlaneInit"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineFabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineFabric2DSanity"

# Blitz Pipeline 16 stage (CPU-only single-pod mesh graph)
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/fabric_cpu_only_blitz_single_pod_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
# Blitz Pipeline 64 stage (CPU-only superpod / quad-pod Blitz mesh graph)
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/fabric_cpu_only_blitz_superpod_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"

# Dual 4x8 Z-direction fallback (SP4 d04u08 / d05u08 — Z-only connections between galaxies; MGD from rank bindings)
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/dual_4x8_z_fallback_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_4x8_z_fallback_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual4x8ZDirectionFallbackControlPlaneInit"

# Triple Pod 32x4 Quad BH Galaxy MGD + SP4 GLX cluster mock (same mapping as 32x4 quad tests)
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyFabric1DSanity"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyFabric2DSanity"

# BH Dual Galaxy Multi-host
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/dual_bh_galaxy_experimental_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.BHDualGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/dual_bh_galaxy_experimental_cluster_desc_mapping.yaml --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto --mpi-args "--allow-run-as-root --oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.BHDualGalaxyFabric2DSanity"

fi # bh-galaxy

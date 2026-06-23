#!/usr/bin/env bash
# Fabric CPU-only unit test driver (same commands as .github/workflows/fabric-cpu-only-tests-impl.yaml).
# Run from repository root, or from anywhere (script cds to root). Requires a built tree under ./build.
# Requires bash (not sh/dash): bash ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh ...
#
# Modes:
#   No args (sequential, default):
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh
#     Runs all groups one after another. Simple, safe, readable output.
#
#   Single group:
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --group unit
#     Runs one group only (same as a single CI matrix job).
#     Groups: unit, phys-grouping, control-plane, t3k, wh-galaxy,
#       bh-6u, bh-single-galaxy, bh-dual-galaxy,
#       bh-subtorus, bh-subtorus-sc16, bh-sp4-glx, bh-blitz-decode, bh-pod-pipeline, bh-misc
#
#   Parallel (all groups at once):
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --parallel
#     Runs all groups in parallel via self-invocation. Each group's output is
#     buffered to a temp file; logs are printed sequentially at the end.
#
#   Keep going after failures:
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --keep-going
#     Continue running remaining tests after a failure; print all failed commands
#     at the end and exit non-zero if any failed. Combines with --group/--parallel.
#
#   Run or list one specific command (by gtest name, MGD path, etc.):
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --group phys-grouping --grep Sp4Glx_Galaxy1x32
#     ./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --group phys-grouping --grep Sp4Glx --dry-run
#
#   Source for ad-hoc reruns (loads paths, TT_RUN_FLAGS, run_test; does not run tests):
#     source tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh
#     CURRENT_GROUP=phys-grouping
#     run_test tt-run --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" ...
#     run_test uses return (not exit) on failure when sourced so your shell stays open.
#
# Strict mode applies only when executed (not when sourced), so sourcing does not
# enable nounset in your interactive shell.
_fabric_cpu_only_abort() {
  local status=${1:-1}
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    return "$status"
  fi
  exit "$status"
}

if [ -z "${BASH_VERSION:-}" ]; then
  echo "error: this script requires bash (not sh/dash). Run: bash $0 ..." >&2
  _fabric_cpu_only_abort 1
fi
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -eo pipefail
fi

KEEP_GOING=0
DRY_RUN=0
GREP_FILTER=""
GROUP="all"
FAILURES=()
CURRENT_GROUP="all"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  export TT_METAL_HOME="$REPO_ROOT"
fi

# Refresh Phase 1 rank-binding cache on every tt-run (new mode with --mesh-graph-descriptor).
# Ignored for legacy --rank-binding-only invocations; harmless there.
TT_RUN_FLAGS=(--force-rediscovery)

# tt-run argument order (MGD/mock first for readability): --mesh-graph-descriptor, --mock-cluster-rank-binding,
# [--rank-binding | --rank-bindings-mapping], --mpi-args, "${TT_RUN_FLAGS[@]}", then the test binary.

# Mock cluster rank-binding mappings (env names match mapping YAML basenames).
SP4_GLX_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revAB_aisleD/SC16_32x4_revAB_aisleD_mapping.yaml"
SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revAB_aisleD/SC16_32x4_revAB_aisleD_single_pod_mapping.yaml"
BH_110C_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revC_aisleC/SC16_32x4_revC_aisleC_mapping.yaml"
BH_110C_SINGLE_POD_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml"
# Full 20-host subtorus 110C galaxy (hosts bh-glx-110-c01..c10). Used for the 80-stage Blitz decode ring, which
# needs the subtorus wrap-around to close (the non-subtorus 16-host SC16_revC_aisleC above cannot).
BH_110C_SC20_SUBTORUS_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC20_32x4_revC_subtorus_aisleC_mapping.yaml"
# Full 20-host non-subtorus SC20 galaxy (revAB, Aisle C). Same 20-host / 80-mesh scale as the subtorus
# mock above but without the torus wrap-around links.
BH_SC20_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revAB_aisleC/SC20_32x4_revAB_aisleC_mapping.yaml"
# SC16 revC subtorus, Aisle C (16-host / 64-mesh subset of the SC20 revC subtorus Aisle C set).
SC16_REVC_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC16_32x4_revC_subtorus_aisleC_mapping.yaml"
SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml"
POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SP3_16x8_revAB_aisleC/SP3_16x8_revAB_aisleC_1pod_mapping.yaml"
SUBTORUS_SC16_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revC_subtorus_aisleD/SC16_32x4_revC_subtorus_aisleD_mapping.yaml"
DUAL_BH_GALAXY_EXPERIMENTAL_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/blackhole/dual_bh_galaxy_experimental/dual_bh_galaxy_experimental_cluster_desc_mapping.yaml"
DUAL_4X8_Z_FALLBACK_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revAB_aisleD/SC16_32x4_revAB_aisleD_dual_4x8_z_fallback_mapping.yaml"
MOCK_GALAXY_QUAD_2X4_FOUR_RANK_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_6u_cluster_desc/mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml"
BH_GALAXY_SP4_RANK_BINDINGS="tests/tt_metal/distributed/config/bh_galaxy_sp4_rank_bindings.yaml"

# Custom mesh graph descriptor directories (MGD filenames appear inline in commands).
MGD_CUSTOM="tests/tt_metal/tt_fabric/custom_mesh_descriptors"
MGD_SUBTORUS="${MGD_CUSTOM}/subtorus"

# Blitz decode pipeline ring MGDs by stage count (mesh-ring length), reusing the existing in-repo MGDs.
MGD_BLITZ_16="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_single_pod_mesh_graph_descriptor.textproto"
MGD_BLITZ_48="tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto"
MGD_BLITZ_64="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto"
MGD_BLITZ_80="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_supercluster_20.textproto"

GTEST_GALAXY_CORNER_PINNINGS="ControlPlaneFixture.TestGalaxyCornerPinnings"

run_group() {
  if [[ "$GROUP" == "all" || "$GROUP" == "$1" ]]; then
    CURRENT_GROUP="$1"
    return 0
  fi
  return 1
}

# Run one test command; with --keep-going record failures and continue.
# Uses explicit status checks (not ERR trap) because set -e still aborts inside
# if/then blocks even when ERR returns 0.
run_test() {
  local cmd_str
  cmd_str=$(printf '%q ' "$@")
  if [[ -n "$GREP_FILTER" && "$cmd_str" != *"$GREP_FILTER"* ]]; then
    return 0
  fi
  echo "+ [${CURRENT_GROUP}] ${cmd_str% }" >&2
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  "$@"
  local status=$?
  if [[ $status -ne 0 ]]; then
    FAILURES+=("[${CURRENT_GROUP}] exit ${status}: ${cmd_str% }")
    if [[ "$KEEP_GOING" -eq 0 ]]; then
      _fabric_cpu_only_abort "$status"
    fi
  fi
}

print_failure_summary() {
  local failure_count=${#FAILURES[@]}
  if [ "$failure_count" -eq 0 ]; then
    return 0
  fi
  echo "========================================" >&2
  echo " Failed commands (${failure_count}):" >&2
  echo "========================================" >&2
  local failure
  for failure in "${FAILURES[@]}"; do
    printf '%s\n' "$failure" >&2
  done
  _fabric_cpu_only_abort 1
}

# When sourced, stop here: paths/vars/run_test are available; tests are not run.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  return 0
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
    --keep-going) KEEP_GOING=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --grep)
      if [[ $# -lt 2 ]]; then
        echo "--grep requires a value" >&2; exit 1
      fi
      GREP_FILTER="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

CURRENT_GROUP="$GROUP"

VALID_GROUPS="all unit phys-grouping control-plane t3k wh-galaxy bh-6u bh-single-galaxy bh-dual-galaxy bh-subtorus bh-subtorus-sc16 bh-subtorus-sc20 bh-sp4-glx bh-blitz-decode bh-pod-pipeline bh-misc"
if ! echo "$VALID_GROUPS" | tr ' ' '\n' | grep -qx "$GROUP"; then
  echo "Invalid --group value '$GROUP'. Valid groups: $VALID_GROUPS" >&2; exit 1
fi

# When running all groups in parallel, self-invoke once per group in the background.
# Each group's output goes to a temp file; logs are cat'd sequentially at the end
# so parallel execution doesn't mangle the output.
if [[ "$GROUP" == "all" && "$PARALLEL" -eq 1 ]]; then
  GROUPS=(
    unit phys-grouping control-plane t3k wh-galaxy
    bh-6u bh-single-galaxy bh-dual-galaxy
    bh-subtorus bh-subtorus-sc16 bh-subtorus-sc20 bh-sp4-glx bh-blitz-decode bh-pod-pipeline bh-misc
  )
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT
  pids=()
  keep_going_args=()
  if [[ "$KEEP_GOING" -eq 1 ]]; then
    keep_going_args=(--keep-going)
  fi
  for g in "${GROUPS[@]}"; do
    "$0" --group "$g" "${keep_going_args[@]}" >"$tmpdir/$g.log" 2>&1 &
    pids+=($!)
  done
  exit_code=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "FAILED: group ${GROUPS[$i]}" >&2; exit_code=1; }
  done
  # Print each group's log sequentially so output is clean and readable
  for g in "${GROUPS[@]}"; do
    echo "========================================"
    echo " Group: $g"
    echo "========================================"
    cat "$tmpdir/$g.log"
  done
  exit $exit_code
fi

# Test commands below use run_test for failure handling; disable errexit here.
set +e

####################################
# Unit tests
####################################
if run_group "unit"; then

run_test ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="FabricTopologyHelpers*"
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/2x2_n300_cluster_desc/2x2_n300_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="RoutingTableValidation*"

run_test ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*LogicalToPhysicalConversionFixture*"
run_test ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphDescriptorTests*"
run_test ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySolverTest.*"
run_test ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySatEncoderTest.*"
run_test ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.*"
run_test ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="PhysicalGroupingDescriptorTests*"

fi # unit

######################################
# Physical Grouping tests
######################################
if run_group "phys-grouping"; then

# Physical Grouping Descriptor tests with real PSDs (using tt-run)
run_test tt-run --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --rank-binding "${BH_GALAXY_SP4_RANK_BINDINGS}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="PhysicalGroupingDescriptorSP4Tests*"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/dual_t3k_ci/dual_t3k_ci_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="PhysicalGroupingDescriptorDualT3kTests*"

# build_physical_multi_mesh_adjacency_graph with SP4 GLX mock (16 ranks; tt-run)
run_test tt-run --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --rank-binding "${BH_GALAXY_SP4_RANK_BINDINGS}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.BuildPhysicalMultiMeshGraph_WithPGDAndPSD_Sp4Glx*"

# build_physical_multi_mesh_adjacency_graph with single BH galaxy (32 ASICs, torus XY links; no tt-run).
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.BuildPhysicalMultiMeshGraph_WithPGDAndPSD_SingleBHGalaxy_*:PhysicalGroupingDescriptorTests.GetValidGroupingsForMGD_SinglePod4x4LineLinePrefersSingleHost"

######################################
# Topology Mapper tests
######################################
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="T3kTopologyMapperCustomMapping/*"
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.T3kMeshGraphTest*"
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/n300_cluster_desc/n300_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.N300MeshGraphTest"
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.P100MeshGraphTest"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.DualGalaxyBigMeshTest"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_qb_4x4/bh_qb_4x4_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.BHQB4x4*MeshGraphTest"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/wormhole/wh_closetbox/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.ClosetBox3PodTTSwitchHostnameAPIs"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.Pinning*"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/wormhole/wh_closetbox/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.ClosetBoxSuperpod*PolicyTest"

fi # phys-grouping

######################################
# Control Plane / Single Host Tests
######################################
if run_group "control-plane"; then

run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*SingleGalaxy*
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*T3k*
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3kCustomMeshGraphControlPlaneTests*
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/2x2_n300_cluster_desc/2x2_n300_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*Custom2x2*
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/2xp150_disconnected_cluster_desc/2xp150_disconnected_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestControlPlaneInitNoMGD
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/wormhole/4xn300_disconnected_cluster_desc/4xn300_disconnected_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestControlPlaneInitNoMGD
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

fi # control-plane

######################################
# T3K Tests
######################################
if run_group "t3k"; then

# Dual T3K Multi-host
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/dual_t3k_ci/dual_t3k_ci_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual2x4ControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_dual_host/t3k_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual2x4ControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_dual_host/t3k_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual2x4Fabric1DSanity"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_dual_host/t3k_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual2x4Fabric2DSanity"

# Split 2x2 T3K Multi-host
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestSplit2x2ControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestSplit2x2Fabric1DSanity"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestSplit2x2Fabric2DSanity"

# T3K 2x2 Assign Z Direction Multi-host
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_assign_z_direction_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.T3K2x2AssignZDirectionControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_assign_z_direction_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.T3K2x2AssignZDirectionFabric2DSanity"

# Big mesh 2x4 T3K Multi-host
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBigMesh2x4ControlPlaneInit"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBigMesh2x4Fabric1DSanity"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_2x4_big_mesh_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBigMesh2x4Fabric2DSanity"

# BHQB4x4 Multi-host
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_qb_4x4/bh_qb_4x4_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4ControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_qb_4x4/bh_qb_4x4_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4RelaxedControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_qb_4x4/bh_qb_4x4_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4Fabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_qb_4x4/bh_qb_4x4_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHQB4x4Fabric2DSanity"

# Closet Box Tests
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/wormhole/wh_closetbox/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/wormhole/wh_closetbox/wh_closetbox_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/wh_closetbox_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"

# Closet Box 3Pod TT-Switch tests
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_3pod_ttswitch_mgd.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/wormhole/wh_closetbox/wh_closetbox_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_3pod_ttswitch_mgd.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/wormhole/wh_closetbox/wh_closetbox_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchAPIs"

fi # t3k

######################################
# WH Galaxy Tests
######################################
if run_group "wh-galaxy"; then

# Dual Galaxy
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric2DSanity"

# 6U Split Galaxy tests (8x2 and 4x4)
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_8x2_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test6uSplit8x2ControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x4_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test6uSplit4x4ControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"

# Quad Galaxy Multi-host
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_quad_host/6u_quad_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_quad_host/6u_quad_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_quad_host/6u_quad_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyFabric2DSanity"

fi # wh-galaxy

######################################
# BH Galaxy: 6U legacy (8x4 single-host shape)
######################################
if run_group "bh-6u"; then

run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/bh_galaxy_8x4_2x2_hosts_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_6u_cluster_desc/bh_6u_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

fi # bh-6u

######################################
# BH Galaxy: single galaxy (SP4 single-pod mock, 32 ASICs)
# Per-host-sliced MGDs 1x1/1x2/2x2/4x2; dual-pod intermesh; 4-stage Blitz ring.
######################################
if run_group "bh-single-galaxy"; then

run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_1x1_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_dual_pod_4x16_intermesh_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

# TODO: This test is currently disabled because otpimized grouping placements is still not implemented for this case to work
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_single_galaxy_4x2_line_4stage_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"

fi # bh-single-galaxy

######################################
# BH Galaxy: dual galaxy (64 ASICs)
# 1_pod 16x8 torus mock + SP4 GLX dual_2x2 slice; torus XY; experimental; Z fallback.
######################################
if run_group "bh-dual-galaxy"; then

run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/dual_glx_2.5d_torus/dual_glx_2.5d_torus_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*TestBHGalaxyTorusXYControlPlaneQueries*:ControlPlaneFixture.TestGalaxyCornerPinnings"
run_test tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${DUAL_BH_GALAXY_EXPERIMENTAL_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.BHDualGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
run_test tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${DUAL_BH_GALAXY_EXPERIMENTAL_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.BHDualGalaxyFabric2DSanity"
run_test tt-run --mock-cluster-rank-binding "${DUAL_4X8_Z_FALLBACK_CLUSTER_DESC_MAPPING}" --rank-binding tests/tt_metal/distributed/config/dual_4x8_z_fallback_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual4x8ZDirectionFallbackControlPlaneInit"

fi # bh-dual-galaxy

######################################
# BH Galaxy: subtorus Rev C quad mock (128 ASICs)
# Tray-pair PGD, per-host slices, subtorus MGDs, Blitz pipelines, dual_4x16 intermesh.
######################################
if run_group "bh-subtorus"; then

# Tray-pair mapping (8 ASICs: 2x4 RING+RING / 4x2 LINE+RING)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_2x4_ring_ring_1x1_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=MultiHost.TestSubtorusTrayPairMapping
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x2_line_ring_1x1_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=MultiHost.TestSubtorusTrayPairMapping
# Subtorus single-galaxy grouping MGDs (16–32 ASICs)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x8_ring_ring_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestGalaxyCornerPinnings:MultiHost.Test8x4TrayMapping"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_ring_ring_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestGalaxyCornerPinnings:MultiHost.TestSplitHost4x4TrayMapping"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_ring_ring_1x1_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestGalaxyCornerPinnings:MultiHost.TestSplitHost4x4TrayMapping"
# Subtorus Blitz decode pipeline MGDs (16-stage 4x2 RING+LINE)
# Original single-pod Blitz decode MGDs on subtorus mock (16-stage 2x4 pipelines, CPU-only test descriptors)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_single_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.Test2x4GroupingHorizontalTrayMapping"
# Quad-galaxy 4x4 split-host / 8x4 full-galaxy torus pipelines (8- and 4-stage) on subtorus mock
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_pipeline_8stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.TestSplitHost4x4TrayMapping"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_8x4_pipeline_4stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.Test8x4TrayMapping"

# Quad-galaxy mixed 4x8+4x4+4x2 10-stage ring (128 ASICs) on subtorus mock
# Currently disabled because complex heterogeneous multi-stage mesh graphs are not supported yet.
# TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_quad_galaxy_4x8_4x4_4x2_10stage_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:ControlPlaneFixture.TestGalaxyCornerPinnings"
# Quad-galaxy heterogeneous 4x8+4x2 10-stage ring (128 ASICs): 2x 4x8 RING+RING + 8x 4x2 RING+LINE on subtorus mock.
# Homogeneous 4x2 hops use NESW (no assign_z_direction); heterogeneous 4x8<->4x2 hops use assign_z_direction.
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x8_2x4_10stage_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:ControlPlaneFixture.TestGalaxyCornerPinnings"
# Dual 4x16 quad-galaxy intermesh (128 ASICs): M0 1x8 hosts + M1 2x16 hosts, 4 intermesh links
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_4x16_blitz_test.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:ControlPlaneFixture.TestGalaxyCornerPinnings"
# Quad BH galaxy subtorus (128 ASICs, 32x4 RING+RING on quad subtorus mock)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_32x4_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

fi # bh-subtorus

######################################
# BH Galaxy: subtorus SC16 superpod (subtorus_sc16, 16 MPI ranks, 512 ASICs)
# Full 2x4 / 4x4 / 8x4 pipeline rings and 32x4 quad torus on the full subtorus mock.
######################################
if run_group "bh-subtorus-sc16"; then

# 2x4 = 64-stage ring (8 ASICs/stage, 4x2 RING+LINE), 4x4 = 32-stage ring (16 ASICs/stage), 8x4 = 16-stage ring (32 ASICs/stage)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_superpod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SUBTORUS_SC16_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.Test2x4GroupingHorizontalTrayMapping"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_pipeline_32stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SUBTORUS_SC16_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.TestSplitHost4x4TrayMapping"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_8x4_pipeline_16stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SUBTORUS_SC16_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.Test8x4TrayMapping"
# Full 32x4 quad torus (16 MPI ranks, 8x2 host grid)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_32x4_8x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SUBTORUS_SC16_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

fi # bh-subtorus-sc16

######################################
# BH Galaxy: SC20 (20-host) subtorus — same ring tests as bh-subtorus-sc16, scaled to 20 hosts
######################################
if run_group "bh-subtorus-sc20"; then

# 2x4 = 80-stage ring (8 ASICs/stage), 4x4 = 40-stage ring (16 ASICs/stage), 8x4 = 20-stage ring (32 ASICs/stage)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${MGD_BLITZ_80}" --mock-cluster-rank-binding "${BH_110C_SC20_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.Test2x4GroupingHorizontalTrayMapping"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_sc20_4x4_pipeline_40stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${BH_110C_SC20_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.TestSplitHost4x4TrayMapping"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_sc20_8x4_pipeline_20stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${BH_110C_SC20_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.Test8x4TrayMapping"
# Full SC20 subtorus torus: five 32x4 groups wired as a ring (20 hosts)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_sc20_32x4_5group_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${BH_110C_SC20_SUBTORUS_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings

fi # bh-subtorus-sc20

######################################
# BH Galaxy: SP4 GLX quad mock (128 ASICs, 32x4)
# Per-host slices, dual_4x16 intermesh, Blitz, 32x4 torus CP/fabric, triple-pod.
######################################
if run_group "bh-sp4-glx"; then

# Per-host-sliced BH galaxy MGDs (pod MGDs in bh-pod-pipeline group)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
# Dual 4x16 quad-galaxy intermesh (M0 1x8 + M1 2x16 hosts, 4 intermesh links)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_single_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_superpod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
# Llama 8b pod MGDs on the FULL 16-host sp4 system — COMMENTED OUT: the 40-host 2-mesh pod has no valid
# mapping onto a 16-host mock. The single-pod (4-host) versions run in the bh-pod-pipeline group.
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_1x2_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_2x1_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyFabric2DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineFabric2DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyCornerPinnings"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SP4_GLX_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyFabric2DSanity"

fi # bh-sp4-glx

######################################
# BH Galaxy: Blitz decode pipeline coverage (SC16/SC20 x ring-stage counts) + 110C corner pinnings
######################################
if run_group "bh-blitz-decode"; then

# Per-host-sliced BH galaxy MGDs (pod MGDs in bh-pod-pipeline group)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${BH_110C_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${BH_110C_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyCornerPinnings
# Dual 4x16 quad-galaxy intermesh (M0 1x8 + M1 2x16 hosts, 4 intermesh links)
# NOTE: Not yet working for full cluster, this is working for if you specify a single pod, because of placemnet optimizations
#TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_4x16_blitz_test.textproto" --mock-cluster-rank-binding "${BH_110C_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:ControlPlaneFixture.TestGalaxyCornerPinnings"
# ---------------------------------------------------------------------------
# Blitz decode pipeline coverage matrix (ControlPlaneFixture.TestBlitzDecodePipelineBuilder).
# Each physical cluster mock x every ring-stage count that fits its mesh-slot budget:
#   16-host SC16 clusters = 64 mesh slots -> up to 16/48/64 stage rings (80 skipped: 80 > 64).
#   20-host SC20 clusters = 80 mesh slots -> up to 16/48/64/80 stage rings.
# (SC4 / 16-slot is covered via the single-pod groups, not here.) Stage MGDs reuse existing in-repo
# blitz ring descriptors (MGD_BLITZ_*). A ring shorter than the cluster's slot count is a
# ring-embedded-into-a-larger-graph mapping -- the host-minimization SAT case guarded by the conflict
# cap; on non-subtorus clusters the ring's closing hop has no direct link and is routed the long way.
# Per-cluster stage lists below: SC16 revC (non-subtorus) runs 16 + 64 (48 omitted -- times out on this
# mock); the 48-stage ring strands on SC16 revC subtorus aisleC, so that one keeps 16/64.
for entry in \
    "SC16_revAB:${SP4_GLX_CLUSTER_DESC_MAPPING}:16 48 64" \
    "SC16_revC:${BH_110C_CLUSTER_DESC_MAPPING}:16 64" \
    "SC16_revC_subtorus_aisleD:${SUBTORUS_SC16_CLUSTER_DESC_MAPPING}:16 48 64" \
    "SC16_revC_subtorus_aisleC:${SC16_REVC_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}:16" \
    "SC20_revAB:${BH_SC20_CLUSTER_DESC_MAPPING}:16 48 64 80" \
    "SC20_revC_subtorus:${BH_110C_SC20_SUBTORUS_CLUSTER_DESC_MAPPING}:16 48 64 80" ; do
  rest="${entry#*:}"; cluster_map="${rest%%:*}"; stages="${rest#*:}"
  for stage in ${stages}; do
    mgd_var="MGD_BLITZ_${stage}"
    run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${!mgd_var}" --mock-cluster-rank-binding "${cluster_map}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:MultiHost.Test2x4GroupingHorizontalTrayMapping"
  done
done
# Llama 8b pod MGDs on the FULL 16-host 110C system — COMMENTED OUT: the 40-host 2-mesh pod has no valid
# mapping onto a 16-host mock. The single-pod (4-host) 110C versions run in the bh-pod-pipeline group
# (via BH_110C_SINGLE_POD_CLUSTER_DESC_MAPPING).
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_1x2_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${BH_110C_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_2x1_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${BH_110C_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"

fi # bh-blitz-decode

######################################
# BH Galaxy: pod pipeline MGDs (TestGalaxyCornerPinnings)
# Per-host-sliced pod MGDs on SP4 single-pod, 110C single-pod, and quad subtorus mocks (4 ranks).
# 8x16 quad pod MGDs (16x8 device, 128 ASIC) on 1-pod 16x8 SP4 mock (4 ranks).
######################################
if run_group "bh-pod-pipeline"; then

# Per-host-sliced pod MGDs (32–128 ASICs with host slices > 1 rank per slice)
for mock in "${SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING}" "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" "${BH_110C_SINGLE_POD_CLUSTER_DESC_MAPPING}"; do
  # Single-galaxy pod MGDs
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_1x1_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
  # Dual-galaxy pod MGDs (64 ASICs; dual_bh_galaxy_1x2 runs on 16x8 mock below)
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
  # Quad-galaxy pod MGD — 32x4 device (4x32_Mesh PGD), per-host sliced on 4-rank mocks
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/quad_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
done

# 8x16 quad pod MGDs (16x8 RING+RING device, 128 ASIC, 8x16_Mesh PGD)
# dual_bh_galaxy_1x2 (8x8 device, 64 ASIC, 8x8_Mesh PGD) — needs 16x8 pod mock, not single-galaxy single-pod mocks
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/quad_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/quad_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"

# Llama 8b pod MGDs (tt-blaze issue #46935): 40-host, 2-mesh (M0 8 hosts + M1 32 hosts) decode pods.
# These only map onto a SINGLE pod (4-host mocks); on a full 16-host system mock the 40-host pod has no
# valid mapping (the 16-host variants are commented out in the bh-sp4-glx and bh-blitz-decode groups).
# Validated on rev A/B (sp4 single-pod), rev C (110C single-pod), and rev C subtorus (subtorus single-pod).
for mock in "${SP4_GLX_SINGLE_POD_CLUSTER_DESC_MAPPING}" "${SINGLE_POD_32X4_SUBTORUS_CLUSTER_DESC_MAPPING}" "${BH_110C_SINGLE_POD_CLUSTER_DESC_MAPPING}"; do
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_1x2_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_2x1_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_CORNER_PINNINGS}"
done

fi # bh-pod-pipeline

######################################
# BH Galaxy: misc (MPI sub-context split communicators)
######################################
if run_group "bh-misc"; then

# Mock BH galaxy MPI sub-context: sub-context 0 = single 4x4 mesh, sub-context 1 = dual 2x4 + intermesh.
# Runs split MPI communicators (4 ranks → two sub-contexts × 2 ranks): fabric KV exchange, subcommunicator vs job-world checks, launcher metadata / rank translation.
run_test tt-run --mock-cluster-rank-binding "${MOCK_GALAXY_QUAD_2X4_FOUR_RANK_CLUSTER_DESC_MAPPING}" --rank-bindings-mapping tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="MpiSubContext.*"

fi # bh-misc

print_failure_summary

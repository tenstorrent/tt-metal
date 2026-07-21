#!/usr/bin/env bash
# Fabric CPU-only unit test driver (keep in sync with tests/pipeline_reorg/fabric_cpu_only_*_tests.yaml).
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
#       bh-subtorus, bh-subtorus-sc16, bh-subtorus-sc20, bh-sp4-glx, bh-blitz-decode, bh-pod-pipeline, bh-ring-stress, bh-misc
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
#     run_test tt-run --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" ...
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
GREP_EXCLUDE=""
GROUP="all"
FAILURES=()
CURRENT_GROUP="all"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  export TT_METAL_HOME="$REPO_ROOT"
fi

if [[ -z "${DONT_USE_VIRTUAL_ENVIRONMENT:-}" && -f "${REPO_ROOT}/python_env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/python_env/bin/activate"
fi

# Refresh Phase 1 rank-binding cache on every tt-run (new mode with --mesh-graph-descriptor).
# Ignored for legacy --rank-binding-only invocations; harmless there.
TT_RUN_FLAGS=(--force-rediscovery)

# tt-run argument order (MGD/mock first for readability): --mesh-graph-descriptor, --mock-cluster-rank-binding,
# [--rank-binding | --rank-bindings-mapping], --mpi-args, "${TT_RUN_FLAGS[@]}", then the test binary.

# Mock cluster rank-binding mappings. Naming convention:
#   SC<hosts>_REV<AB|C>[_SUBTORUS]_AISLE<C|D>[_SINGLE_POD]_CLUSTER_DESC[_MAPPING]
# where SC<hosts> is the full supercluster host count (SC4 = 4-host single pod, SC16 = 16-host,
# SC20 = 20-host), _SUBTORUS marks torus wrap-around links, _SINGLE_POD marks a 4-host pod subset,
# _MAPPING marks a mock-cluster-rank-binding mapping (no suffix = a single-host cluster_desc yaml).
SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revAB_aisleD/SC16_32x4_revAB_aisleD_mapping.yaml"
SC4_REVAB_AISLED_SINGLE_POD_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revAB_aisleD/SC4_32x4_revAB_aisleD_mapping.yaml"
# revAB subtorus / system-120 (bh-glx-120-*). tt-cluster-descriptors renamed this set to
# SC36_32x4_revAB_subtorus_aisleC, matching tt-metal's subtorus naming.
SUBTORUS_REVAB_AISLEC_CLUSTER_DESC_BASE="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC36_32x4_revAB_subtorus_aisleC"
SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING="${SUBTORUS_REVAB_AISLEC_CLUSTER_DESC_BASE}/SC20_32x4_revAB_subtorus_aisleC_mapping.yaml"
# SC4 revAB single-pod (4-rank) mock: column-1 single-pod subset of the revAB subtorus Aisle C set.
SC4_REVAB_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING="${SUBTORUS_REVAB_AISLEC_CLUSTER_DESC_BASE}/SC4_32x4_revAB_subtorus_aisleC_mapping.yaml"
# Full 20-host SC20 revC subtorus galaxy (system-110, hosts bh-glx-110-c01..c10). Used for the 80-stage Blitz
# decode ring, which needs the subtorus wrap-around to close (the revAB subtorus mock above cannot).
SC20_REVC_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC20_32x4_revC_subtorus_aisleC_mapping.yaml"
# Full 36-host subtorus SC36 galaxy (revC, Aisle D, hosts bh-glx-120-d01..d10). 36 hosts / 144 mesh
# slots -- the largest all-hosts mock; used by bh-ring-stress to exercise the mapper at scale.
SC36_REVC_SUBTORUS_AISLED_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC36_32x4_revC_subtorus_aisleD/SC36_32x4_revC_subtorus_aisleD_mapping.yaml"
# (The non-subtorus flat SC20 revAB Aisle C mock was removed: real revAB systems are subtorus, and the
# flat mock only exposes 12 physical meshes, so the SC20 rings can't map onto it. Use the revAB subtorus
# mock (SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING) instead.)
# SC16 revC subtorus, Aisle C (16-host / 64-mesh subset of the SC20 revC subtorus Aisle C set).
SC16_REVC_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC16_32x4_revC_subtorus_aisleC_mapping.yaml"
SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml"
# SC16 revC subtorus, Aisle D; 4-rank single-pod mock (revAB-style tray layout, system-110 hosts).
SC4_REVC_SUBTORUS_AISLED_SINGLE_POD_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revC_subtorus_aisleD/SC16_32x4_revC_subtorus_aisleD_single_pod_mapping.yaml"
POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SP3_16x8_revAB_aisleC/SP3_16x8_revAB_aisleC_1pod_mapping.yaml"
SC16_REVC_SUBTORUS_AISLED_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revC_subtorus_aisleD/SC16_32x4_revC_subtorus_aisleD_mapping.yaml"
DUAL_BH_GALAXY_EXPERIMENTAL_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/blackhole/dual_bh_galaxy_experimental/dual_bh_galaxy_experimental_cluster_desc_mapping.yaml"
DUAL_4X8_Z_FALLBACK_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revAB_aisleD/SC16_32x4_revAB_aisleD_dual_4x8_z_fallback_mapping.yaml"
MOCK_GALAXY_QUAD_2X4_FOUR_RANK_CLUSTER_DESC_MAPPING="tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_6u_cluster_desc/mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml"
BH_GALAXY_SP4_RANK_BINDINGS="tests/tt_metal/distributed/config/bh_galaxy_sp4_rank_bindings.yaml"
BH_GALAXY_XYZ_CLUSTER_DESC="tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml"
# Single-host 32-ASIC single-galaxy mocks: revAB (aisle D, non-subtorus), revC (aisle C, non-subtorus), revC subtorus (aisle C / aisle D).
SC16_REVAB_AISLED_SINGLE_GALAXY_CLUSTER_DESC="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revAB_aisleD/SC16_32x4_revAB_aisleD_cluster_desc/SC16_32x4_revAB_aisleD_cluster_desc_bh-glx-d07u08_rank_9.yaml"
SC16_REVC_AISLEC_SINGLE_GALAXY_CLUSTER_DESC="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revC_aisleC/SC16_32x4_revC_aisleC_cluster_desc/SC16_32x4_revC_aisleC_cluster_desc_bh-glx-110-c06u08_rank_50.yaml"
SC20_REVC_SUBTORUS_AISLEC_SINGLE_GALAXY_CLUSTER_DESC="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC20_32x4_revC_subtorus_aisleC_cluster_desc/SC20_32x4_revC_subtorus_aisleC_cluster_desc_bh-glx-110-c07u08.yaml"
SC16_REVC_SUBTORUS_AISLED_SINGLE_GALAXY_CLUSTER_DESC="tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revC_subtorus_aisleD/SC16_32x4_revC_subtorus_aisleD_cluster_desc/SC16_32x4_revC_subtorus_aisleD_cluster_desc_bh-glx-110-d07u08_rank_13.yaml"

# Custom mesh graph descriptor directories (MGD filenames appear inline in commands).
MGD_CUSTOM="tests/tt_metal/tt_fabric/custom_mesh_descriptors"
MGD_SUBTORUS="${MGD_CUSTOM}/subtorus"

# Blitz decode pipeline ring MGDs by stage count (mesh-ring length), reusing the existing in-repo MGDs.
MGD_BLITZ_16="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_single_pod_mesh_graph_descriptor.textproto"
MGD_BLITZ_48="tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto"
MGD_BLITZ_64="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto"
MGD_BLITZ_80="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_supercluster_20.textproto"
# Non-pod-aligned ring lengths (20/24/28/32/36 stages) for the long-running bh-ring-stress group.
# Each is N x M0(4x2) meshes wired in a closed ring; used to stress the mapper's general-SAT fallback.
MGD_BLITZ_20="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_ring_20stage_mesh_graph_descriptor.textproto"
MGD_BLITZ_24="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_ring_24stage_mesh_graph_descriptor.textproto"
MGD_BLITZ_28="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_ring_28stage_mesh_graph_descriptor.textproto"
MGD_BLITZ_32="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_ring_32stage_mesh_graph_descriptor.textproto"
MGD_BLITZ_36="models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_ring_36stage_mesh_graph_descriptor.textproto"

GTEST_GALAXY_LAYOUT_CHECK="ControlPlaneFixture.TestGalaxyLayoutCheck"
GTEST_GALAXY_4X4_SPLIT_HOST_LAYOUT_CHECK="ControlPlaneFixture.TestGalaxy4x4SplitHostLayoutCheck"
GTEST_GALAXY_CORNER_PINS="ControlPlaneFixture.TestGalaxyCornerPins"
GTEST_PIPELINE_BUILDER_CHECK="ControlPlaneFixture.TestPipelineBuilderCheck"
GTEST_SUBTORUS_2X4_PIPELINE="${GTEST_GALAXY_LAYOUT_CHECK}:ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
GTEST_SUBTORUS_8X4_PIPELINE="${GTEST_GALAXY_LAYOUT_CHECK}:ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
GTEST_SUBTORUS_4X4_PIPELINE="${GTEST_GALAXY_4X4_SPLIT_HOST_LAYOUT_CHECK}:ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
GTEST_SINGLE_GALAXY_SLICE="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
GTEST_SINGLE_GALAXY_BLITZ="${GTEST_GALAXY_LAYOUT_CHECK}:ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
# Llama 8b pod MGDs (40 host ranks): layout + corner pins + pod CP init; omit TestPipelineBuilderCheck
# (40-stage resolve_graph_layout ring does not finish in reasonable time on these MGDs).
GTEST_LLama_8B_POD_LAYOUT="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}"
GTEST_LLama_8B_1X2_POD="MultiHost.TestLlama8b1x2PodControlPlaneInit:${GTEST_LLama_8B_POD_LAYOUT}"
GTEST_LLama_8B_2X1_POD="MultiHost.TestLlama8b2x1PodControlPlaneInit:${GTEST_LLama_8B_POD_LAYOUT}"

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
  if [[ -n "$GREP_EXCLUDE" && "$cmd_str" == *"$GREP_EXCLUDE"* ]]; then
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
  echo "=c======================================" >&2
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
    --grep-exclude)
      if [[ $# -lt 2 ]]; then
        echo "--grep-exclude requires a value" >&2; exit 1
      fi
      GREP_EXCLUDE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

CURRENT_GROUP="$GROUP"

VALID_GROUPS="all unit phys-grouping control-plane t3k wh-galaxy bh-6u bh-single-galaxy bh-dual-galaxy bh-subtorus bh-subtorus-sc16 bh-subtorus-sc20 bh-sp4-glx bh-blitz-decode bh-pod-pipeline bh-ring-stress bh-misc"
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
    bh-subtorus bh-subtorus-sc16 bh-subtorus-sc20 bh-sp4-glx bh-blitz-decode bh-pod-pipeline bh-ring-stress bh-misc
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
run_test tt-run --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --rank-binding "${BH_GALAXY_SP4_RANK_BINDINGS}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="PhysicalGroupingDescriptorSP4Tests*"
run_test tt-run --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/dual_t3k_ci/dual_t3k_ci_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="PhysicalGroupingDescriptorDualT3kTests*"

# build_physical_multi_mesh_adjacency_graph with SP4 GLX mock (16 ranks; tt-run)
run_test tt-run --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --rank-binding "${BH_GALAXY_SP4_RANK_BINDINGS}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.BuildPhysicalMultiMeshGraph_WithPGDAndPSD_Sp4Glx*"

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
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
run_test env TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins

fi # control-plane

######################################
# T3K Tests
######################################
if run_group "t3k"; then

# Dual T3K Multi-host
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/dual_t3k_ci/dual_t3k_ci_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual2x4ControlPlaneInit"
# t3k_dual_host is a physically distinct dual-T3K cluster from dual_t3k_ci (same logical
# 2x4 mapping, different asic_ids/hostnames), so it has its own test + golden.
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_dual_host/t3k_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual2x4T3kDualHostControlPlaneInit"
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
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric2DSanity"

# 6U Split Galaxy tests (8x2 and 4x4)
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_8x2_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test6uSplit8x2ControlPlaneInit:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
run_test tt-run --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x4_mesh_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test6uSplit4x4ControlPlaneInit:ControlPlaneFixture.TestGalaxyLayoutCheck"

# Quad Galaxy Multi-host
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_quad_host/6u_quad_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_quad_host/6u_quad_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_quad_host/6u_quad_host_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestQuadGalaxyFabric2DSanity"

fi # wh-galaxy

######################################
# BH Galaxy: 6U legacy (8x4 single-host shape)
######################################
if run_group "bh-6u"; then

run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/bh_galaxy_8x4_2x2_hosts_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_6u_cluster_desc/bh_6u_cluster_desc.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins

fi # bh-6u

######################################
# BH Galaxy: single galaxy (32 ASICs)
# Per-host-sliced MGDs 1x1/1x2/2x2/4x2; dual-pod intermesh; 4-stage Blitz ring (subtorus only).
# Cluster mocks: revAB (aisle D, non-subtorus), revC (aisle C, non-subtorus), revC subtorus (aisle C / aisle D).
######################################
if run_group "bh-single-galaxy"; then

for mock in \
    "${SC16_REVAB_AISLED_SINGLE_GALAXY_CLUSTER_DESC}" \
    "${SC16_REVC_AISLEC_SINGLE_GALAXY_CLUSTER_DESC}" \
    "${SC16_REVC_SUBTORUS_AISLED_SINGLE_GALAXY_CLUSTER_DESC}" \
    "${SC20_REVC_SUBTORUS_AISLEC_SINGLE_GALAXY_CLUSTER_DESC}"; do
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_1x1_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SINGLE_GALAXY_SLICE}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SINGLE_GALAXY_SLICE}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SINGLE_GALAXY_SLICE}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SINGLE_GALAXY_SLICE}"

  # TODO: https://github.com/tenstorrent/tt-metal/issues/47718 Currently 4 stage loopback is not supported for non-subtorus galaxies
  if [[ "${mock}" == *subtorus* ]]; then
    run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_single_galaxy_4x2_line_4stage_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SINGLE_GALAXY_BLITZ}"
  fi
done

# TODO: This test is currently disabled because otpimized grouping placements is still not implemented for this case to work
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_single_galaxy_4x2_line_4stage_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVAB_AISLED_SINGLE_GALAXY_CLUSTER_DESC}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"

fi # bh-single-galaxy

######################################
# BH Galaxy: dual galaxy (64 ASICs)
# 1_pod 16x8 torus mock + SP4 GLX dual_2x2 slice; torus XY; experimental; Z fallback.
######################################
if run_group "bh-dual-galaxy"; then

run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/dual_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/blackhole/dual_glx_2.5d_torus/dual_glx_2.5d_torus_cluster_desc_mapping.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*TestBHGalaxyTorusXYControlPlaneQueries*:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
run_test tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${DUAL_BH_GALAXY_EXPERIMENTAL_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.BHDualGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
run_test tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${DUAL_BH_GALAXY_EXPERIMENTAL_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.BHDualGalaxyFabric2DSanity"
run_test tt-run --mock-cluster-rank-binding "${DUAL_4X8_Z_FALLBACK_CLUSTER_DESC_MAPPING}" --rank-binding tests/tt_metal/distributed/config/dual_4x8_z_fallback_rank_bindings.yaml --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDual4x8ZDirectionFallbackControlPlaneInit"

fi # bh-dual-galaxy

######################################
# BH Galaxy: subtorus Rev C quad mock (128 ASICs)
# Per-host layout checks, subtorus MGDs, Blitz pipelines, dual_4x16 intermesh.
######################################
if run_group "bh-subtorus"; then

# Per-host layout checks (8 ASICs: 2x4 RING+RING / 4x2 LINE+RING)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_2x4_ring_ring_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_PIPELINE_BUILDER_CHECK}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x2_line_ring_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_PIPELINE_BUILDER_CHECK}"
# Subtorus single-galaxy grouping MGDs (16–32 ASICs)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x8_ring_ring_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_ring_ring_1x1_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_4X4_SPLIT_HOST_LAYOUT_CHECK}:${GTEST_PIPELINE_BUILDER_CHECK}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_ring_ring_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_4X4_SPLIT_HOST_LAYOUT_CHECK}:${GTEST_PIPELINE_BUILDER_CHECK}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_ring_ring_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_4X4_SPLIT_HOST_LAYOUT_CHECK}:${GTEST_PIPELINE_BUILDER_CHECK}"
# Subtorus Blitz decode pipeline MGDs (16-stage 4x2 RING+LINE)
# Original single-pod Blitz decode MGDs on subtorus mock (16-stage 2x4 pipelines, CPU-only test descriptors)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_single_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_2X4_PIPELINE}"
# Quad-galaxy 4x4 split-host / 8x4 full-galaxy torus pipelines (8- and 4-stage) on subtorus mock
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_pipeline_8stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_4X4_PIPELINE}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_8x4_pipeline_4stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_8X4_PIPELINE}"

# Quad-galaxy mixed 4x8+4x4+4x2 10-stage ring (128 ASICs) on subtorus mock
# Currently disabled because complex heterogeneous multi-stage mesh graphs are not supported yet.
# TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_quad_galaxy_4x8_4x4_4x2_10stage_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
# Quad-galaxy heterogeneous 4x8+4x2 10-stage ring (128 ASICs): 2x 4x8 RING+RING + 8x 4x2 RING+LINE on subtorus mock.
# Homogeneous 4x2 hops use NESW (no assign_z_direction); heterogeneous 4x8<->4x2 hops use assign_z_direction.
# Runs the pipeline-builder and layout checks. Corner-pin checks are not run because the corner-fold
# invariant (mesh endpoints must map to asic_location 1 / trays 1-4) does not hold for the 4x2 mesh
# endpoints.
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x8_2x4_10stage_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:ControlPlaneFixture.TestGalaxyLayoutCheck"
# Dual 4x16 quad-galaxy intermesh (128 ASICs): M0 1x8 hosts + M1 2x16 hosts, 4 intermesh links
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_4x16_blitz_test.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
# Quad BH galaxy subtorus (128 ASICs, 32x4 RING+RING on quad subtorus mock)
# TODO(https://github.com/tenstorrent/tt-metal/issues/49275): TestPipelineBuilderCheck is omitted here: the 32x4 RING pipeline is a 4-stage ring
# over the 4 host slices (8x4 each), so it needs the torus wrap (row 31 <-> row 0, i.e. submesh 3 <-> submesh 0) to close.
# On this single-pod mock that wrap has no direct ethernet link (discover_connections sees a line
# 0-1-2-3, not a ring; get_chip_neighbors returns 0 neighbors for the 3<->0 pair even though routing
# finds a multi-hop path), so resolve_graph_layout cannot place the loopback edge. Not a routing-plane
# limit (the wrap direction has active planes) — the physical wrap link is simply absent on 1 pod.
# Layout + corner-pin checks still run and pass.
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_32x4_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins

fi # bh-subtorus

######################################
# BH Galaxy: subtorus SC16 superpod (subtorus_sc16, 16 MPI ranks, 512 ASICs)
# Full 2x4 / 4x4 / 8x4 pipeline rings and 32x4 quad torus on the full subtorus mock.
######################################
if run_group "bh-subtorus-sc16"; then

# 2x4 = 64-stage ring (8 ASICs/stage, 4x2 RING+LINE), 4x4 = 32-stage ring (16 ASICs/stage), 8x4 = 16-stage ring (32 ASICs/stage)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_superpod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVC_SUBTORUS_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_2X4_PIPELINE}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_4x4_pipeline_32stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVC_SUBTORUS_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_4X4_PIPELINE}"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_8x4_pipeline_16stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVC_SUBTORUS_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_8X4_PIPELINE}"
# Full 32x4 quad torus (16 MPI ranks, 8x2 host grid)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_32x4_8x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVC_SUBTORUS_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins:${GTEST_PIPELINE_BUILDER_CHECK}

fi # bh-subtorus-sc16

######################################
# BH Galaxy: SC20 (20-host) subtorus — same ring tests as bh-subtorus-sc16, scaled to 20 hosts.
# Run on both 20-host SUBTORUS mocks (revC subtorus and revAB subtorus); the non-subtorus (flat) revAB
# SC20 mock is not used for these rings (it lacks the torus wrap — see the loop comment below).
######################################
if run_group "bh-subtorus-sc20"; then

# 2x4 = 80-stage ring (8 ASICs/stage), 4x4 = 40-stage ring (16 ASICs/stage), 8x4 = 20-stage ring
# (32 ASICs/stage), plus the full 32x4 5-group torus ring.
#
# Run on both 20-host SUBTORUS mocks: revC subtorus and revAB subtorus. Both provide the torus
# wrap-around links, so the physical grouping packs the full 40 meshes and the rings map (verified: each
# yields "found 40 PSD placement(s)"). The non-subtorus (flat) revAB SC20 mock is intentionally NOT used
# for the rings — it exposes only 12 physical meshes, so e.g. the 40-stage ring (40 logical meshes) fails
# inter-mesh mapping ("target graph is larger with 40 nodes, but global graph only has 12 nodes").
for mock in "${SC20_REVC_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}" "${SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}"; do
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${MGD_BLITZ_80}" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_2X4_PIPELINE}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_sc20_4x4_pipeline_40stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_4X4_PIPELINE}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_sc20_8x4_pipeline_20stage_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_8X4_PIPELINE}"
  # Full SC20 torus: five 32x4 groups wired as a ring (20 hosts).
  # TODO(https://github.com/tenstorrent/tt-metal/issues/49275): TestPipelineBuilderCheck omitted: the 5-group 32x4 ring fails resolve_graph_layout
  # ("no valid submesh assignment found") on the subtorus mocks — a distinct pipeline-builder bug worth
  # investigating. Layout + corner-pin checks still run and pass.
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${MGD_SUBTORUS}/subtorus_sc20_32x4_5group_ring_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
done

fi # bh-subtorus-sc20

######################################
# BH Galaxy: SP4 GLX quad mock (128 ASICs, 32x4)
# Per-host slices, dual_4x16 intermesh, Blitz, 32x4 torus CP/fabric, triple-pod.
######################################
if run_group "bh-sp4-glx"; then

# Per-host-sliced BH galaxy MGDs (pod MGDs in bh-pod-pipeline group)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
# Dual 4x16 quad-galaxy intermesh (M0 1x8 + M1 2x16 hosts, 4 intermesh links)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_single_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/fabric_cpu_only_blitz_superpod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
# Llama 8b pod MGDs on the FULL 16-host sp4 system — COMMENTED OUT: the 40-host 2-mesh pod has no valid
# mapping onto a 16-host mock. The single-pod (4-host) versions run in the bh-pod-pipeline group.
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_1x2_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_2x1_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test32x4QuadGalaxyFabric2DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineControlPlaneInit"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestBHBlitzPipelineFabric2DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyControlPlaneInit:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyFabric1DSanity"
run_test tt-run --mesh-graph-descriptor tt_metal/fabric/mesh_graph_descriptors/triple_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto --mock-cluster-rank-binding "${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestTriplePod32x4QuadBHGalaxyFabric2DSanity"

######################################
# Blitz superpod mapping determinism tests (mock cluster / CPU sim — canonical + 5 variations)
# Variations: shuffled mock rank binding, relabeled MGD descriptors, permuted mesh_id in instances/connections.
######################################
AUTOMAPPER_DEFAULT_ARGS=(
  --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml
  --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/fabric_cpu_only_blitz_superpod_mesh_graph_descriptor.textproto
  --num-variations 5
  --seed 42
  --golden tests/tt_metal/tt_fabric/golden_mapping_files/TestBlitzSuperpodAutoMapperControlPlaneInit.yaml
)
# Optional override: AUTOMAPPER_TEST_ARGS="--num-variations 1 --force-regenerate"
if [[ -n "${AUTOMAPPER_TEST_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  AUTOMAPPER_ARGS=(${AUTOMAPPER_TEST_ARGS})
else
  AUTOMAPPER_ARGS=("${AUTOMAPPER_DEFAULT_ARGS[@]}")
fi
TT_METAL_SLOW_DISPATCH_MODE=1 python_env/bin/python3 tests/scripts/multihost/run_blitz_superpod_automapper_tests.py "${AUTOMAPPER_ARGS[@]}"


fi # bh-sp4-glx

######################################
# BH Galaxy: Blitz decode pipeline coverage (SC16/SC20 x ring-stage counts) + revAB subtorus corner pinnings
######################################
if run_group "bh-blitz-decode"; then

# Per-host-sliced BH galaxy MGDs (pod MGDs in bh-pod-pipeline group)
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins
# Dual 4x16 quad-galaxy intermesh (M0 1x8 + M1 2x16 hosts, 4 intermesh links)
# NOTE: Not yet working for full cluster, this is working for if you specify a single pod, because of placemnet optimizations
#TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_4x16_blitz_test.textproto" --mock-cluster-rank-binding "${SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder:ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestGalaxyCornerPins"
# ---------------------------------------------------------------------------
# Blitz decode pipeline coverage matrix (ControlPlaneFixture.TestBlitzDecodePipelineBuilder).
# Each physical cluster mock x every ring-stage count that fits its mesh-slot budget:
#   16-host SC16 clusters = 64 mesh slots -> up to 16/48/64 stage rings (80 skipped: 80 > 64).
#   20-host SC20 clusters = 80 mesh slots -> up to 16/48/64/80 stage rings.
# (SC4 / 16-slot is covered via the single-pod groups, not here.) Stage MGDs reuse existing in-repo
# blitz ring descriptors (MGD_BLITZ_*). A ring shorter than the cluster's slot count is a
# ring-embedded-into-a-larger-graph mapping -- the host-minimization SAT case guarded by the conflict
# cap; on non-subtorus clusters the ring's closing hop has no direct link and is routed the long way.
# Per-cluster stage exceptions below: SC20 revAB subtorus aisleC keeps 16 + 64 only; SC16 revC subtorus
# aisleC keeps 16 only -- its 48- and 64-stage rings strand (the closing hop lands on a Z-link that the
# host-minimization SAT pass intermittently fails to assign -- "No inter-mesh connection mesh 62->63").
# TODO(#49629): SC16 revC subtorus aisleD 64-stage (superpod MGD) fails the same z-link way after the
# tt-cluster-descriptors uplift -- the ring's closing hop lands on a Z-link the mapper doesn't assign.
# Repro: tt-run --mesh-graph-descriptor .../blitz_decode_mesh_graph_descriptor_superpod.textproto
#   --mock-cluster-rank-binding .../SC16_32x4_revC_subtorus_aisleD/SC16_32x4_revC_subtorus_aisleD_mapping.yaml
#   --gtest_filter="ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
# Kept at 16+48 here until #49629 lands; re-add 64 once the Z-link assignment is fixed.
for entry in \
    "SC16_revAB_aisleD:${SC16_REVAB_AISLED_CLUSTER_DESC_MAPPING}:16 48 64" \
    "SC20_revAB_subtorus_aisleC:${SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}:16 64" \
    "SC16_revC_subtorus_aisleC:${SC16_REVC_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}:16" \
    "SC16_revC_subtorus_aisleD:${SC16_REVC_SUBTORUS_AISLED_CLUSTER_DESC_MAPPING}:16 48" \
    "SC20_revC_subtorus_aisleC:${SC20_REVC_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}:16 48 64 80" ; do
  rest="${entry#*:}"; cluster_map="${rest%%:*}"; stages="${rest#*:}"
  for stage in ${stages}; do
    mgd_var="MGD_BLITZ_${stage}"
    run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 tt-run --mesh-graph-descriptor "${!mgd_var}" --mock-cluster-rank-binding "${cluster_map}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_SUBTORUS_2X4_PIPELINE}"
  done
done
# Llama 8b pod MGDs on the FULL 20-host revAB subtorus system — COMMENTED OUT: the 40-host 2-mesh pod has no valid
# mapping onto a 16-host mock. The single-pod (4-host) revAB subtorus versions run in the bh-pod-pipeline group
# (via SC4_REVAB_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING).
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_1x2_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
#run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_2x1_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${SC20_REVAB_SUBTORUS_AISLEC_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"

fi # bh-blitz-decode

######################################
# BH Galaxy: ring-mapping stress test (LONG RUNNING -- own group)
# Non-pod-aligned Blitz-decode ring lengths (20/24/28/32/36 stages) mapped onto the full 36-host SC36 revC
# subtorus aisleD mock. These lengths don't align to pod (4-host) / galaxy boundaries, so they exercise
# the topology mapper's general-SAT host-minimization fallback -- erratic cost that scales with ring
# length (36-stage ~42s local vs sub-second for 20/24/28/32). The subtorus wrap-around lets every length
# close and map. Own shard; ~1.5 min end-to-end locally, ~6.5 min on the ~4x-slower cpu_medium CI runner.
######################################
if run_group "bh-ring-stress"; then

# Per-op mapper watchdog: 300s -- above the worst per-stage solve on CI (~3 min on the slower cpu_medium
# runner) and below the shard step timeout (10 min, see tests/pipeline_reorg/fabric_cpu_only_unit_tests.yaml)
# so a stuck solve is caught/reported here rather than cancelled mid-shard by GitHub Actions.
RING_STRESS_TIMEOUT=300
for entry in \
    "SC36_revC_subtorus_aisleD:${SC36_REVC_SUBTORUS_AISLED_CLUSTER_DESC_MAPPING}:20 24 28 32 36" ; do
  rest="${entry#*:}"; cluster_map="${rest%%:*}"; stages="${rest#*:}"
  for stage in ${stages}; do
    mgd_var="MGD_BLITZ_${stage}"
    run_test env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=${RING_STRESS_TIMEOUT} tt-run --mesh-graph-descriptor "${!mgd_var}" --mock-cluster-rank-binding "${cluster_map}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
  done
done

fi # bh-ring-stress

######################################
# BH Galaxy: pod pipeline MGDs (TestGalaxyLayoutCheck + TestGalaxyCornerPins)
# Per-host-sliced pod MGDs on SP4 single-pod, revAB subtorus single-pod, and quad subtorus mocks (4 ranks).
# 8x16 quad pod MGDs (16x8 device, 128 ASIC) on 1-pod 16x8 SP4 mock (4 ranks).
######################################
if run_group "bh-pod-pipeline"; then

# Per-host-sliced pod MGDs (32–128 ASICs with host slices > 1 rank per slice)
for mock in "${SC4_REVAB_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" "${SC4_REVAB_AISLED_SINGLE_POD_CLUSTER_DESC_MAPPING}" "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" "${SC4_REVC_SUBTORUS_AISLED_SINGLE_POD_CLUSTER_DESC_MAPPING}"; do
  # Single-galaxy pod MGDs
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_1x1_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/single_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
  # Dual-galaxy pod MGDs (64 ASICs; dual_bh_galaxy_1x2 runs on 16x8 mock below)
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
  # Quad-galaxy pod MGD — 32x4 device (4x32_Mesh PGD), per-host sliced on 4-rank mocks
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/quad_bh_galaxy_4x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
done

# 8x16 quad pod MGDs (16x8 RING+RING device, 128 ASIC, 8x16_Mesh PGD)
# dual_bh_galaxy_1x2 (8x8 device, 64 ASIC, 8x8_Mesh PGD) — needs 16x8 pod mock, not single-galaxy single-pod mocks
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/dual_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"
# TestPipelineBuilderCheck is dropped from quad_bh_galaxy_1x2 (see https://github.com/tenstorrent/tt-metal/issues/49275):
# this is a 16x8 RING+RING torus sliced into 64 host-rank submeshes, so resolve_graph_layout must place a
# 64-stage pipeline ring. assign_submeshes (pipeline_builder.cpp) is a naive DFS backtracker whose loopback
# (ring-closure s63->s0) constraint is only checked at the leaf, so it degenerates into a Hamiltonian-cycle
# search that does not terminate on the dense torus (confirmed via gdb: 30+ nested solve() frames, hangs
# >4 min locally / never on CI). TestGalaxyLayoutCheck + TestGalaxyCornerPins still run (they pass fast).
run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/quad_bh_galaxy_1x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}"
# DISABLED on CI. quad_bh_galaxy_2x2 on the 16x8 pod mock (8x16 / 128-node mesh)
# passes locally (<4 min) but wedges in tt-run setup (control-plane init / rank-binding for the 128-node
# mesh) for >20 min on the cpu_medium runner — it never reaches a gtest and blocks the whole shard
# (0 progress at 30/50/35-min timeouts). Removing only the pipeline-builder check did not help since the
# slowness is pre-gtest. Re-enable once large-mesh control-plane init / SAT solve is fast enough for CI.
# run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/quad_bh_galaxy_2x2_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${POD_16X8_BH_GALAXY_CLUSTER_DESC_MAPPING}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_GALAXY_LAYOUT_CHECK}:${GTEST_GALAXY_CORNER_PINS}:${GTEST_PIPELINE_BUILDER_CHECK}"

# Llama 8b pod MGDs: 40-host, 2-mesh (M0 8 hosts + M1 32 hosts) decode pods.
# These only map onto a SINGLE pod (4-host mocks); on a full 16-host system mock the 40-host pod has no
# valid mapping (the 16-host variants are commented out in the bh-sp4-glx and bh-blitz-decode groups).
# Validated on rev A/B (sp4 single-pod), revAB subtorus/120c single-pod, revC subtorus aisle C (SC4),
# and revC subtorus aisle D (SC16 subtorus single-pod).
for mock in "${SC4_REVAB_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" "${SC4_REVAB_AISLED_SINGLE_POD_CLUSTER_DESC_MAPPING}" "${SC4_REVC_SUBTORUS_AISLEC_SINGLE_POD_CLUSTER_DESC_MAPPING}" "${SC4_REVC_SUBTORUS_AISLED_SINGLE_POD_CLUSTER_DESC_MAPPING}"; do
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_1x2_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_LLama_8B_1X2_POD}"
  run_test env TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --mesh-graph-descriptor "${MGD_CUSTOM}/llama_8b_2x1_pod_mesh_graph_descriptor.textproto" --mock-cluster-rank-binding "${mock}" --mpi-args "--allow-run-as-root --oversubscribe" "${TT_RUN_FLAGS[@]}" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="${GTEST_LLama_8B_2X1_POD}"
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

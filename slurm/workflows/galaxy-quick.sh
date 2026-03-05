#!/usr/bin/env bash
#SBATCH --job-name=galaxy-quick
#SBATCH --partition=wh-galaxy
#SBATCH --time=01:00:00

# Galaxy quick tests — array job with inline matrix (health + quick).
# Equivalent to .github/workflows/galaxy-quick-impl.yaml (WH sections only).
# Runs cluster validation, fabric smoke tests, and CCL smoke tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image dev

export BUILD_ARTIFACT=1
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

# ---------------------------------------------------------------------------
# Inline matrix
# ---------------------------------------------------------------------------
if [[ -z "${MATRIX_FILE:-}" ]]; then
    read -r -d '' HEALTH_CMD <<'EOCMD' || true
./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path tt_metal/fabric/cabling_descriptors/wh_galaxy_xy_torus.textproto --hard-fail --send-traffic
for i in {0..31}; do TT_VISIBLE_DEVICES="$i" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter=Cluster.ReportSystemHealth; done
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardProgramFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"
TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"
TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml
./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_deadlock_stability_6U_galaxy.yaml
TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_6U_galaxy_quick.yaml
EOCMD

    read -r -d '' QUICK_CMD <<'EOCMD' || true
./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path tt_metal/fabric/cabling_descriptors/wh_galaxy_xy_torus.textproto --hard-fail --send-traffic
uv pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt
MESH_DEVICE=TG pytest models/demos/deepseek_v3/tests/test_decoder_block.py -k "model.layers.3 and (mode_decode or mode_prefill_seq_128)" --timeout 180 --durations=0
MESH_DEVICE=TG pytest models/demos/deepseek_v3/tests/test_decoder_block.py -k "model.layers.0 and (mode_decode or mode_prefill_seq_128)" --timeout 60 --durations=0
pytest "tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4[wormhole_b0-dram_in_l1_out_axis0-bfloat16-None-num_links_4-2-dense-s2-7000-8-256-32-8x4_grid-False-fabric_2d]" --timeout=300
pytest "tests/nightly/tg/ccl/test_minimal_all_gather_async.py::test_all_gather_async[wormhole_b0-mesh_device0-normal-2-1-1-fabric_linear-DRAM_memconfig-sd35_prompt_check-3links]" --timeout=300
pytest "tests/nightly/tg/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async[wormhole_b0-mesh_device0-2-1-1-fabric_ring-mem_config_input0-mem_config_rs0-batch_8-perf-4links]" --timeout=300
pytest "tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4[wormhole_b0-l1_in_dram_out-DataType.BFLOAT16-None-4-s2-7168-8-256-32-8x4_grid-False-fabric_1d_line]" --timeout=300
pytest "tests/nightly/tg/ccl/test_all_broadcast.py::test_all_broadcast_trace[wormhole_b0-mesh_device0-device_params0-3-mem_config0-deepseek_1]" --timeout=300
pytest "tests/nightly/tg/ccl/test_all_reduce.py::test_line_all_reduce_on_TG_rows_post_commit[wormhole_b0-device_params0-ReduceType.Sum-8x4_grid-8-BufferType.DRAM-DataType.BFLOAT16-4-2-per_chip_output_shape0-Layout.TILE]" --timeout=300
pytest tests/nightly/tg/ccl/test_neighbor_pad_async.py::test_neighbor_pad_async_1d -k "Wan_shape_1 and check" --timeout=300
pytest tests/nightly/tg/ccl/test_neighbor_pad_async.py::test_neighbor_pad_async_2d -k "small_5d_h0w1 and wh_4x8_1link" --timeout=300
./build/tools/scaleout/run_fabric_manager --mesh-shape 8x4 --fabric-config FABRIC_2D --initialize-fabric
pytest "tests/scale_out/test_ccl_fabric_manager.py::test_all_to_all_combine_fabric_manager_8x4[wormhole_b0-dram_in_l1_out_axis0-bfloat16-None-num_links_4-2-dense-s2-7000-8-256-32-8x4_grid-False-fabric_manager_enabled_2d]" --timeout=300
./build/tools/scaleout/run_fabric_manager --mesh-shape 8x4 --fabric-config FABRIC_2D --terminate-fabric
./build/tools/scaleout/run_fabric_manager --mesh-shape 8x4 --fabric-config FABRIC_1D --initialize-fabric
pytest "tests/scale_out/test_ccl_fabric_manager.py::test_all_gather_async_fabric_manager[wormhole_b0-mesh_device0-normal-2-2-20-fabric_manager_enabled_linear-DRAM_memconfig-sd35_prompt_check-3links]" --timeout=300
pytest "tests/scale_out/test_ccl_fabric_manager.py::test_reduce_scatter_async_fabric_manager[wormhole_b0-mesh_device0-8-2-2-fabric_manager_enabled_linear-mem_config_input0-mem_config_rs0-batch_8-perf-3links]" --timeout=300
pytest "tests/scale_out/test_ccl_fabric_manager.py::test_all_to_all_dispatch_fabric_manager_8x4[wormhole_b0-l1_in_dram_out-DataType.BFLOAT16-None-4-s2-7168-8-256-32-8x4_grid-False-fabric_manager_enabled_1d_line]" --timeout=300
./build/tools/scaleout/run_fabric_manager --mesh-shape 8x4 --fabric-config FABRIC_1D --terminate-fabric
EOCMD

    # Escape for JSON embedding (newlines -> \n, quotes already handled)
    HEALTH_ESC="$(echo "$HEALTH_CMD" | jq -Rs '.')"
    QUICK_ESC="$(echo "$QUICK_CMD" | jq -Rs '.')"

    MATRIX_JSON="[
        {\"name\": \"Galaxy WH health tests\", \"cmd\": ${HEALTH_ESC}, \"timeout\": 20},
        {\"name\": \"Galaxy WH quick tests\",  \"cmd\": ${QUICK_ESC},  \"timeout\": 20}
    ]"
    MATRIX_FILE="$(create_matrix_file "$MATRIX_JSON")"
fi

TASK_ID="$(get_array_task_id)"
TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" cmd)"
TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" name)"

log_info "Running array task ${TASK_ID}: ${TEST_NAME}"

# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="TT_METAL_ENABLE_ERISC_IRAM=1
GTEST_OUTPUT=xml:${TT_METAL_HOME}/generated/test_reports/
DEEPSEEK_V3_HF_MODEL=${MLPERF_BASE}/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
DEEPSEEK_V3_CACHE=${MLPERF_BASE}/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI
LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"
export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"

docker_run "$DOCKER_IMAGE" "
    mkdir -p \${TT_METAL_HOME}/generated/test_reports
    ${TEST_CMD}
"

log_info "Galaxy quick test '${TEST_NAME}' complete"

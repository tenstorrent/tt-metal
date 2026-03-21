#!/usr/bin/env bash
#SBATCH --job-name=galaxy-unit-tests
#SBATCH --partition=wh-galaxy
#SBATCH --time=02:00:00

# Galaxy unit tests — array job with inline matrix covering UMD, fabric,
# multiprocess, distributed-ops, GPT-OSS, and tttv2 test groups.
# Equivalent to .github/workflows/galaxy-unit-tests-impl.yaml

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
# Inline matrix (mirrors GHA generate-matrix + galaxy-umd-tests)
# Each entry's "cmd" is a shell snippet executed inside the container.
# ---------------------------------------------------------------------------
if [[ -z "${MATRIX_FILE:-}" ]]; then
    MATRIX_JSON='[
        {"name": "Galaxy UMD unit tests",       "model": "umd",             "timeout": 10, "mlperf": false,
         "cmd": "./build/test/umd/galaxy/unit_tests_glx"},
        {"name": "UMD API tests",               "model": "umd-api",         "timeout": 20, "mlperf": false,
         "cmd": "./build/test/umd/api/api_tests"},
        {"name": "Galaxy unit tests",           "model": "unit",            "timeout": 10, "mlperf": false,
         "cmd": "TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=\"CommandQueueSingleCard*Fixture.*\" && TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_device --gtest_filter=\"GalaxyFixture.*:TGFixture.*\" && ./build/test/tt_metal/unit_tests_device --gtest_filter=\"GalaxyFixture.*:TGFixture.*\" && TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_GTEST_NUM_HW_CQS=2 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=\"UnitMeshMultiCQMultiDevice*Fixture.*\" && TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter=\"*DispatchContext*\""},
        {"name": "Galaxy Fabric tests",         "model": "fabric",          "timeout": 5,  "mlperf": false,
         "cmd": "TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*TG* && TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"Fabric2D*Fixture.*\" && ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"Fabric2D*Fixture.*\" && ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"Fabric*MuxFixture.*\""},
        {"name": "Galaxy Multi-Process tests",  "model": "multiprocess",    "timeout": 15, "mlperf": false,
         "cmd": "python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py && tt-run --mpi-args \"--allow-run-as-root\" --rank-binding 4x4_multi_big_mesh_rank_binding.yaml ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_dual_big_mesh_fabric_2d_sanity.yaml && tt-run --rank-binding 4x4_multi_mesh_rank_binding.yaml --mpi-args \"--allow-run-as-root\" python3 tests/ttnn/distributed/test_multi_mesh.py && tt-run --mpi-args \"--allow-run-as-root\" --rank-binding 4x4_multi_mesh_rank_binding.yaml ./build/test/tt_metal/multi_host_socket_tests && tt-run --mpi-args \"--allow-run-as-root\" --rank-binding 4x4_multi_big_mesh_rank_binding.yaml ./build/test/tt_metal/multi_host_fabric_tests --gtest_filter=\"*Socket*\" && tt-run --rank-binding 2x4_multi_mesh_cyclic_rank_binding.yaml --mpi-args \"--allow-run-as-root\" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_wh_6u_quad_2x4_acyclic.yaml && tt-run --rank-binding 2x4_multi_mesh_cyclic_rank_binding.yaml --mpi-args \"--allow-run-as-root\" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_wh_6u_quad_2x4_cyclic.yaml && tt-run --rank-binding 4x2_multi_mesh_rank_binding.yaml --mpi-args \"--allow-run-as-root\" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_multi_mesh_sanity_common.yaml"},
        {"name": "Galaxy distributed ops tests","model": "distributed-ops", "timeout": 5,  "mlperf": false,
         "cmd": "mkdir -p generated/test_reports && pytest tests/ttnn/distributed/test_distributed_layernorm_TG.py"},
        {"name": "Galaxy GPT-OSS unit tests",   "model": "gpt-oss",        "timeout": 25, "mlperf": true,
         "cmd": "uv pip install -r models/demos/gpt_oss/requirements.txt && TT_CACHE_PATH=${MLPERF_BASE}/huggingface/tt_cache/openai--gpt-oss-120b/ HF_MODEL=${MLPERF_BASE}/tt_dnn-models/openai/gpt-oss-120b/ pytest models/demos/gpt_oss/tests/unit -k \"4x8\" --timeout 600"},
        {"name": "Galaxy tttv2 tests",          "model": "tttv2",          "timeout": 10, "mlperf": true,
         "cmd": "failed=0; pytest --durations-min=3.0 models/common/tests/modules/rmsnorm/test_rmsnorm_2d.py -m \"not slow\" --tb=short --cov=models.common.modules.rmsnorm.rmsnorm_2d --cov-report=term-missing --cov-config=models/common/tests/setup.cfg || failed=1; pytest --durations-min=3.0 models/common/tests/modules/mlp/test_mlp_2d.py -m \"not slow\" --tb=short --cov=models.common.modules.mlp.mlp_2d --cov-report=term-missing --cov-config=models/common/tests/setup.cfg || failed=1; pytest --durations-min=3.0 models/common/tests/test_auto_compose.py --tb=short --cov=models.common.auto_compose --cov-report=term-missing --cov-config=models/common/tests/setup.cfg || failed=1; exit \\$failed"}
    ]'
    MATRIX_FILE="$(create_matrix_file "$MATRIX_JSON")"
fi

TASK_ID="$(get_array_task_id)"
TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" cmd)"
TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" name)"
NEEDS_MLPERF="$(get_task_field "$MATRIX_FILE" "$TASK_ID" mlperf)"

log_info "Running array task ${TASK_ID}: ${TEST_NAME}"

# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:${TT_METAL_HOME}/generated/test_reports/
LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"

if [[ "${NEEDS_MLPERF}" == "true" ]]; then
    export DOCKER_EXTRA_ENV="${DOCKER_EXTRA_ENV}
HF_HUB_OFFLINE=1
HF_HOME=${MLPERF_BASE}/huggingface"
    export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"
fi

docker_run "$DOCKER_IMAGE" "
    mkdir -p \${TT_METAL_HOME}/generated/test_reports
    ${TEST_CMD}
"

log_info "Galaxy unit test '${TEST_NAME}' complete"

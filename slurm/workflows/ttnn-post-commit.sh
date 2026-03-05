#!/usr/bin/env bash
# Orchestrator: generates TTNN test matrix via prepare_test_matrix.py then
# submits an array job whose size matches the matrix.
# Equivalent to .github/workflows/ttnn-post-commit.yaml (load-test-matrix + ttnn jobs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"
source_config sku_map

parse_common_args "$@"

log_info "=== ttnn-post-commit orchestrator ==="
log_info "Pipeline: ${PIPELINE_ID}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENABLED_SKUS="${ENABLED_SKUS:-wh_n300_civ2}"
MERGE_GATE_CALL="${MERGE_GATE_CALL:-false}"
TIMEOUT_OVERRIDE="${TIMEOUT_OVERRIDE:-0}"    # minutes; 0 = use per-test YAML budget
TESTS_YAML="${REPO_ROOT}/tests/pipeline_reorg/ttnn-tests.yaml"
SKU_CONFIG="${REPO_ROOT}/.github/sku_config.yaml"
TIME_BUDGET="${REPO_ROOT}/.github/time_budget.yaml"

# ---------------------------------------------------------------------------
# Verify time budget (mirrors GHA verify_time_budget.py step)
# ---------------------------------------------------------------------------
if (( TIMEOUT_OVERRIDE == 0 )); then
    log_info "Verifying test timeouts against time budget..."
    python3 "${REPO_ROOT}/.github/scripts/utils/verify_time_budget.py" \
        "$TESTS_YAML" "$TIME_BUDGET" "sanity" || \
        log_warn "Time budget verification failed — continuing anyway"
fi

# ---------------------------------------------------------------------------
# Build test matrix (mirrors GHA prepare_test_matrix.py step)
# ---------------------------------------------------------------------------
log_info "Building TTNN test matrix (skus=${ENABLED_SKUS})..."
TTNN_MATRIX_JSON="$(python3 "${REPO_ROOT}/.github/scripts/utils/prepare_test_matrix.py" \
    "$TESTS_YAML" "$ENABLED_SKUS" "$SKU_CONFIG" 2>/dev/null \
    || echo '[]')"

# Filter by merge_gate flag when called from the merge gate
if [[ "$MERGE_GATE_CALL" == "true" ]]; then
    TTNN_MATRIX_JSON="$(echo "$TTNN_MATRIX_JSON" \
        | jq -c '[.[] | select((.merge_gate // false) == true)]')"
else
    TTNN_MATRIX_JSON="$(echo "$TTNN_MATRIX_JSON" \
        | jq -c '[.[] | select((.merge_gate // false) == false)]')"
fi

# Validate non-empty
MATRIX_LEN="$(echo "$TTNN_MATRIX_JSON" | jq 'length')"
if (( MATRIX_LEN == 0 )); then
    log_fatal "No tests left after filtering (merge_gate=${MERGE_GATE_CALL})"
fi

MATRIX_FILE="$(create_matrix_file "$TTNN_MATRIX_JSON")"
log_info "Generated TTNN matrix with ${MATRIX_LEN} tasks"

# ---------------------------------------------------------------------------
# Resolve partition from SKU
# ---------------------------------------------------------------------------
PRIMARY_SKU="${ENABLED_SKUS%%,*}"   # first SKU drives partition
eval "$(get_slurm_args "$PRIMARY_SKU")"
PARTITION_FLAG="${SLURM_PARTITION}"
CONSTRAINT_FLAG="${SLURM_CONSTRAINT}"

# ---------------------------------------------------------------------------
# Submit: build -> array worker
# ---------------------------------------------------------------------------
BUILD_JOB="${DEPENDENCY_JOBID:-}"
if [[ -z "$BUILD_JOB" ]]; then
    BUILD_JOB="$(submit_after "" \
        "${SCRIPT_DIR}/workflows/build-artifact.sh" \
        --partition=build --time=02:00:00)"
    log_info "Build job: ${BUILD_JOB}"
fi

TIMEOUT_FLAG="02:00:00"
if (( TIMEOUT_OVERRIDE > 0 )); then
    TIMEOUT_FLAG="$(printf '%02d:%02d:00' $((TIMEOUT_OVERRIDE / 60)) $((TIMEOUT_OVERRIDE % 60)))"
fi

TTNN_JOB="$(sbatch \
    --parsable \
    --dependency="afterok:${BUILD_JOB}" \
    --array="0-$((MATRIX_LEN - 1))" \
    ${PARTITION_FLAG} ${CONSTRAINT_FLAG} \
    --time="${TIMEOUT_FLAG}" \
    --job-name=ttnn-post-commit \
    --output=/weka/ci/logs/%x/%j/%a.log \
    --error=/weka/ci/logs/%x/%j/%a.err \
    --export="ALL,PIPELINE_ID=${PIPELINE_ID},MATRIX_FILE=${MATRIX_FILE}" \
    "${SCRIPT_DIR}/workflows/_ttnn-post-commit-worker.sh")"

TTNN_JOB="${TTNN_JOB%%_*}"

log_info "Submitted ttnn-post-commit array job ${TTNN_JOB} (${MATRIX_LEN} tasks, after build ${BUILD_JOB})"
log_info "  Partition:  ${PARTITION_FLAG}"
log_info "  Constraint: ${CONSTRAINT_FLAG:-none}"
log_info "  Timeout:    ${TIMEOUT_FLAG}"
echo "${TTNN_JOB}"

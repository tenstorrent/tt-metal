#!/usr/bin/env bash
#SBATCH --job-name=galaxy-multi-user-isolation
#SBATCH --partition=wh-galaxy
#SBATCH --time=01:00:00
#SBATCH --output=/weka/ci/logs/%x/%j.log
#SBATCH --error=/weka/ci/logs/%x/%j.err

# Galaxy multi-user isolation tests — single job running Docker Compose with
# multiple containers to validate device isolation across 3 scenarios:
#   single (32x1-chip), tp2 (16x2-chip), tray (4x8-chip).
# Equivalent to .github/workflows/galaxy-multi-user-isolation-tests.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image dev

export BUILD_ARTIFACT=1
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
declare -A SCENARIOS
SCENARIOS=(
    [single]="32 1 chip n150_mesh_graph_descriptor.textproto tests/ttnn/unit_tests/operations/eltwise/test_fill.py -k test_fill"
    [tp2]="16 2 tp2 n300_mesh_graph_descriptor.textproto tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py "
    [tray]="4 8 tray t3k_mesh_graph_descriptor.textproto tests/nightly/t3000/ccl/test_minimal_all_gather_async.py "
)

CONTAINERS_DIR=".multi-user-galaxy-docker-files"
RESULTS_DIR=".multi-user-test-results"
COMPOSE_PROJECT="multi-user"
COMPOSE_FILE="multi-user-dc.yaml"
STARTUP_TIMEOUT=300

register_cleanup "docker compose -p ${COMPOSE_PROJECT} -f ${COMPOSE_FILE} down 2>/dev/null || true"

# ---------------------------------------------------------------------------
# Run each scenario
# ---------------------------------------------------------------------------
overall_rc=0

for scenario in single tp2 tray; do
    read -r num_containers chips_per prefix mesh_descriptor test_path test_args <<< "${SCENARIOS[$scenario]}"

    log_info "=== Scenario: ${scenario} (${num_containers}x${chips_per}) ==="

    # -- Generate tray mapping (needed for tp2 and tray) --
    python3 .github/scripts/utils/generate_tray_mapping.py 2>/dev/null || true

    # -- Create Docker Compose file --
    TRAY_ARG=""
    if [[ "$scenario" == "tray" || "$scenario" == "tp2" ]]; then
        TRAY_ARG="--tray-mapping-file tray_to_pcie_device_mapping.yaml"
    fi

    python3 .github/scripts/utils/multi-user-create-files.py \
        --image "$DOCKER_IMAGE" \
        --num-containers "$num_containers" \
        --chips-per-container "$chips_per" \
        $TRAY_ARG \
        --mesh-descriptor "$mesh_descriptor" \
        --container-prefix "$prefix"

    # -- Launch containers --
    rm -rf "$CONTAINERS_DIR"
    mkdir -p "$CONTAINERS_DIR"

    docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" up -d

    epoch_start="$(date +%s)"
    while true; do
        count=$(find "$CONTAINERS_DIR" -name "${prefix}-*.txt" 2>/dev/null | wc -l)
        elapsed=$(( $(date +%s) - epoch_start ))
        if (( count >= num_containers )); then
            break
        fi
        if (( elapsed >= STARTUP_TIMEOUT )); then
            log_error "Timeout: only ${count}/${num_containers} containers started in ${STARTUP_TIMEOUT}s"
            overall_rc=1
            break
        fi
        sleep 1
    done

    log_info "All ${num_containers} containers started for scenario '${scenario}'"

    # -- Run tests in parallel across containers --
    mkdir -p "$RESULTS_DIR"
    pids=()
    for i in $(seq 0 $(( num_containers - 1 ))); do
        container="${prefix}-${i}"
        (
            docker exec "$container" bash -c "pytest -v ${test_path} ${test_args}" 2>&1 | sed "s/^/[${container}] /"
            echo "${PIPESTATUS[0]}" > "${RESULTS_DIR}/${container}.status"
        ) &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done

    # -- Check results --
    for i in $(seq 0 $(( num_containers - 1 ))); do
        container="${prefix}-${i}"
        status_file="${RESULTS_DIR}/${container}.status"
        if [[ -f "$status_file" ]]; then
            status="$(< "$status_file")"
            if [[ "$status" != "0" ]]; then
                log_error "Container ${container} failed (exit ${status})"
                overall_rc=1
            fi
        else
            log_error "Container ${container} did not produce a status file"
            overall_rc=1
        fi
    done

    # -- Tear down --
    docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" down
    rm -rf "$CONTAINERS_DIR" "$RESULTS_DIR"

    log_info "Scenario '${scenario}' complete"
done

if [[ "$overall_rc" -ne 0 ]]; then
    log_error "One or more multi-user isolation scenarios failed"
    exit "$overall_rc"
fi

log_info "Galaxy multi-user isolation tests complete"

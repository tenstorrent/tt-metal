#!/usr/bin/env bash
#SBATCH --job-name=unit-tests-infra
#SBATCH --partition=build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# Infrastructure unit tests (no hardware required).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== Infra unit tests starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

docker_run "${IMAGE}" "
cd /work
export PYTHONPATH=/work

pytest tests/infra/ \
    --timeout=300 \
    --junitxml=generated/test_reports/unit_tests_infra.xml \
    -v \
    2>&1 | tee generated/test_reports/unit_tests_infra.log

pytest tests/scripts/ \
    --timeout=300 \
    --junitxml=generated/test_reports/unit_tests_scripts.xml \
    -v \
    2>&1 | tee generated/test_reports/unit_tests_scripts.log
"

log_info "=== Infra unit tests complete ==="

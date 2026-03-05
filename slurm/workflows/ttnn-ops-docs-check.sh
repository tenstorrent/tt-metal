#!/usr/bin/env bash
#SBATCH --job-name=ttnn-ops-docs-check
#SBATCH --partition=build
#SBATCH --time=00:30:00
#SBATCH --output=/weka/ci/logs/%x/%j.log
#SBATCH --error=/weka/ci/logs/%x/%j.err

# Validates that all TTNN ops have corresponding documentation (CPU-only).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

export BUILD_ARTIFACT=1

parse_common_args "$@"
resolve_workflow_docker_image ci-build
setup_job
trap 'cleanup_job --exit-code $?' EXIT

docker_run --no-device -- "\
    python scripts/check_ops_docs.py \
        --ops-dir ttnn/ttnn/operations \
        --docs-dir docs/source/ttnn/operations \
        --report generated/test_reports/ops_docs_check.json
"

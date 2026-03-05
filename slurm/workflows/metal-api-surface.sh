#!/usr/bin/env bash
#SBATCH --job-name=metal-api-surface
#SBATCH --partition=build
#SBATCH --time=00:30:00
#SBATCH --output=/weka/ci/logs/%x/%j.log
#SBATCH --error=/weka/ci/logs/%x/%j.err

# API surface compatibility check (CPU-only, no device needed).

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
    python scripts/check_api_surface.py \
        --golden golden/api_surface.json \
        --output generated/test_reports/api_surface_diff.json
"

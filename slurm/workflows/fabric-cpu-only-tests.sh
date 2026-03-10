#!/usr/bin/env bash
#SBATCH --job-name=fabric-cpu-only-tests
#SBATCH --partition=build
#SBATCH --time=00:45:00

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
resolve_docker_image dev
setup_job
trap 'cleanup_job --exit-code $?' EXIT

TEST_CMD="pytest tests/tt_fabric/cpu_only -x --timeout=300"

docker_run "$DOCKER_IMAGE" "${TEST_CMD}"

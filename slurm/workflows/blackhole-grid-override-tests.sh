#!/usr/bin/env bash
#SBATCH --job-name=blackhole-grid-override-tests
#SBATCH --partition=bh-p150
#SBATCH --time=01:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

export ARCH_NAME=blackhole
export BUILD_ARTIFACT=1

parse_common_args "$@"
resolve_docker_image dev
setup_job
trap 'cleanup_job --exit-code $?' EXIT

TEST_CMD="pytest tests/tt_metal/blackhole/grid_override -x --timeout=600"

docker_run -- "${TEST_CMD}"

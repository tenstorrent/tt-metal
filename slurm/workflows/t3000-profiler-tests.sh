#!/usr/bin/env bash
#SBATCH --job-name=t3000-profiler-tests
#SBATCH --partition=wh-t3k
#SBATCH --time=01:00:00

# T3000 profiler tests — single job (no array).
# Equivalent to .github/workflows/t3000-profiler-tests.yaml calling
# profiler-tests-impl.yaml with test-script=run_t3000_profiler_tests.sh.
# Requires a Tracy-enabled build (DOCKER_IMAGE should point to a Tracy image).

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
# Container execution
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"

docker_run "$DOCKER_IMAGE" "
    mkdir -p \${TT_METAL_HOME}/generated/test_reports
    source tests/scripts/t3000/run_t3000_profiler_tests.sh
    run_t3000_profiler_tests
"

log_info "T3000 profiler tests complete"

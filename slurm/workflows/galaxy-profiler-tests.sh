#!/usr/bin/env bash
#SBATCH --job-name=galaxy-profiler-tests
#SBATCH --partition=wh-galaxy
#SBATCH --time=01:00:00
#SBATCH --output=/weka/ci/logs/%x/%j.log
#SBATCH --error=/weka/ci/logs/%x/%j.err

# Galaxy profiler tests — single job (no array).
# Equivalent to .github/workflows/galaxy-profiler-tests.yaml calling
# profiler-tests-impl.yaml with test-script=run_wh_6u_profiler_tests.sh.
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
export DOCKER_EXTRA_ENV="LD_LIBRARY_PATH=/work/build/lib"

docker_run "$DOCKER_IMAGE" "
    mkdir -p /work/generated/test_reports
    source tests/scripts/wh_6u/run_wh_6u_profiler_tests.sh
    run_wh_6u_profiler_tests
"

log_info "Galaxy profiler tests complete"

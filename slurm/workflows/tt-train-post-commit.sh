#!/usr/bin/env bash
#SBATCH --job-name=tt-train-post-commit
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#SBATCH --output=/weka/ci/logs/%x/%j.log
#SBATCH --error=/weka/ci/logs/%x/%j.err
#
# GHA source: .github/workflows/tt-train-post-commit.yaml
# Runs tt-train C++ tests via ctest inside the build directory.
# Uses the --output-junit flag to produce JUnit XML for test report staging.
#
# Environment overrides:
#   ARCH_NAME      - Architecture (default: wormhole_b0)
#   GTEST_FILTER   - gtest filter expression (default: *)
#   TT_TRAIN_TIMEOUT - Timeout in minutes (default: 45)

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
setup_job
trap 'cleanup_job $?' EXIT

ARCH="${ARCH_NAME:-wormhole_b0}"
GTEST_FILTER="${GTEST_FILTER:-*}"
TIMEOUT="${TT_TRAIN_TIMEOUT:-45}"

export ARCH_NAME="${ARCH}"

log_info "Running tt-train post-commit tests (arch=${ARCH}, timeout=${TIMEOUT}m)"

# ---------------------------------------------------------------------------
# Docker environment — mirrors GHA container env block
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="TT_METAL_RUNTIME_ROOT=/work
TEST_DATA_DIR=/work/data
ENABLE_CI_ONLY_TT_TRAIN_TESTS=1
ARCH_NAME=${ARCH}"

# ---------------------------------------------------------------------------
# Run ctest from build/tt-train directory
# ---------------------------------------------------------------------------
docker_run "$DOCKER_IMAGE" "
    set -euo pipefail

    cd build/tt-train
    mkdir -p generated/test_reports
    ldd tests/ttml_tests || true

    ctest -E NIGHTLY \
        --no-tests=error \
        --output-on-failure \
        --output-junit generated/test_reports/ctest_report.xml
"

log_info "tt-train post-commit tests complete (arch=${ARCH})"

#!/usr/bin/env bash
#SBATCH --job-name=smoke
#SBATCH --partition=wh-n150
#SBATCH --time=00:30:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err

# Quick smoke test: installs .deb packages and runs the product validation-smoke
# gtest binary inside a Docker container.  Equivalent to .github/workflows/smoke.yaml.

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

# ---------------------------------------------------------------------------
# Configuration (overridable via env)
# ---------------------------------------------------------------------------
PRODUCT="${PRODUCT:-tt-metalium}"            # tt-metalium | tt-nn
PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-4.0}"  # seconds
ENABLE_COVERAGE="${ENABLE_COVERAGE:-0}"

export BUILD_ARTIFACT=1
setup_job
trap 'cleanup_job $?' EXIT

# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------
docker_run "$DOCKER_IMAGE" "
    set -euo pipefail

    # -- Sanitizer options ---------------------------------------------------
    export ASAN_OPTIONS='color=always'
    export TSAN_OPTIONS='color=always'
    export UBSAN_OPTIONS='color=always:print_stacktrace=1:halt_on_error=1'
    export LSAN_OPTIONS='suppressions=/usr/share/tt-metalium/lsan.supp'

    # -- Coverage (optional) -------------------------------------------------
    if [[ '${ENABLE_COVERAGE}' == '1' ]]; then
        export LLVM_PROFILE_FILE='/work/coverage/%p-%m.profraw'
        mkdir -p /work/coverage
    fi

    # -- Install packages ----------------------------------------------------
    apt install -y \
        ./pkgs/tt-metalium_*.deb \
        ./pkgs/tt-metalium-jit_*.deb \
        ./pkgs/${PRODUCT}_*.deb \
        ./pkgs/${PRODUCT}-validation_*.deb

    # -- Run smoke tests -----------------------------------------------------
    export GTEST_COLOR=yes
    export GTEST_OUTPUT='xml:/work/test-reports/'
    export TT_METAL_WATCHER_TEST_MODE=1
    mkdir -p /work/test-reports

    /usr/bin/${PRODUCT}-validation-smoke

    # -- Coverage merge (optional) -------------------------------------------
    if [[ '${ENABLE_COVERAGE}' == '1' ]]; then
        shopt -s nullglob
        files=(/work/coverage/*.profraw)
        if [[ \${#files[@]} -gt 0 ]]; then
            llvm-profdata-17 merge -sparse /work/coverage/*.profraw \
                -o /work/coverage/coverage.profdata
        fi
    fi
"

log_info "Smoke tests complete (product=${PRODUCT})"

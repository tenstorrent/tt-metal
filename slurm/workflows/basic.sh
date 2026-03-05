#!/usr/bin/env bash
#SBATCH --job-name=basic
#SBATCH --partition=wh-n150
#SBATCH --time=01:00:00

# Basic sanity tests: installs .deb packages and runs the product validation-basic
# gtest binary inside a Docker container.  Equivalent to .github/workflows/basic.yaml.

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
PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-10}"   # seconds
ENABLE_WATCHER="${ENABLE_WATCHER:-0}"

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

    # -- Install packages ----------------------------------------------------
    apt install -y \
        ./pkgs/tt-metalium_*.deb \
        ./pkgs/tt-metalium-jit_*.deb \
        ./pkgs/${PRODUCT}_*.deb \
        ./pkgs/${PRODUCT}-validation_*.deb

    # -- Watcher (optional) --------------------------------------------------
    if [[ '${ENABLE_WATCHER}' == '1' ]]; then
        export TT_METAL_WATCHER=2
        export TT_METAL_WATCHER_APPEND=1
        export TT_METAL_WATCHER_NOINLINE=1
    fi

    # -- Run basic tests -----------------------------------------------------
    export GTEST_COLOR=yes
    export GTEST_OUTPUT=\"xml:\${TT_METAL_HOME}/test-reports/\"
    export TT_METAL_HOME=/usr/libexec/tt-metalium
    export TT_METAL_WATCHER=5
    export TT_METAL_WATCHER_TEST_MODE=1
    mkdir -p \${TT_METAL_HOME}/test-reports

    /usr/bin/${PRODUCT}-validation-basic
"

log_info "Basic tests complete (product=${PRODUCT})"

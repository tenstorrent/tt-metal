#!/usr/bin/env bash
#SBATCH --job-name=ttnn-tutorials-post-commit
#SBATCH --partition=wh-n150
#SBATCH --time=01:00:00
#
# GHA source: part of ttnn-post-commit workflow
# Runs TTNN tutorial tests on a single N150 node.

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

log_info "Running TTNN tutorial tests"

export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:generated/test_reports/"
docker_run "$DOCKER_IMAGE" "\
    pytest tests/ttnn/tutorials/ \
        -x --timeout=600 \
        --junit-xml=generated/test_reports/ttnn_tutorials.xml
"

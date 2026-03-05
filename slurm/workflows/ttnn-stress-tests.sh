#!/usr/bin/env bash
#SBATCH --job-name=ttnn-stress-tests
#SBATCH --partition=wh-n150
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/ttnn-stress-tests.yaml
# Runs TTNN stress tests split across array tasks.
# Override arch via ARCH_NAME env var (default: wormhole_b0).
# Override timeout via STRESS_TIMEOUT_SEC (default: 1200).

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SPLITS="${STRESS_NUM_SPLITS:-4}"
GROUP=$(( TASK_ID + 1 ))
TIMEOUT="${STRESS_TIMEOUT_SEC:-1200}"

log_info "Running TTNN stress tests: split ${GROUP}/${NUM_SPLITS} (timeout ${TIMEOUT}s)"

export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:generated/test_reports/"
docker_run "$DOCKER_IMAGE" "\
    pytest tests/ttnn/stress/ \
        --splits ${NUM_SPLITS} --group ${GROUP} \
        --splitting-algorithm least_duration \
        --timeout=${TIMEOUT} \
        --junit-xml=generated/test_reports/ttnn_stress_${TASK_ID}.xml
"

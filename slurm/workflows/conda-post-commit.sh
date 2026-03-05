#!/usr/bin/env bash
#SBATCH --job-name=conda-post-commit
#SBATCH --partition=wh-n300
#SBATCH --time=02:00:00
#SBATCH --array=0-13
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/conda-post-commit.yaml
# Runs ttnn unit tests in a conda environment.
# Tasks 0-11: pytest split groups; Task 12: fast runtime off; Task 13: example tests
#
# Environment overrides:
#   NUM_SPLITS       - Number of pytest splits (default: 12)
#   CONDA_TIMEOUT    - Timeout in minutes (default: 45)
#   CONDA_DOCKER_IMAGE - Docker image override (default: quay.io/condaforge/linux-anvil-x86_64:alma9)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"

# Conda tests use a specialized docker image
DOCKER_IMAGE="${CONDA_DOCKER_IMAGE:-quay.io/condaforge/linux-anvil-x86_64:alma9}"
export DOCKER_IMAGE

export BUILD_ARTIFACT=1
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SPLITS="${NUM_SPLITS:-12}"
TIMEOUT="${CONDA_TIMEOUT:-45}"

# Conda environment setup runs inside the container
CONDA_SETUP="\
    source /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n metalium_env python=3.10 tt-metalium accelerate click ipython \
        multiprocess nbconvert nbformat pandas psutil pydantic pytest pytest-split \
        pytest-timeout pyyaml seaborn torchvision transformers -c conda-forge && \
    conda activate metalium_env && \
    wget -O /tmp/sfpi-info.sh https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/sfpi-info.sh && \
    wget -O /tmp/sfpi-version https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/sfpi-version && \
    eval \$(/tmp/sfpi-info.sh SHELL rpm) && \
    wget \$sfpi_url/\$sfpi_filename && \
    dnf -y install ./\$sfpi_filename && \
    rm -f \$sfpi_filename /tmp/sfpi-info.sh /tmp/sfpi-version && \
    export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib"

if (( TASK_ID < NUM_SPLITS )); then
    GROUP=$(( TASK_ID + 1 ))
    TEST_NAME="ttnn group ${GROUP}"
    TEST_CMD="pytest tests/ttnn/unit_tests -xv --splits ${NUM_SPLITS} --group ${GROUP} -m 'not disable_fast_runtime_mode'"
elif (( TASK_ID == NUM_SPLITS )); then
    TEST_NAME="ttnn fast runtime off"
    TEST_CMD="pytest tests/ttnn/unit_tests -xv -m requires_fast_runtime_mode_off"
    export DOCKER_EXTRA_ENV='TTNN_CONFIG_OVERRIDES={"enable_fast_runtime_mode": false}'
elif (( TASK_ID == NUM_SPLITS + 1 )); then
    TEST_NAME="ttnn example tests"
    TEST_CMD="./tests/scripts/run_ttnn_examples.sh"
else
    log_fatal "Unknown task ID: ${TASK_ID} (expected 0-$((NUM_SPLITS + 1)))"
fi

log_info "Running conda test: ${TEST_NAME} (task ${TASK_ID})"

export DOCKER_EXTRA_VOLUMES="/mnt/MLPerf:/mnt/MLPerf:ro"
docker_run "$DOCKER_IMAGE" "\
    ${CONDA_SETUP} && \
    ${TEST_CMD}
"

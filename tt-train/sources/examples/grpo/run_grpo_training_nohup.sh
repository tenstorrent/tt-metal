#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_grpo_training_nohup.sh [optional-run-name]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

RUN_NAME="${1:-grpo_training}"
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_BASE_DIR="${REPO_ROOT}/generated/tt-train/grpo_training_runs"
RUN_DIR="${RUN_BASE_DIR}/${RUN_NAME}_${TIMESTAMP_UTC}"

SCRIPT_PATH="${SCRIPT_DIR}/grpo_training.py"
MODEL_CONFIG_DIR="${REPO_ROOT}/tt-train/configs/model_configs"
TRAINING_CONFIG_DIR="${REPO_ROOT}/tt-train/configs/training_configs"

LOG_FILE="${RUN_DIR}/grpo_training.log"
META_FILE="${RUN_DIR}/run_metadata.txt"
PID_FILE="${RUN_DIR}/pid.txt"

mkdir -p "${RUN_DIR}"

cp -a "${SCRIPT_PATH}" "${RUN_DIR}/"
cp -a "${MODEL_CONFIG_DIR}" "${RUN_DIR}/"
cp -a "${TRAINING_CONFIG_DIR}" "${RUN_DIR}/"

{
  echo "run_name=${RUN_NAME}"
  echo "timestamp_utc=${TIMESTAMP_UTC}"
  echo "started_at_human_utc=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "repo_root=${REPO_ROOT}"
  echo "script=${SCRIPT_PATH}"
  echo "python=$(command -v python || true)"
  echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
  echo "git_branch=$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
} > "${META_FILE}"

(
  cd "${REPO_ROOT}"
  nohup python "${SCRIPT_PATH}" > "${LOG_FILE}" 2>&1 < /dev/null &
  echo $! > "${PID_FILE}"
)

echo "Started."
echo "Run dir: ${RUN_DIR}"
echo "PID: $(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"
echo "Metadata: ${META_FILE}"
echo "Tail logs: tail -f ${LOG_FILE}"

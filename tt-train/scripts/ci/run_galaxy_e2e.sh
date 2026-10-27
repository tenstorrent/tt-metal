#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
set -euo pipefail

FILTER="${1:?usage: run_galaxy_e2e.sh <run_models filename>[,<filename>...]}"

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PIPELINE_DIR:?PIPELINE_DIR must be set to an NFS scratch dir shared with the mpirun rank}"
export HF_HOME="${HF_HOME:-/mnt/models/huggingface}"

MODEL_CONFIG="${TT_TRAIN_MODEL_CONFIG:-tt-train/scripts/run_models_configs/galaxy.yaml}"
METRICS_DIR="${PIPELINE_DIR}/tt-train-metrics"
# The rank cannot write $GITHUB_STEP_SUMMARY, so it writes here and we append it after mpirun exits.
SUMMARY="${PIPELINE_DIR}/tt_train_summary.md"

printf -v RUN_MODELS '%q ' python tt-train/scripts/run_models.py \
  --model_config "${MODEL_CONFIG}" \
  --filter-filenames "${FILTER}" \
  --output-dir "${METRICS_DIR}" \
  --summary-file "${SUMMARY}"

rc=0
mpirun -np 1 --bind-to none --tag-output --wdir "${TT_METAL_HOME}" bash -lc "
  ./tt-train/scripts/setup_pth.sh
  ${RUN_MODELS}
" || rc=$?

if [ -s "${SUMMARY}" ] && [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "${SUMMARY}" >> "${GITHUB_STEP_SUMMARY}"
fi
exit "${rc}"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Convenience wrapper: chip reset + full dots.ocr bottom-up test suite.
# Runs every group (ops/ modules/ e2e/) in one pytest invocation, with
# MESH_DEVICE=T3K and the DP-on-T3K mesh the capture uses.
#
# Usage:
#   bash models/experimental/tt_symbiote/tests/unit/dots_ocr/scripts/regression_check.sh \
#       [extra pytest args ...]
#
# Examples:
#   # Full suite, quiet:
#   bash .../scripts/regression_check.sh -q
#
#   # Single op file:
#   bash .../scripts/regression_check.sh ops/test_rms_norm.py
#
#   # Single row by id substring:
#   bash .../scripts/regression_check.sh -k cid8
set -euo pipefail

REPO_ROOT="/home/ttuser/salnahari/tt-metal"
TEST_ROOT="${REPO_ROOT}/models/experimental/tt_symbiote/tests/unit/dots_ocr"

cd "${REPO_ROOT}"

# Activate the in-tree venv if not already on PATH.
if ! command -v pytest >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/python_env/bin/activate"
fi

echo "[regression_check] tt-smi -r (chip reset)"
unset TT_VISIBLE_DEVICES
"${REPO_ROOT}/python_env/bin/tt-smi" -r

LOG="${LOG:-/tmp/dots_ocr_regression_check_$(date +%Y%m%d_%H%M%S).log}"
echo "[regression_check] log -> ${LOG}"

# If the user passed any positional file arg, treat that as the test target.
# Otherwise default to the whole suite (ops + modules + e2e).
TARGETS=()
EXTRA=()
for arg in "$@"; do
  if [[ "${arg}" == -* || "${arg}" == *"="* ]]; then
    EXTRA+=("${arg}")
  else
    # Accept both bare relative paths and ones rooted at the test dir.
    if [[ "${arg}" = /* ]]; then
      TARGETS+=("${arg}")
    elif [[ -e "${TEST_ROOT}/${arg}" ]]; then
      TARGETS+=("${TEST_ROOT}/${arg}")
    else
      TARGETS+=("${arg}")
    fi
  fi
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(
    "${TEST_ROOT}/ops/"
    "${TEST_ROOT}/modules/"
    "${TEST_ROOT}/e2e/"
  )
fi

MESH_DEVICE="${MESH_DEVICE:-T3K}" \
DOTS_OCR_PARALLELISM="${DOTS_OCR_PARALLELISM:-DP}" \
TT_SYMBIOTE_RUN_MODE="${TT_SYMBIOTE_RUN_MODE:-NORMAL}" \
pytest \
  "${TARGETS[@]}" \
  --timeout=0 --tb=short --durations=20 -q \
  "${EXTRA[@]}" \
  2>&1 | tee "${LOG}"

echo "[regression_check] done — see ${LOG}"

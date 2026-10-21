#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Trial only: the LLK smoke shard with LLK_PERF_PHASE=all|compile|run.
set -euo pipefail

GROUP="${1:?usage: run_llk_smoke_trial.sh <group> <splits>}"
SPLITS="${2:?usage: run_llk_smoke_trial.sh <group> <splits>}"
PHASE="${LLK_PERF_PHASE:-all}"
WORKERS="${LLK_PERF_COMPILE_WORKERS:-auto}"
MARKERS="${PYTEST_MARKERS:?}"

cd "$(dirname "${BASH_SOURCE[0]}")/python_tests"
COMMON=(-q --override-ini=log_cli=false -m "$MARKERS" --splits "$SPLITS" --group "$GROUP" --timeout=60)
REPORT="pytest-report-${CHIP_ARCH}-${GROUP}"

case "$PHASE" in
  all)
    pytest "${COMMON[@]}" -n auto -x --dist=worksteal --junitxml="${REPORT}.xml" .
    ;;
  compile)
    pytest "${COMMON[@]}" --compile-producer -n "$WORKERS" --junitxml="${REPORT}-compile.xml" .
    ;;
  run)
    pytest "${COMMON[@]}" --compile-consumer -n auto -x --dist=worksteal --junitxml="${REPORT}.xml" .
    ;;
esac

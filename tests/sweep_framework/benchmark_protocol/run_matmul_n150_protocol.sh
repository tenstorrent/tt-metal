#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# One-shot local driver for matmul model-traced N150 protocol (Milestone 1).
# Requires: built tt-metal Python env, ARCH_NAME, N150 (or any Wormhole) device available.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${ARCH_NAME:-}" ]]; then
  export ARCH_NAME="${IRD_ARCH_NAME:-wormhole_b0}"
  echo "ARCH_NAME not set; using ${ARCH_NAME}"
fi

SWEEP_FW="${REPO_ROOT}/tests/sweep_framework"
GEN="${SWEEP_FW}/benchmark_protocol/generated"
VECTORS="${SWEEP_FW}/vectors_export"
MODULE="model_traced.matmul_model_traced"
SUITE="model_traced"
MANIFEST="${GEN}/matmul_n150_protocol_manifest.json"
PROTO_JSON="${GEN}/matmul_n150_protocol_all.json"
REPORT_JSON="${GEN}/matmul_n150_last_report.json"

usage() {
  cat <<EOF
Usage: $0 <command>

  generate   — sweeps_parameter_generator for matmul model_traced only (needs working ttnn import)
  partition  — write manifest (smoke/train/holdout) from vectors_export
  write-json — write slice JSON files + combined protocol file for --vector-source file
  run        — sweeps_runner on combined protocol vectors (perf + optional memory)
  report     — summarize results_export using manifest
  all        — generate && partition && write-json && run && report

Environment:
  ARCH_NAME           — wormhole_b0 (default) or blackhole
  RUNNER_LABEL        — e.g. N150 (shown in results card_type)
  SWEEPS_TAG          — vector tag (default: USER)
  MEASURE_MEMORY=1    — for 'run', add --measure-memory

EOF
}

run_generate() {
  ( cd "${SWEEP_FW}" && python3 sweeps_parameter_generator.py \
    --module-name "${MODULE}" \
    --suite-name "${SUITE}" \
    --model-traced all )
}

run_partition() {
  mkdir -p "${GEN}"
  python3 "${SWEEP_FW}/benchmark_protocol/matmul_n150_protocol.py" partition \
    --vectors-export "${VECTORS}" \
    --output "${MANIFEST}"
}

run_write_json() {
  mkdir -p "${GEN}"
  python3 "${SWEEP_FW}/benchmark_protocol/matmul_n150_protocol.py" write-json \
    --manifest "${MANIFEST}" \
    --vectors-export "${VECTORS}" \
    --output-dir "${GEN}"
}

run_sweeps() {
  local extra=()
  if [[ "${MEASURE_MEMORY:-0}" == "1" ]]; then
    extra+=(--measure-memory)
  fi
  ( cd "${SWEEP_FW}" && python3 sweeps_runner.py \
    --module-name "${MODULE}" \
    --suite-name "${SUITE}" \
    --vector-source file \
    --file-path "${PROTO_JSON}" \
    --result-dest results_export \
    --perf \
    --tag "${SWEEPS_TAG:-${USER:-protocol}}" \
    --summary \
    "${extra[@]}" )
}

run_report() {
  python3 "${SWEEP_FW}/benchmark_protocol/matmul_n150_protocol.py" report \
    --manifest "${MANIFEST}" \
    --results-glob "${SWEEP_FW}/results_export/model_traced_*.json" \
    --json-out "${REPORT_JSON}"
  echo "Report JSON: ${REPORT_JSON}"
}

cmd="${1:-}"
case "${cmd}" in
  generate) run_generate ;;
  partition) run_partition ;;
  write-json) run_write_json ;;
  run) run_sweeps ;;
  report) run_report ;;
  all)
    run_generate
    run_partition
    run_write_json
    run_sweeps
    run_report
    ;;
  ""|help|-h|--help) usage ;;
  *) echo "Unknown command: ${cmd}"; usage; exit 1 ;;
esac

#!/usr/bin/env bash
# Run the short-seq embedding perf parametrizations one at a time; each run is
# appended to its own log file (test id in the filename).
#
# Usage:
#   ./run_short_seq_embedding_perf_logs.sh
#   LOG_DIR=/path/to/logs ./run_short_seq_embedding_perf_logs.sh
#
# From repo root you can also:
#   bash models/demos/wormhole/qwen3_embedding_8b/demo/run_short_seq_embedding_perf_logs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  # demo -> qwen3_embedding_8b -> wormhole -> demos -> models -> tt-metal
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
fi

DEMO_PY="${SCRIPT_DIR}/demo.py"
model_name="qwen3_embedding_4b"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs_newest_${model_name}}"
mkdir -p "${LOG_DIR}"

# batch-1 and batch-25: seqlt32, seqlt64, seqlt128, seqlt512, seq512
TEST_IDS=(
  dp1-batch1-seqlt512
  dp1-batch1-seqlt128
  dp1-batch1-seqlt32
  dp1-batch1-seqlt64
  dp1-batch1-seq512
  dp1-batch25-seqlt512
  dp1-batch25-seqlt128
  dp1-batch25-seqlt32
  dp1-batch25-seqlt64
  dp1-batch25-seq512
)

cd "${REPO_ROOT}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo ""

for tid in "${TEST_IDS[@]}"; do
  log="${LOG_DIR}/test_embedding_perf_${tid}.log"
  {
    echo "================================================================================"
    echo "run_start: $(date -Is)"
    echo "test_id: ${tid}"
    echo "command: pytest \"${DEMO_PY}::test_embedding_perf\" -k \"${tid}\" -v --tb=short"
    echo "================================================================================"
  } >>"${log}"

  # shellcheck disable=SC2094
  set +e
  pytest "${DEMO_PY}::test_embedding_perf" -k "${tid}" -v --tb=short 2>&1 | tee "${log}"
  ec=${PIPESTATUS[0]}
  set -e

  {
    echo ""
    echo "run_end: $(date -Is)  exit_code=${ec}"
    echo ""
  } >>"${log}"

  if [[ "${ec}" -ne 0 ]]; then
    echo "FAILED: ${tid} (exit ${ec}) — see ${log}" >&2
  else
    echo "OK: ${tid} — appended to ${log}"
  fi
done

echo ""
echo "All runs finished. Logs under: ${LOG_DIR}"

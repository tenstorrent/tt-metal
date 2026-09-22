#!/usr/bin/env bash
# TEST 1 -- matmul only. One kind of test on the card, so no two cores can be
# running different tests. Everything else is the production runner.
set -euo pipefail
GROUP="${1:?}"; N_GROUPS="${2:?}"
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-false}"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"
case "$SPEED_OF_LIGHT" in
  true) SPEED_OF_LIGHT_ARGS=(--speed-of-light) ;;
  false) SPEED_OF_LIGHT_ARGS=() ;;
  *) echo "SPEED_OF_LIGHT must be true or false" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data
PQ="-q --override-ini=log_cli=false"
M="perf and not accuracy"
TARGET=perf_math_matmul.py

echo "===== matmul-only, shard $GROUP/$N_GROUPS  $(date -u +%H:%M:%S)"
pytest $PQ "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 -m "$M" \
  --timeout=60 --splits "$N_GROUPS" --group "$GROUP" "$TARGET" \
  --junitxml="pytest-report-blackhole-${GROUP}-compile.xml"
echo "===== measure  $(date -u +%H:%M:%S)"
PERF_CORE_LOG=/tmp/corelog \
pytest $PQ "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer -n 15 -m "$M" \
  --timeout=60 --splits "$N_GROUPS" --group "$GROUP" "$TARGET" \
  --junitxml="pytest-report-blackhole-${GROUP}-run.xml"
D="$LLK_ROOT/perf_data/runs/corelog-$GROUP"; mkdir -p "$D"
cat /tmp/corelog.*.tsv > "$D/cores.tsv" 2>/dev/null || true
echo "  core log lines: $(wc -l < "$D/cores.tsv" 2>/dev/null || echo 0)"

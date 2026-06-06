#!/usr/bin/env bash
# Run PCC suite once, demo N times, and a same-process determinism check.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
PY="${ROOT}/python_env/bin/python"
OUT_DIR="${ROOT}/models/experimental/seamless_m4t_v2_large/scripts/multi_run_logs"
mkdir -p "$OUT_DIR"
DEMO_ITERS="${1:-4}"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT="${OUT_DIR}/report_${TS}.txt"

exec > >(tee "$REPORT") 2>&1

echo "=== Seamless M4T v2 multi-run report ${TS} ==="
echo "Repo: $ROOT"
cd "$ROOT"

echo ""
echo "=== 1. PCC test suite (blackhole-1x4) ==="
${PY} -m pytest models/experimental/seamless_m4t_v2_large/tests/pcc/ \
  -v -k 'blackhole-1x4' --tb=short -q 2>&1 | tail -40

echo ""
echo "=== 2. Demo x${DEMO_ITERS} (separate processes, cold JIT preflight each) ==="
for i in $(seq 1 "$DEMO_ITERS"); do
  LOG="${OUT_DIR}/demo_${TS}_run${i}.log"
  echo "--- demo run ${i}/${DEMO_ITERS} -> ${LOG} ---"
  ${PY} models/experimental/seamless_m4t_v2_large/demo/demo.py 2>&1 | tee "$LOG"
  echo ""
  echo "  [run ${i} TT summary extract]"
  sed -n '/TT-aligned runtime summary/,/^---/p' "$LOG" | head -20
  echo ""
done

echo ""
echo "=== 3. Same-process determinism (T2TT + T2ST x3 each, one device session) ==="
${PY} models/experimental/seamless_m4t_v2_large/scripts/check_determinism.py 2>&1

echo ""
echo "Report saved: $REPORT"

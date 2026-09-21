#!/usr/bin/env bash
# CONCURRENCY EXPERIMENT -- do the neighbours decide the number?
#
# The same 64 matmul targets, measured under different amounts of concurrent
# work, each arm run twice. Stage 2 ruled out the predecessor on one core.
set -euo pipefail
GROUP="${1:?}"
N_GROUPS="${2:?}"
if [ "$GROUP" != "1" ]; then
  echo "experiment: only group 1 runs; this group exits."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR/python_tests"
export PERF_KEEP_RUNS=0
unset PERF_RUN_TAG

M="perf and not accuracy"
PQ="-q --override-ini=log_cli=false"

pytest -q --collect-only -m "$M" perf_math_matmul.py > /tmp/mm.txt 2>&1 || true
grep '::' /tmp/mm.txt > /tmp/mm_ids.txt
TOTAL=$(wc -l < /tmp/mm_ids.txt)
echo "===== matmul items collected: $TOTAL"
[ "$TOTAL" -ge 1000 ] || { echo "FATAL: collection looks wrong" >&2; exit 1; }

STEP=$(( TOTAL / 64 ))
awk -v s="$STEP" 'NR % s == 1' /tmp/mm_ids.txt | head -64 > /tmp/target.txt
mapfile -t TARGET < /tmp/target.txt
# Filler is disjoint from the target: a different residue of the same stride.
awk -v s="$STEP" 'NR % s == 5' /tmp/mm_ids.txt > /tmp/filler_all.txt
head -600  /tmp/filler_all.txt > /tmp/f600.txt
mapfile -t F600 < /tmp/f600.txt
awk 'NR % 12 == 7' /tmp/mm_ids.txt | head -3000 > /tmp/f3000.txt
mapfile -t F3000 < /tmp/f3000.txt
echo "===== target ${#TARGET[@]}  f600 ${#F600[@]}  f3000 ${#F3000[@]}"

arm() {
  local label="$1" workers="$2" filler="$3"
  local -a extra=()
  case "$filler" in
    0)     extra=() ;;
    600)   extra=("${F600[@]}") ;;
    3000)  extra=("${F3000[@]}") ;;
  esac
  echo "===== ARM $label  workers=$workers filler=$filler  $(date -u +%H:%M:%S)"
  export PERF_RUN_TAG="$label"
  pytest $PQ --compile-producer -n 10 -m "$M" --timeout=60 \
    "${TARGET[@]}" ${extra[@]+"${extra[@]}"} > "/tmp/$label.c.log" 2>&1 \
    || echo "  (producer rc=$?)"
  pytest $PQ --compile-consumer -n "$workers" -m "$M" --timeout=60 \
    "${TARGET[@]}" ${extra[@]+"${extra[@]}"} > "/tmp/$label.r.log" 2>&1 \
    || echo "  (consumer rc=$?)"
  tail -2 "/tmp/$label.r.log" | sed 's/^/  /'
  unset PERF_RUN_TAG
}

arm n1_f600_a   1  600
arm n1_f600_b   1  600
arm n15_f0_a   15    0
arm n15_f0_b   15    0
arm n15_f600_a 15  600
arm n15_f600_b 15  600
arm n15_f3000_a 15 3000
arm n15_f3000_b 15 3000

echo "===== arms written:"
ls -1 "$LLK_ROOT/perf_data/runs/" || true
echo "===== experiment done ====="

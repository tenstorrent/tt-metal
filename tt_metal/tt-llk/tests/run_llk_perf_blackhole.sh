#!/usr/bin/env bash
# PREDECESSOR EXPERIMENT -- does what ran before change what we measure?
#
# 64 matmul tests, measured eight times on one core with one worker, each time
# after different predecessors. Scheduling cannot explain a difference here:
# there is one worker, so the order within an arm is fixed.
set -euo pipefail
GROUP="${1:?}"
N_GROUPS="${2:?}"
if [ "$GROUP" != "1" ]; then
  echo "experiment: only group 1 runs; this group exits."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
export PERF_KEEP_RUNS=0          # keep every arm; the default prunes to 10
unset PERF_RUN_TAG               # each arm sets its own

M="perf and not accuracy"
PQ="-q --override-ini=log_cli=false"

# --- the target: 64 matmul tests spread across the module ------------------
pytest -q --collect-only -m "$M" perf_math_matmul.py > /tmp/mm.txt 2>&1 || true
grep '::' /tmp/mm.txt > /tmp/mm_ids.txt
TOTAL=$(wc -l < /tmp/mm_ids.txt)
echo "===== matmul items collected: $TOTAL"
if [ "$TOTAL" -lt 1000 ]; then
  echo "FATAL: matmul collection looks wrong" >&2
  exit 1
fi
STEP=$(( TOTAL / 64 ))
awk -v s="$STEP" 'NR % s == 1' /tmp/mm_ids.txt | head -64 > /tmp/target.txt
awk -v s="$STEP" 'NR % s == 3' /tmp/mm_ids.txt | head -64 > /tmp/other.txt
mapfile -t TARGET < /tmp/target.txt
mapfile -t OTHER  < /tmp/other.txt
echo "===== target ids: ${#TARGET[@]}   other-matmul ids: ${#OTHER[@]}"
printf '  %s\n' "${TARGET[@]:0:3}"

arm() {
  local label="$1"; shift
  echo "===== ARM $label  ($# predecessor arg(s))  $(date -u +%H:%M:%S)"
  export PERF_RUN_TAG="$label"
  # Producer builds every ELF this arm needs; consumer measures on ONE worker.
  pytest $PQ --compile-producer -n 10 -m "$M" --timeout=60 "$@" "${TARGET[@]}" \
    > "/tmp/${label}.compile.log" 2>&1 || echo "  (producer rc=$?)"
  pytest $PQ --compile-consumer -n 1 -m "$M" --timeout=60 "$@" "${TARGET[@]}" \
    > "/tmp/${label}.run.log" 2>&1 || echo "  (consumer rc=$?)"
  tail -2 "/tmp/${label}.run.log" | sed 's/^/  /'
  unset PERF_RUN_TAG
}

arm solo
arm after_binary        perf_eltwise_binary.py
arm after_reduce        perf_reduce.py
arm after_tilize        perf_fast_tilize_full.py
arm after_transpose     perf_unpack_transpose.py
arm after_packdestbank  perf_pack_dest_bank.py
arm after_matmul_other  "${OTHER[@]}"
arm solo2

echo "===== arms written:"
ls -1 ../perf_data/runs/
echo "===== experiment done ====="

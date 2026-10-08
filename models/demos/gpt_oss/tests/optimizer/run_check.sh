#!/usr/bin/env bash
# One-command check of gpt-oss-20b decode on a QuietBox 2 (4 Blackhole chips, 1x4 mesh, batch 1):
# the shared perf test N times (default 5, median reported) and the accuracy test once.
# Run from the tt-metal root with the python env active:  bash models/demos/gpt_oss/tests/optimizer/run_check.sh [N]
# See models/demos/gpt_oss/MARK_TOOL_RESULT.md for the expected numbers and the reference files.
set -uo pipefail
N=${1:-5}
G=models/demos/gpt_oss/tests/optimizer
REF=generated/optimizer_reference/gpt-oss-20b-logits.pt
PIN=generated/optimizer_accuracy_baseline_gpt-oss-20b.json
[ -f "$G/test_optimizer_perf.py" ] || { echo "run from the tt-metal root"; exit 2; }
export HF_MODEL=${HF_MODEL:-openai/gpt-oss-20b} TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
LOG=${CHECK_LOG_DIR:-generated/run_check}
mkdir -p "$LOG"
echo "tree $(git rev-parse --short HEAD), $N perf runs, logs in $LOG/"

ms=()
for i in $(seq 1 "$N"); do
  PERF_GATE_ROLE=verdict OPTIMIZER_DECODE_ONLY=1 pytest "$G/test_optimizer_perf.py" --timeout 1800 > "$LOG/perf_$i.log" 2>&1
  v=$(grep -o 'TRACE_STAGE_MS\[decode\]=[0-9.]*' "$LOG/perf_$i.log" | tail -1 | cut -d= -f2)
  echo "perf $i: ${v:-FAILED (see $LOG/perf_$i.log)} ms/token"
  [ -n "$v" ] && ms+=("$v")
done
if [ ${#ms[@]} -gt 0 ]; then
  python3 -c "import statistics,sys; v=[float(x) for x in sys.argv[1:]]; m=statistics.median(v); print(f'decode median {m:.4f} ms/token = {1000/m:.1f} tokens/s/user (range {min(v):.4f}-{max(v):.4f})')" "${ms[@]}"
fi
grep -m1 '^GATE_TEXT' "$LOG/perf_1.log" 2>/dev/null | cut -c1-200

# The accuracy test compares against the pinned scores of the unmodified tree. Without the pin it would pin
# THIS tree as the baseline and pass trivially, so refuse instead.
if [ ! -f "$PIN" ] || [ ! -f "$REF" ]; then
  echo "accuracy: SKIPPED. Missing $PIN and/or $REF (unpack gptoss20b_accuracy_ref.tgz in the tt-metal root)."
  exit 1
fi
pytest "$G/test_optimizer_pcc.py" --timeout 1800 > "$LOG/accuracy.log" 2>&1
rc=$?
python3 - "$LOG/accuracy.log" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
for line in text.splitlines():
    if re.match(r"^(ACCURACY|GATE_SCORE)", line):
        print(line[:200])
EOF
echo "accuracy: $([ $rc -eq 0 ] && echo PASSED || echo "FAILED (see $LOG/accuracy.log)")"
exit $rc

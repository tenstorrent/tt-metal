#!/bin/bash
# Lever-1 sweep: resident rows (rmax) vs stream depth. Prints the 15s/10s median lines per config.
cd "$(dirname "$0")"
OUT=/tmp/claude-4015/-data-cglagovich-tt-metal/7bf353e5-b3ae-491a-987f-f299bc56b26a/scratchpad/sweep_rd.log
: > "$OUT"
for cfg in "15 14" "15 16" "12 18" "10 20" "8 22"; do
  set -- $cfg
  echo "== rmax=$1 depth=$2" >> "$OUT"
  TT_VSA_RMAX=$1 TT_VSA_DEPTH=$2 ./scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/sdpa/test_vsa_sdpa_perf.py -q -s > /tmp/claude-4015/-data-cglagovich-tt-metal/7bf353e5-b3ae-491a-987f-f299bc56b26a/scratchpad/sweep_one.log 2>&1
  grep -E "median (15s|10s).*stream|worst.*stream|RESULT" /tmp/claude-4015/-data-cglagovich-tt-metal/7bf353e5-b3ae-491a-987f-f299bc56b26a/scratchpad/sweep_one.log >> "$OUT"
done
echo SWEEP_DONE >> "$OUT"

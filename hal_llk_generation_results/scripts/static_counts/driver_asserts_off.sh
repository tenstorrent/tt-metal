#!/bin/bash
# Runs build_counts.sh for HAL runs 7a-7e and 8a one after another (asserts on + off, -k add only),
# then compare.py. Checks before and after that tools/perf_cycles/ws_8a (hard-link source) is unchanged.
# Start inside a free bh-30 board container (NOT the board of a running harness):
#   setsid nohup tools/asserts_off/driver_asserts_off.sh > tools/asserts_off/driver_nohup.log 2>&1 < /dev/null &
set -uo pipefail
B=/proj_sw/user_dev/ctr-sbratic; A=$B/tools/asserts_off; H=$B/results/hal-runs
P=$A/progress.txt
RUNS="7a:20261104_211905_B_c0e38f1f_claude-opus-5-5
7b:20261105_102435_B_c0e38f1f_claude-opus-5-5
7c:20261105_122703_B_c0e38f1f_claude-opus-5-5
7d:20261106_020508_B_c0e38f1f_claude-opus-5-5
7e:20261106_095649_B_c0e38f1f_claude-opus-5-5
8a:20261119_163437_B_c0e38f1f_claude-opus-5-5"
manifest() { (cd $B/tools/perf_cycles/ws_8a/tt_metal/tt-llk && find . -type f -not -path '*/__pycache__/*' \
  -not -path '*/.pytest_cache/*' -not -path './perf_data/*' -print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1); }
echo "driver start $(date -u +%FT%TZ) host $(hostname) pid $$" >> $P
m0=$(manifest); echo "ws_8a manifest before: $m0" >> $P
while IFS=: read -r n d; do
  [[ -f $A/STOP ]] && { echo "STOP file, stopping before $n" >> $P; break; }
  echo "$n start $(date -u +%FT%TZ)" >> $P
  bash $A/build_counts.sh "$n" "$H/$d" >> $P 2>&1; echo "$n exit $?" >> $P
done <<< "$RUNS"
m1=$(manifest); echo "ws_8a manifest after:  $m1 ($([[ $m0 == $m1 ]] && echo unchanged || echo CHANGED))" >> $P
python3 $A/compare.py $A/out $H --md $A/out/summary.md --json $A/out/summary.json >> $P 2>&1; echo "compare exit $?" >> $P
echo "driver end $(date -u +%FT%TZ)" >> $P

#!/bin/bash
# Static counts with LLK asserts on (control) and off for HAL runs 8b-8e (user 2026-11-22: "ok pokreni"),
# one after another with the UNCHANGED build_counts.sh, then compare.py for 8a-8e.
# Differences to driver_asserts_off.sh (7a-8a):
#  - run dirs come from tools/run<NAME>/run.txt (first two fields: name, run dir);
#  - an existing out/<NAME> is never overwritten (that run is skipped);
#  - the hard-link copy ws/<NAME> of tools/perf_cycles/ws_8a is made HERE, and its tests/test_run*.log are removed
#    before any pytest: the test framework rewrites those logs in place, so on 2026-11-21 it changed ws_8a's copies
#    through the hard links (logs only). build_counts.sh reuses an existing ws/<NAME>.
# Compile only, no card access. Start inside a free bh-30 board container:
#   setsid nohup bash tools/asserts_off/driver_asserts_off_8x.sh > tools/asserts_off/driver_8x_nohup.log 2>&1 < /dev/null &
set -uo pipefail
B=/proj_sw/user_dev/ctr-sbratic; A=$B/tools/asserts_off; P=$A/progress_8x.txt
NAMES=${NAMES:-"8b 8c 8d 8e"}
RUNS8A=$B/results/hal-runs/20261119_163437_B_c0e38f1f_claude-opus-5-5
manifest() { (cd $B/tools/perf_cycles/ws_8a/tt_metal/tt-llk && find . -type f -not -path '*/__pycache__/*' \
  -not -path '*/.pytest_cache/*' -not -path './perf_data/*' -print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1); }
now() { date -u +%FT%TZ; }
echo "driver start $(now) host $(hostname) pid $$ names: $NAMES" >> $P
m0=$(manifest); echo "ws_8a manifest before: $m0" >> $P
LIST="8a:$RUNS8A"
for n in $NAMES; do
  [[ -f $A/STOP ]] && { echo "STOP file, stopping before $n" >> $P; break; }
  read -r nm R _ < $B/tools/run$n/run.txt
  R=${R%/}
  if [[ "$nm" != "$n" || ! -d "$R/ws" ]]; then echo "$n: bad tools/run$n/run.txt ($nm $R), skipped" >> $P; continue; fi
  LIST+=",$n:$R"
  if [[ -e $A/out/$n ]]; then echo "$n: out/$n already exists, not rebuilt" >> $P; continue; fi
  W=$A/ws/$n
  if [[ ! -d $W ]]; then
    cp -al $B/tools/perf_cycles/ws_8a $W && rm -f $W/tt_metal/tt-llk/tests/test_run*.log \
      || { echo "$n: ws copy failed, skipped" >> $P; continue; }
  fi
  echo "$n start $(now), run dir $R" >> $P
  bash $A/build_counts.sh "$n" "$R" >> $P 2>&1; echo "$n exit $? $(now)" >> $P
done
m1=$(manifest); echo "ws_8a manifest after:  $m1 ($([[ $m0 == $m1 ]] && echo unchanged || echo CHANGED))" >> $P
python3 $A/compare.py $A/out $B/results/hal-runs --runs "$LIST" --md $A/out/summary_8x.md --json $A/out/summary_8x.json \
  >> $P 2>&1; echo "compare exit $? (1 = a problem is listed in summary_8x.md, e.g. a run without postcheck)" >> $P
echo "driver end $(now)" >> $P

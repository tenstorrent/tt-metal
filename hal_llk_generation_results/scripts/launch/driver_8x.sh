#!/bin/bash
# Runs 8c, 8d, 8e: three repeats of run 8b, one per free bh-30 board (user 2026-11-21: repeat as run8c,
# run8d, run8e on bh-30 on the remaining boards; dynamic analysis too, but only for HAL).
# Run 8b = the run 8a configuration (HAL arm, mode B, run-7a conditions, prompt-run6b-cfgcost.md) with the
# current English harness run_hal_gen.sh 4cfda0c8 and the 2026-11-21 folder layout.
# Each run in its own IRD container (one board each). orchestrate_8cde.sh starts the three 30 s apart,
# so every run's ws/hal-settings.json denies the other two run dirs (check_cross_deny_8x.sh checks it).
# Steps (the steps of run 8b):
#  1. harness, same environment as run 8b (see 8b logs/run_meta.txt), BG=0 so the driver waits
#  2. postcheck: tools/run7bcde/postcheck_7x.sh <run> <NAME>
#  3. perf cycles: tools/perf_cycles/setup_ws.sh <NAME> <run>, then driver_perf.sh p<NAME> <NAME>
#     (3 x original + this run, LLK asserts off) -> out/retro/p<NAME>_orig_r1..3 and <NAME>_r1..3
#  4. dynamic TT/TTI analysis of the HAL kernel ONLY: tools/dyn_gdb/run_hal_gdb.sh <run> run<NAME>
#     (its output and dyn_nohup.log are never read by the launcher; only tools/dyn_gdb/status/run<NAME>_summary.txt)
#  5. English report: tools/report_en/make_report_en.py <run> <NAME> -> <run>/report_en/
# Steps 2-5 run only if the harness wrote logs/SUMMARY.txt and claude_exit_code=0. A failed step 3 or 5
# does not stop the next step. `touch tools/run<NAME>/STOP` = no further step starts.
# Started by launch_8x.sh with setsid nohup. progress.txt gets only run dir, logs, exit codes, times.
set -u
NAME=${1:?usage: driver_8x.sh <NAME>}
B=/proj_sw/user_dev/ctr-sbratic
H=$B/harness/hal-harness-run5rules
D=$B/tools/run$NAME
P=$D/progress.txt
PC=$B/tools/perf_cycles
source $B/infra/hal-env-ird/.venv/bin/activate
# claude pinned to 2.1.287 (the version of runs 7a-8b); ~/.local/bin/claude was auto-updated on 2026-11-16.
export PATH=$B/tools/claude-2.1.287:$PATH:$HOME/.local/bin
export DISABLE_AUTOUPDATER=1
now() { date -u +%FT%TZ; }
stop_if_requested() {
  if [[ -e $D/STOP ]]; then
    echo "run$NAME: STOP file found, $1 not started $(now)" >> $P
    echo "driver end $(now)" >> $P
    exit 0
  fi
}

echo "driver start $(now), pid $$, container $(hostname), board $(ls /dev/tenstorrent | tr '\n' ' ')" >> $P

# ---- 1. harness
LOG=$B/results/hal-runs/harness_$(date +%Y%m%d_%H%M%S)_B_$NAME.log
echo "run$NAME harness start $(now), log $LOG" >> $P
(cd $H && BG=0 MODE=B TESTS="test_eltwise_binary:add" HAL_FILTER=0 PROMPT_FILE=$H/prompt-run6b-cfgcost.md \
  HAL_COMMIT=b9e522d39ae534f0281499dae11b425574d89d0d DOCS_COMMIT=d8c6fa7c22fb89e109178778a537e5860fae0ac6 \
  DOCS_INVALID="address_counters_hal_interface.html replay_hal_interface.html gpr_ops_hal_interface.html atomic_hal_interface.html" \
  MODEL=claude-opus-5-5 TIMEOUT=4h MAX_TURNS=500 MAX_BUDGET=60 ELFRAY=$B/tools/elfray \
  REPO=$B/repos/tt-metal RUNS_ROOT=$B/results/hal-runs RESULTS_ROOT=$B/results/hal-results \
  ./run_hal_gen.sh > $LOG 2>&1 < /dev/null)
RC=$?
# Run dir from the harness log line "Run: <dir>" (the other runs start at the same time on other boards).
R=$(grep -oP '^\[harness [0-9:]+\] Run: \K\S+' $LOG | head -1); R=${R%/}
echo "run$NAME harness exit $RC $(now), run dir ${R:-NONE}" >> $P
if [[ -z "$R" || ! -f $R/logs/SUMMARY.txt ]]; then
  echo "run$NAME: no run dir or no SUMMARY.txt, driver stops (no postcheck, perf, dynamic analysis, report)" >> $P
  echo "driver end $(now)" >> $P
  exit 1
fi
CEC=$(grep -oP '^claude_exit_code=\K.*' $R/logs/run_meta.txt)
if [[ "$CEC" != 0 ]]; then
  echo "run$NAME: claude_exit_code=$CEC (agent did not finish normally), driver stops (no postcheck, perf, dynamic analysis, report)" >> $P
  echo "driver end $(now)" >> $P
  exit 1
fi
echo "$NAME $R $LOG" > $D/run.txt

# ---- 2. postcheck
stop_if_requested postcheck
bash $B/tools/run7bcde/postcheck_7x.sh $R $NAME > $D/postcheck_nohup.log 2>&1 < /dev/null
echo "run$NAME postcheck end $(now), checks with exit 0: $(grep -c ': exit 0,' $R/postcheck_$NAME/checks_$NAME.txt 2>/dev/null) of 4" >> $P

# ---- 3. perf cycles
stop_if_requested perf
if bash $PC/setup_ws.sh $NAME $R > $D/perf_setup.log 2>&1 < /dev/null; then
  echo "run$NAME perf start $(now)" >> $P
  REPS=3 bash $PC/driver_perf.sh p$NAME $NAME > $D/perf_nohup.log 2>&1 < /dev/null
  echo "run$NAME perf exit $? $(now)" >> $P
else
  echo "run$NAME perf setup FAILED (see perf_setup.log), perf skipped $(now)" >> $P
fi

# ---- 4. dynamic analysis, HAL kernel only (run_hal_gdb.sh refuses if files of an earlier attempt exist)
stop_if_requested "dynamic analysis"
echo "run$NAME dynamic (HAL only) start $(now)" >> $P
bash $B/tools/dyn_gdb/run_hal_gdb.sh $R run$NAME > $D/dyn_nohup.log 2>&1 < /dev/null
echo "run$NAME dynamic exit $? $(now)" >> $P

# ---- 5. English report
stop_if_requested "English report"
python3 $B/tools/report_en/make_report_en.py $R $NAME > $D/report_en.log 2>&1 < /dev/null
echo "run$NAME English report exit $? $(now)" >> $P
echo "driver end $(now)" >> $P

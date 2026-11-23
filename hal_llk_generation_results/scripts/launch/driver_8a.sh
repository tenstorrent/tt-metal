#!/bin/bash
# Run 8a: ONE HAL-arm run (mode B) under the run 7a conditions, with one change: the prompt
# PROMPT_FILE=prompt-run6b-cfgcost.md (= prompt-run6b.md + rules on the instruction cost of cfg writes in
# per-tile code). On our bh-30 board 2 (IRD job 151073, the board of runs 7a-7e).
# Then the postcheck (4-step reset / no-reset sequence + tt_tti_counts, tools/run7bcde/postcheck_7x.sh).
# The dynamic TT/TTI analysis (tools/dyn_gdb/run_full_gdb.sh) runs ONLY if the file $D/DYNAMIC_YES exists
# when the postcheck ends; otherwise the driver ends and the dynamic analysis can be started later by hand.
# Started by launch_8a.sh with setsid nohup. progress.txt gets only run dir, logs, exit codes, times.
set -u
B=/proj_sw/user_dev/ctr-sbratic
H=$B/hal-harness-run5rules
D=$B/tools/run8a
P=$D/progress.txt
NAME=8a
source $B/hal-env-ird/.venv/bin/activate
# claude pinned to 2.1.287 (the version of runs 7a-7e); ~/.local/bin/claude was auto-updated to 2.1.325 on 2026-11-16.
export PATH=$B/tools/claude-2.1.287:$PATH:$HOME/.local/bin
export DISABLE_AUTOUPDATER=1

echo "driver start $(date -u +%FT%TZ), pid $$" >> $P
LOG=$B/hal-runs/harness_$(date +%Y%m%d_%H%M%S)_B.log
echo "run$NAME harness start $(date -u +%FT%TZ), log $LOG" >> $P
# Same environment as run 7a (tools/run7a/launch_7a.sh), except PROMPT_FILE and BG=0 (driver waits).
(cd $H && BG=0 MODE=B TESTS="test_eltwise_binary:add" HAL_FILTER=0 PROMPT_FILE=$H/prompt-run6b-cfgcost.md \
  HAL_COMMIT=b9e522d39ae534f0281499dae11b425574d89d0d DOCS_COMMIT=d8c6fa7c22fb89e109178778a537e5860fae0ac6 \
  DOCS_INVALID="address_counters_hal_interface.html replay_hal_interface.html gpr_ops_hal_interface.html atomic_hal_interface.html" \
  MODEL=claude-opus-5-5 TIMEOUT=4h MAX_TURNS=500 MAX_BUDGET=60 ELFRAY=$B/tools/elfray \
  REPO=$B/tt-metal RUNS_ROOT=$B/hal-runs RESULTS_ROOT=$B/hal-results \
  ./run_hal_gen.sh > $LOG 2>&1 < /dev/null)
RC=$?
# Run dir from the harness log line "Run: <dir>" (robust if other runs start on other boards meanwhile).
R=$(grep -oP '^\[harness [0-9:]+\] Run: \K\S+' $LOG | head -1); R=${R%/}
echo "run$NAME harness exit $RC $(date -u +%FT%TZ), run dir ${R:-NONE}" >> $P
if [[ -z "$R" || ! -f $R/logs/SUMMARY.txt ]]; then
  echo "run$NAME: no run dir or no SUMMARY.txt, driver stops (no postcheck, no dynamic analysis)" >> $P
  echo "driver end $(date -u +%FT%TZ)" >> $P
  exit 1
fi
CEC=$(grep -oP '^claude_exit_code=\K.*' $R/logs/run_meta.txt)
if [[ "$CEC" != 0 ]]; then
  echo "run$NAME: claude_exit_code=$CEC (agent did not finish normally), driver stops (no postcheck, no dynamic analysis)" >> $P
  echo "driver end $(date -u +%FT%TZ)" >> $P
  exit 1
fi
echo "$NAME $R" > $D/run.txt
bash $B/tools/run7bcde/postcheck_7x.sh $R $NAME > $D/postcheck_nohup.log 2>&1 < /dev/null
echo "run$NAME postcheck end $(date -u +%FT%TZ)" >> $P
if [[ ! -e $D/DYNAMIC_YES ]]; then
  echo "run$NAME: dynamic analysis not requested (no DYNAMIC_YES), driver ends" >> $P
  echo "driver end $(date -u +%FT%TZ)" >> $P
  exit 0
fi
# run_full_gdb.sh APPENDS to its status/output files: stop if status files with this name already exist.
if compgen -G "$B/tools/dyn_gdb/status/run${NAME}_*" > /dev/null; then
  echo "run$NAME: status files with this name already exist, dynamic analysis NOT started" >> $P
  echo "driver end $(date -u +%FT%TZ)" >> $P
  exit 1
fi
echo "run$NAME dynamic start $(date -u +%FT%TZ)" >> $P
bash $B/tools/dyn_gdb/run_full_gdb.sh $R run$NAME > $D/dyn_nohup.log 2>&1 < /dev/null
echo "run$NAME dynamic exit $? $(date -u +%FT%TZ)" >> $P
echo "driver end $(date -u +%FT%TZ)" >> $P

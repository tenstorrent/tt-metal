#!/bin/bash
# Runs 7b, 7c, 7d, 7e: four repeats of run 7a, one after another on our bh-30 board.
# Each run = the harness with exactly the environment of launch_7a.sh (BG=0 so this driver waits for it),
# then postcheck_7x.sh. Started by launch_7bcde.sh with setsid nohup.
# progress.txt gets only run dirs, logs, exit codes and times (no HAL content).
# Stop before the next run: touch $D/STOP
# Dynamic analysis (tools/dyn_gdb/run_full_gdb.sh) runs only if $D/DYNAMIC_YES exists; it is checked
# after every postcheck and once more at the end, and covers every finished run not yet analysed.
set -u
B=/proj_sw/user_dev/ctr-sbratic
H=$B/hal-harness-run5rules
D=$B/tools/run7bcde
P=$D/progress.txt
source $B/hal-env-ird/.venv/bin/activate
export PATH=$PATH:$HOME/.local/bin

dyn_pending() {
  [[ -f $D/DYNAMIC_YES ]] || return 0
  while read -r N R; do
    grep -qx "$N" $D/dyn_done.txt 2>/dev/null && continue
    echo "run$N dynamic start $(date -u +%FT%TZ)" >> $P
    bash $B/tools/dyn_gdb/run_full_gdb.sh $R run$N > $D/dyn_run${N}_nohup.log 2>&1 < /dev/null
    echo "$N" >> $D/dyn_done.txt
    echo "run$N dynamic end $(date -u +%FT%TZ)" >> $P
  done < $D/runs.txt
}

echo "driver start $(date -u +%FT%TZ), pid $$" >> $P
for NAME in 7b 7c 7d 7e; do
  [[ -f $D/STOP ]] && { echo "STOP file found, run$NAME not started $(date -u +%FT%TZ)" >> $P; break; }
  ls -d $B/hal-runs/*/ > $D/dirs_before_$NAME.txt
  LOG=$B/hal-runs/harness_$(date +%Y%m%d_%H%M%S)_B.log
  echo "run$NAME harness start $(date -u +%FT%TZ), log $LOG" >> $P
  (cd $H && BG=0 MODE=B TESTS="test_eltwise_binary:add" HAL_FILTER=0 PROMPT_FILE=$H/prompt-run6b.md \
    HAL_COMMIT=b9e522d39ae534f0281499dae11b425574d89d0d DOCS_COMMIT=d8c6fa7c22fb89e109178778a537e5860fae0ac6 \
    DOCS_INVALID="address_counters_hal_interface.html replay_hal_interface.html gpr_ops_hal_interface.html atomic_hal_interface.html" \
    MODEL=claude-opus-5-5 TIMEOUT=4h MAX_TURNS=500 MAX_BUDGET=60 ELFRAY=$B/tools/elfray \
    REPO=$B/tt-metal RUNS_ROOT=$B/hal-runs RESULTS_ROOT=$B/hal-results \
    ./run_hal_gen.sh > $LOG 2>&1 < /dev/null)
  RC=$?
  R=$(ls -d $B/hal-runs/*/ | grep -vxF -f $D/dirs_before_$NAME.txt | head -1); R=${R%/}
  echo "run$NAME harness exit $RC $(date -u +%FT%TZ), run dir ${R:-NONE}" >> $P
  if [[ -z "$R" || ! -f $R/logs/SUMMARY.txt ]]; then
    echo "run$NAME: no run dir or no SUMMARY.txt, driver stops" >> $P; break
  fi
  echo "$NAME $R" >> $D/runs.txt
  bash $D/postcheck_7x.sh $R $NAME > $D/postcheck_${NAME}_nohup.log 2>&1 < /dev/null
  echo "run$NAME postcheck end $(date -u +%FT%TZ)" >> $P
  dyn_pending
done
dyn_pending
echo "driver end $(date -u +%FT%TZ)" >> $P

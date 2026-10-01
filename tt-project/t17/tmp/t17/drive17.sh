#!/bin/bash
# On blx03, detached: wait for the project's one device slot, then queue the t17 A/B (conv145_t17).
# run25.sh sets TT_METAL_HOME=$W; the trailing overrides point it back at the main tree so the
# JIT cache stays warm, while models/ (the conv3d blocking table) comes from t17 (cwd + PYTHONPATH).
M=/home/smarton/fasth3/tt-metal; T=/home/smarton/fasth3/t17; LOG=/home/smarton/fasth3/drive17.log
O=/home/smarton/fasth3/out/ltx25_1080p_6s/conv145_t17
mkdir -p $O; [ -f $O/run.log ] && mv -f $O/run.log $O/run_884_cold.log
while true; do
  out=$(bash $M/tmp/blx03/submit.sh 900 "W=$T bash $M/tmp/blx03/run25.sh conv145_t17 LTX25_DIFFVAE=0 TT_METAL_HOME=$M PYTHONPATH=$T:$M/ttnn:$M/tools" 2>&1); rc=$?
  [ $rc -ne 75 ] && break; sleep 120
done
echo "$out" > $LOG; echo "SUBMIT_RC=$rc" >> $LOG

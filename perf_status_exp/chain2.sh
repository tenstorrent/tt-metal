#!/bin/bash
# chain2.sh <final_mode>: final sweep base vs final (alternating, 2 runs/side), then threshold sweep x2.
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522/perf_status_exp
F=${1:-new}
$W/final_side.sh fb1 base
$W/final_side.sh ff1 $F
$W/final_side.sh fb2 base
$W/final_side.sh ff2 $F
$W/thresh.sh th1 $F
$W/thresh.sh th2 $F
echo "CHAIN2_DONE" >> /localdev/mbezulj/logs/merge8_runs.txt

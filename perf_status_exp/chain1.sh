#!/bin/bash
# Step A/Bs (m7 = merge, m8 = +step 5, new = +step 6), correctness suites, second A/B pass reversed.
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522/perf_status_exp
$W/ab.sh ab1 m7 m8 new
$W/unit.sh new
$W/ab.sh ab2 new m8 m7
echo "CHAIN1_DONE" >> /localdev/mbezulj/logs/merge8_runs.txt

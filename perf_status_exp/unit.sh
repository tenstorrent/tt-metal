#!/bin/bash
# unit.sh <mode>: correctness suites on the merged build.
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522
MODE=${1:-new}
T=tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill
for f in test_single_routed_expert test_moe_fused_swiglu test_routed_expert_hybrid test_hybrid_routed_expert; do
  TMO=2400 $W/perf_status_exp/run.sh unit_$f $MODE $T/$f.py ""
done
echo "UNIT_DONE $MODE" >> /localdev/mbezulj/logs/merge8_runs.txt

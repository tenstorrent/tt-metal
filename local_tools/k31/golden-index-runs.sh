#!/bin/bash
# 8v1.6 validation: positive run on the real golden, then two negative runs (absent / mutated index layer 47).
cd /home/mvasilijevic/repos/tt-metal.worktrees/k31-perf-model
source python_env/bin/activate; export PYTHONPATH=$PWD
L=/home/mvasilijevic/repos/tt-metal.worktrees/.artifacts/k31-agents/golden-index
N=/tmp/claude-4227/-home-mvasilijevic-repos-tt-metal-worktrees/72922994-cfdf-468b-a766-f3350fb84175/scratchpad/neg
T=models/demos/deepseek_v3_d_p/tests/kimi_k3/test_k31_golden.py
ts(){ TZ=CET date +%H:%M:%S; }
touch $TT_DEVICE_PRIORITY_DIR/P1b-golden-index
wait_prio(){ while ls $TT_DEVICE_PRIORITY_DIR | grep -qE '^(P0|P1a)'; do echo "$(ts) waiting on higher priority marker"; sleep 30; done; }
run(){ name=$1; shift; wait_prio; echo "$(ts) START $name"; "$@" > $L/$name.log 2>&1; rc=$?; echo "$(ts) END $name rc=$rc"; grep -E "passed|failed|AssertionError|^E " $L/$name.log | tail -8; }
run positive-full scripts/run_safe_pytest.sh --run-all $T
run negative-absent env KIMI_K3_GOLDEN_TRACE=$N/absent scripts/run_safe_pytest.sh $T::test_k31_golden_serves_the_runner_gate
run negative-rowmismatch env KIMI_K3_GOLDEN_TRACE=$N/rowmismatch scripts/run_safe_pytest.sh $T::test_k31_golden_serves_the_runner_gate
rm -f $TT_DEVICE_PRIORITY_DIR/P1b-golden-index
echo "$(ts) DONE"

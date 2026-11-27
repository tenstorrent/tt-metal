#!/bin/bash
# Keeps the bh-lb-a10u24 reservation from idling: every 15 min, if none of my pytest processes ran on the node for
# 45 min, run the all_to_all_matmul unit tests on chips 0-3 (precompile off: the node's per-user process limit is
# shared with other sessions). Log: /data/malimpic/tmp/lb-guard.log
idle=0
while true; do
  job=$(squeue --me -h -t RUNNING -w bh-lb-a10u24 -o %i -n no-shell | head -1)
  [ -z "$job" ] && { echo "$(date -u +%T) no reservation job; stop"; exit 1; }
  if srun --jobid=$job --overlap -N1 pgrep -u $USER -f pytest >/dev/null 2>&1; then idle=0; else idle=$((idle+15)); fi
  echo "$(date -u +%T) job $job idle=${idle}min"
  if [ $idle -ge 45 ]; then
    cd /data/malimpic/tt-metal-atamm && TT_VISIBLE_DEVICES=0,1,2,3 timeout 1500 scripts/exabox/node-run bash -c '
      cd /data/malimpic/tt-metal-atamm; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0,1,2,3 TMPDIR=/data/malimpic/tmp
      source python_env/bin/activate
      scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/operations/all_to_all_matmul/test_all_to_all_matmul.py 2>&1 | grep -E "passed|failed|SAFE_PYTEST_RESULT"' 2>&1 | tail -2
    idle=0
  fi
  sleep 900
done

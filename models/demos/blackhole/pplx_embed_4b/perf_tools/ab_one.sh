#!/bin/bash
# same-chip sequential A/B for one batch size: ab_one.sh <bs> <chip> "<ENV_A>" "<ENV_B>" [iters]
BS=$1; CHIP=$2; EA=$3; EB=$4; IT=${5:-10}; S=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$S/../../../../.." && pwd)
cd "$REPO"; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_MODEL=perplexity-ai/pplx-embed-v1-4b MESH_DEVICE=P150 TT_VISIBLE_DEVICES=$CHIP
best() { grep -a 'Best ' "$1" | grep -aoE '[0-9]+\.[0-9]+ms' | tail -1; }
env $EA timeout 2400 ./python_env/bin/python $S/e2e_run_fp.py $BS $IT > /tmp/ab_${BS}_c${CHIP}_A.log 2>&1; A=$(best /tmp/ab_${BS}_c${CHIP}_A.log)
env $EB timeout 2400 ./python_env/bin/python $S/e2e_run_fp.py $BS $IT > /tmp/ab_${BS}_c${CHIP}_B.log 2>&1; B=$(best /tmp/ab_${BS}_c${CHIP}_B.log)
echo "RES ab bs$BS chip$CHIP  A[$EA]=$A  B[$EB]=$B  $(grep -aoE 'TT_FATAL: [^(]{0,60}|Error: [^(]{0,60}' /tmp/ab_${BS}_c${CHIP}_B.log | head -1)"

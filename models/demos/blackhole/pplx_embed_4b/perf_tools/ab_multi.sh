#!/bin/bash
# alternating multi-launch A/B for one batch size (for the bimodal bs16): ab_multi.sh <bs> <chip> "<ENV_A>" "<ENV_B>" <n_pairs> [iters]
BS=$1; CHIP=$2; EA=$3; EB=$4; N=${5:-3}; IT=${6:-10}; S=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$S/../../../../.." && pwd)
cd "$REPO"; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_MODEL=perplexity-ai/pplx-embed-v1-4b MESH_DEVICE=P150 TT_VISIBLE_DEVICES=$CHIP
best() { grep -a 'Best ' "$1" | grep -aoE '[0-9]+\.[0-9]+ms' | tail -1 | tr -d ms; }
A=(); B=()
for i in $(seq 1 $N); do
  env $EA timeout 2400 ./python_env/bin/python $S/e2e_run_fp.py $BS $IT > /tmp/abm_${BS}_c${CHIP}_A$i.log 2>&1; A+=($(best /tmp/abm_${BS}_c${CHIP}_A$i.log))
  env $EB timeout 2400 ./python_env/bin/python $S/e2e_run_fp.py $BS $IT > /tmp/abm_${BS}_c${CHIP}_B$i.log 2>&1; B+=($(best /tmp/abm_${BS}_c${CHIP}_B$i.log))
done
minA=$(printf "%s\n" "${A[@]}" | sort -n | head -1); minB=$(printf "%s\n" "${B[@]}" | sort -n | head -1)
echo "RES abm bs$BS chip$CHIP  A[$EA]=${A[*]} (min $minA)  B[$EB]=${B[*]} (min $minB)  $(grep -aoE 'TT_FATAL: [^(]{0,60}|Error: [^(]{0,60}' /tmp/abm_${BS}_c${CHIP}_B1.log | head -1)"

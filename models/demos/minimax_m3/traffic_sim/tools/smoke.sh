#!/bin/bash
# Smoke test every feature path at one concurrency. Usage: JOB=<slurm job> tools/smoke.sh [C]
cd "$(dirname "$0")/.." || exit 1
NODE=${NODE:-node}
C=${1:-256}
run() { echo "## $*"; ./on_node.sh "$NODE" run.js --preset today-C --conc "$C" "$@" 2>&1 | grep -v '^plan'; }
run
run --set boundedDense=true
run --set boundedDense=true --set cache=paging
run --set boundedDense=true --set cache=inf
run --set boundedDense=true --set cache=pool --set lanes=3
run --set boundedDense=true --set cache=pool --set laneArena=true --set arenaTokens=3000000
run --set boundedDense=true --set cache=pool --set lanes=3 --set hostTier=true
run --set boundedDense=true --set cache=inf --set unaligned=true
run --set boundedDense=true --set cache=inf --set layout=var
run --set boundedDense=true --set cache=inf --set layout=var --set batch=true --set budget=16384
run --set boundedDense=true --set cache=inf --set layout=fixed --set batch=true --set budget=8192
run --set boundedDense=true --set cache=inf --set asyncHandoff=true
run --set boundedDense=true --set cache=inf --set opEff=1
run --set boundedDense=true --set cache=inf --set opEff=1 --set layout=var --set batch=true --set budget=16384 --set asyncHandoff=true
run --set boundedDense=true --set cache=inf --set policy=srpt

#!/bin/bash
# Reproduce user-reported oddities: replicas, paging vs pool, chunk size. Usage: JOB=<id> tools/investigate.sh
cd "$(dirname "$0")/.." || exit 1
NODE=${NODE:-node}
C=${C:-256,384,512}
run() { echo "## $*"; ./on_node.sh "$NODE" run.js --conc "$C" "$@" 2>&1 | grep -v '^plan' | cut -c1-200; }
echo "### replicas"
run --preset today-C --set replicas=2
run --preset today-C --set replicas=2 --set stages=8
echo "### paging vs pool (best g4_k0 preset)"
run --study-preset g4_k0
run --study-preset g4_k0 --set cache=paging
run --study-preset g4_k0 --set hostTier=false
echo "### chunk size"
run --preset today-C --set chunk=1024 --conc 16,24,32
run --preset today-C --set chunk=2048 --conc 16,24,32
run --preset today-C --set chunk=5120 --conc 16,24,32
run --study-preset g4_k0 --set chunk=1024
run --study-preset g4_k0 --set chunk=5120
run --study-preset g4_k0 --set budget=4096
run --study-preset g4_k0 --set budget=32768

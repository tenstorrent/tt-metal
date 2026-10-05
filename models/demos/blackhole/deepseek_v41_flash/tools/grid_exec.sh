#!/bin/bash
# usage: GRID_KV=VAR=val+VAR2=val2 grid_exec.sh <command...>
# Provenance wrapper of the grid: unset EVERY DSV41_* / MOE_COMPUTE_* variable inherited from the shell (leftovers of short runs!), then export only the
# documented per-cell variables + DSV41_LAYERS=0-39 (all 40 layers), print the final environment into the log, and exec the command.
for v in $(compgen -e | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_(CACHE|HOME))'); do unset "$v"; done
export MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h44 TT_METAL_HOME=/mnt/tt-data/ssinghal/tests/tt-metal
export DSV41_LAYERS=0-39
for kv in ${GRID_KV//+/ }; do export "$kv"; done
unset GRID_KV
if [ "$DSV41_LAYERS" != "0-39" ]; then echo "GRID ENV ERROR: DSV41_LAYERS=$DSV41_LAYERS"; exit 3; fi
echo "GRID_ENV $(hostname -s) $(date +%FT%T) head=$(git -C /mnt/tt-data/ssinghal/tests/tt-metal rev-parse --short=11 HEAD)"
env | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_)' | sort | sed 's/^/GRID_ENV /'
exec "$@"

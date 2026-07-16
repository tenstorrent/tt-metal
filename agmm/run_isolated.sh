#!/usr/bin/env bash
# Isolated AG + MM sweeps -> dedicated CSVs (keeps the AGMM dashboard data separate).
set -uo pipefail
W=/data/cglagovich/tt-metal/.claude/worktrees/drifting-seeking-lantern
cd "$W"
export PATH="$W/python_env/bin:$PATH" ARCH_NAME=blackhole TT_METAL_HOME="$W" PYTHONPATH="$W"

echo "=== ISOLATED SWEEP START $(date +%F_%H-%M-%S) on $(hostname) ==="

echo "=== [1/2] isolated AG (15 shapes) ==="
python agmm/run_sweeps.py --shapes agmm/isolated_ag_spec.json --mode full \
  --history agmm/isolated_ag_history.csv --latest agmm/isolated_ag_latest.csv
echo "=== AG sweep rc=$? ==="

echo "=== [2/2] isolated MM (37 shapes) ==="
python agmm/run_sweeps.py --shapes agmm/isolated_mm_spec.json --mode full \
  --history agmm/isolated_mm_history.csv --latest agmm/isolated_mm_latest.csv
echo "=== MM sweep rc=$? ==="

echo "=== ISOLATED SWEEP DONE $(date +%F_%H-%M-%S) ==="

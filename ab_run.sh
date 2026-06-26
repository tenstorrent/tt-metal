#!/usr/bin/env bash
# A/B sweep: old all_gather_async vs new all_gather, 3 GLM-5.1 tp=4 reshard gathers, bf16/bfp8.
# Sequential (one device test at a time). Each cell -> own log + own copied ops_perf CSV.
#
# Configurable via env:
#   MESH    mesh id   (default 2x4; use 8x4 on Galaxy)
#   TOPO    topology  (default line; use ring on Galaxy — line-only on 8-chip boxes)
#   CONFIG  cfg id    (default trace_bar)
#   IMPLS   impls      (default "old new")
# Example (Galaxy): MESH=8x4 TOPO=ring bash ab_run.sh
set -u
cd "$(dirname "$0")"
source python_env/bin/activate 2>/dev/null
# websockets (needed by tracy) may live in user-site if the venv has no pip:
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PWD"
MESH="${MESH:-2x4}"
TOPO="${TOPO:-line}"
CONFIG="${CONFIG:-trace_bar}"
IMPLS="${IMPLS:-old new}"
TP=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reshard_ag_perf.py
OUT="ab_results_${MESH}_${TOPO}"
mkdir -p "$OUT/logs" "$OUT/csv"
echo "=== A/B sweep start $(date)  mesh=$MESH topo=$TOPO cfg=$CONFIG ==="
for impl in $IMPLS; do
  for op in q_heads out_seq qdev_heads; do
    for dt in bf16 bfp8; do
      K="$impl and $op and $CONFIG and $MESH and $TOPO and $dt"
      tag="${impl}_${op}_${dt}_${TOPO}"
      log="$OUT/logs/${tag}.log"
      echo ">>> CELL $tag start $(date)"
      timeout 900 python -m tracy -p -r -m "pytest $TP -k '$K' -s" > "$log" 2>&1
      rc=$?
      csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
      [ -n "$csv" ] && cp "$csv" "$OUT/csv/${tag}.csv"
      pcc=$(grep -oE "PCC: [0-9.]+" "$log" | tail -1)
      pass=$(grep -cE " 1 passed" "$log")
      echo ">>> CELL $tag rc=$rc pass=$pass $pcc csv=$(basename "${csv:-NONE}")"
    done
  done
done
echo "=== A/B sweep done $(date) ===  (parse with: python3 parse_ab.py $OUT/csv)"

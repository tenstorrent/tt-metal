#!/bin/bash
# back-to-back bench of two commits in the tirex2_base worktree (never touched while this runs):
#   b2b.sh <out dir> <-k expr> <base commit> <cand commit> [repeats]
set -e
out=$1; k=$2; base=$3; cand=$4; n=${5:-1}
cd /home/andy/bringup/tirex2_base
for i in $(seq $n); do
  for c in base:$base cand:$cand; do
    git checkout -q ${c#*:}
    MODEL_PERF_JSON=$out/b2b_${c%%:*}.json python -m pytest -q models/experimental/tirex2_forecast/tests/perf/test_forward_perf.py -k "$k" -s
  done
done

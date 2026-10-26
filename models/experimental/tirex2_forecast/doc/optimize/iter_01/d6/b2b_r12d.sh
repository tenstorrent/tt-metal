#!/bin/bash
# D6 (c): multivariate b2b, pre-R1-2d (c52e07a3f30) vs R1-2d (7c92331e8f2), both precisions, in tirex2_base
set -e
cd /home/andy/bringup/tirex2_base
for c in c52e07a3f30:pre 7c92331e8f2:r12d c52e07a3f30:pre2 7c92331e8f2:r12d2; do
  git checkout -q ${c%%:*}
  MODEL_PERF_JSON=/home/andy/bringup/tirex2/models/experimental/tirex2_forecast/doc/optimize/iter_01/d6/b2b_${c##*:}.json python -m pytest -q models/experimental/tirex2_forecast/tests/perf/test_forward_perf.py -k multivariate -s
done
git checkout -q 6af7772ff84

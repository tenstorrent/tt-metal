#!/usr/bin/env bash
# #58817: the g70b outputs that differ from main in the broad bits run, in the same module order, with a float64
# reference per output tile and the location of every differing element.
set +e +o pipefail
cd "$(git rev-parse --show-toplevel)"
export CI_FRESH=1 BITS_ASSERTS=${1:-0} CI_LIMIT=2400
bash scratch_r3/run_ci.sh scratch_r3/plan_gold.txt
for v in head main; do echo "== gold $v"; grep real generated/r3ci/bits_$v/gold.txt; done
echo "== where main differs from the head"; python scratch_r3/tools/gold_diff.py generated/r3ci main

#!/usr/bin/env bash
# #58817: the g70b outputs that differ from main in the broad bits run: the same module order per variant, twice per side,
# with a float64 reference per output tile.
set +e +o pipefail
cd "$(git rev-parse --show-toplevel)"
export CI_FRESH=1 BITS_ASSERTS=${1:-0} CI_LIMIT=2400
bash scratch_r3/run_ci.sh scratch_r3/plan_gold.txt
for v in head ctl main mainb; do echo "== gold $v"; cat generated/r3ci/bits_$v/gold.txt; done

#!/usr/bin/env bash
# scratch: op-level A/B of #59149 head and the no-MOP-patch fast untilize pack (pack-lever round)
export MM_REPS=${MM_REPS:-6} CI_LIMIT=${CI_LIMIT:-2400}
bash $(git rev-parse --show-toplevel)/scratch_pl/run_op.sh $(git rev-parse --show-toplevel)/scratch_pl/plan.txt

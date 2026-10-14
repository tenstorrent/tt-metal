#!/usr/bin/env bash
# scratch: harness confirmation then op-level A/B (pack-lever round)
R=$(git rev-parse --show-toplevel)
bash $R/ci_ab/run_pl.sh
cd $R && git checkout -- . 2>/dev/null
MM_REPS=6 CI_LIMIT=2400 bash $R/scratch_pl/run_op.sh $R/scratch_pl/plan.txt

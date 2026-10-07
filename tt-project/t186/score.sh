#!/bin/bash
# t186 rescoring: post.sh put sbs_*.mp4 next to seed*.mp4 and ltx_eval found no reference for them.
# Move the side-by-side clips/stills into sbs/ and rerun ltx_eval on seed*.mp4 only (as t185 does).
# Usage: bash score.sh <label>   Marker: $R/SCORE.done = "<code> <reason>"
set -eo pipefail
BASE=/home/smarton/fasth3/tt-metal; S=$BASE/tt-project/worktrees/t170; W=$BASE/tt-project/worktrees/t158
REF=$BASE/tt-project/data/g15/ref_t48_f6b8
label=${1:?label}; R=$BASE/tt-project/data/g15/t186_$label
reason="died"
trap 'echo "$? $reason" > $R/SCORE.done' EXIT
mkdir -p $R/sbs
mv -f $R/sbs_seed* $R/sbs/ 2>/dev/null || true
reason="score"
cd $S
source $BASE/python_env/bin/activate
export PYTHONPATH=$S:$W/ttnn:$W/tools LTX_EVAL_THREADS=8 HF_HUB_OFFLINE=1
nice -n 19 timeout 2400 python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir $R --ref-dir $REF \
  --out $R/eval_vs_ref_t48_f6b8 --jobs 3 --vbench-ref \
  --vbench subject_consistency,background_consistency,motion_smoothness,imaging_quality,aesthetic_quality \
  < /dev/null > $R/eval_vs_ref_t48_f6b8.log 2>&1
reason="ok"

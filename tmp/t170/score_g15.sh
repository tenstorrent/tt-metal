#!/bin/bash
# t170 phase-2 scoring on g15blx02 (CPU only): fetch the 5-seed clips (mp4 + their prompt/seed sidecars) of
# each phase-2 label from blx01 and score them per seed against ref_dv145 with PCC/PSNR and VBench
# (ltx_eval batch --vbench-ref, niced, 3 clips at a time). gen#N of a 5-seed job is seed N-1 on the default
# prompt, the ref_dv145 protocol. Also fetches baseline/<winner> pack clips for the visual check.
# Usage: bash score_g15.sh <label>...   Marker: $D/SCORE.done = "<code> <reason>"
set -eo pipefail
BASE=/home/smarton/fasth3/tt-metal; S=$BASE/tt-project/worktrees/t170; W=$BASE/tt-project/worktrees/t158
D=$BASE/tt-project/data/g15/t170; REF=$BASE/tt-project/baselines/ltx25_1080p_6s/ref_dv145
REMOTE=blx01:/var/tmp/fasth3/t170/res
mkdir -p $D
reason="died"
trap 'echo "$? $reason" > $D/SCORE.done' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $D/score.log; }
cd $S
source $BASE/python_env/bin/activate
export PYTHONPATH=$S:$W/ttnn:$W/tools LTX_EVAL_THREADS=8 HF_HUB_OFFLINE=1
for label in "$@"; do
  d=$D/$label; mkdir -p $d/seeds
  reason="fetch $label"
  for i in 0 1 2 3 4; do
    g=$((i + 1))
    scp -q $REMOTE/$label/ltx_av_fast_1920x1088_$g.mp4 $d/seeds/seed$i.mp4
    scp -q $REMOTE/$label/ltx_av_fast_1920x1088_$g.json $d/seeds/seed$i.json 2>/dev/null || log "$label seed$i: no sidecar"
  done
  reason="score $label"
  vrc=0
  nice -n 19 timeout 3000 python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir $d/seeds \
    --ref-dir $REF --out $d/vbench --jobs 3 --vbench-ref < /dev/null > $d/vbench.log 2>&1 || vrc=$?
  log "$label rc=$vrc: $(grep -E '^BATCH' $d/vbench.log | cut -c1-400)"
done
reason="ok"

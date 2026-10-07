#!/bin/bash
# t186 post-processing on g15blx02 (CPU only). Fetch blx01 job's 5-seed clips (gen#N = seed N-1, DEFAULT prompt)
# with sidecars, score them against ref_t48_f6b8 (PCC/PSNR + VBench incl. aesthetic, reference scored too),
# and write per-seed stills and a side-by-side (reference | candidate) mp4 for the visual check.
# Usage: bash post.sh <label>   Marker: $R/POST.done = "<code> <reason>"
set -eo pipefail
BASE=/home/smarton/fasth3/tt-metal; S=$BASE/tt-project/worktrees/t170; W=$BASE/tt-project/worktrees/t158
REF=$BASE/tt-project/data/g15/ref_t48_f6b8
label=${1:?label}; R=$BASE/tt-project/data/g15/t186_$label; REMOTE=g15blx01:/var/tmp/fasth3/t186/res/$label
mkdir -p $R
reason="died"
trap 'echo "$? $reason" > $R/POST.done' EXIT
reason="fetch"
scp -q $REMOTE/run.log $R/run.log
for i in 0 1 2 3 4; do
  g=$((i + 1))
  scp -q $REMOTE/ltx_av_fast_1920x1088_$g.mp4 $R/seed$i.mp4
  scp -q $REMOTE/ltx_av_fast_1920x1088_$g.json $R/seed$i.json
done
reason="sbs"
for i in 0 1 2 3 4; do
  ffmpeg -loglevel error -y -i $REF/seed$i.mp4 -i $R/seed$i.mp4 \
    -filter_complex "[0:v]scale=960:-2[a];[1:v]scale=960:-2[b];[a][b]hstack" -map 1:a? -c:v libx264 -crf 20 \
    -c:a copy $R/sbs_seed$i.mp4
  ffmpeg -loglevel error -y -ss 3 -i $R/sbs_seed$i.mp4 -frames:v 1 $R/sbs_seed${i}_t3s.png
done
ffmpeg -loglevel error -y -ss 3 -i $R/seed0.mp4 -frames:v 1 $R/seed0_t3s.png
reason="score"
cd $S
source $BASE/python_env/bin/activate
export PYTHONPATH=$S:$W/ttnn:$W/tools LTX_EVAL_THREADS=8 HF_HUB_OFFLINE=1
nice -n 19 timeout 2400 python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir $R --ref-dir $REF \
  --out $R/eval_vs_ref_t48_f6b8 --jobs 3 --vbench-ref \
  --vbench subject_consistency,background_consistency,motion_smoothness,imaging_quality,aesthetic_quality \
  < /dev/null > $R/eval_vs_ref_t48_f6b8.log 2>&1 || echo "ltx_eval rc=$?" >> $R/eval_vs_ref_t48_f6b8.log
reason="ok"

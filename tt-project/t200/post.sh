#!/bin/bash
# t200 post-processing on g15blx02 (CPU only). Fetch the blx01 job's 5-seed clips (gen#N = seed N-1, DEFAULT prompt),
# score them against ref_t48_s2x2 (PCC/PSNR + VBench incl. aesthetic, reference scored too), and write the visual
# check material: seed 0 frames every 0.1 s over t=2.6-3.4 s (#186's face smear), a 1 fps ref|cand strip per seed,
# side-by-side mp4s under sbs/ (kept out of the cand dir so ltx_eval only sees seed*.mp4).
# Usage: bash post.sh <label>   Marker: $R/POST.done = "<code> <reason>"
set -eo pipefail
BASE=/home/smarton/fasth3/tt-metal; S=$BASE/tt-project/worktrees/t170; W=$BASE/tt-project/worktrees/t158
REF=$BASE/tt-project/data/g15/ref_t48_s2x2; D=$BASE/tt-project/t200
label=${1:?label}; R=$BASE/tt-project/data/g15/t200_$label; REMOTE=g15blx01:/var/tmp/fasth3/t200/res/$label
mkdir -p $R/sbs $R/vis
reason="died"
trap 'echo "$? $reason" > $R/POST.done' EXIT
reason="fetch"
scp -q $REMOTE/run.log $R/run.log
for i in 0 1 2 3 4; do
  g=$((i + 1))
  scp -q $REMOTE/ltx_av_fast_1920x1088_$g.mp4 $R/seed$i.mp4
  scp -q $REMOTE/ltx_av_fast_1920x1088_$g.json $R/seed$i.json
done
reason="visual"
ffmpeg -loglevel error -y -ss 3 -i $R/seed0.mp4 -frames:v 1 $R/seed0_t3s.png
for who in ref cand; do
  src=$R/seed0.mp4; [ $who = ref ] && src=$REF/seed0.mp4
  ffmpeg -loglevel error -y -ss 2.6 -t 0.85 -i $src -vf "fps=10,scale=640:-2,tile=3x3" -frames:v 1 $R/vis/seed0_${who}_t2.6-3.4.png
done
for i in 0 1 2 3 4; do
  ffmpeg -loglevel error -y -i $REF/seed$i.mp4 -i $R/seed$i.mp4 \
    -filter_complex "[0:v]fps=1,scale=320:-2,tile=6x1[a];[1:v]fps=1,scale=320:-2,tile=6x1[b];[a][b]vstack" \
    -frames:v 1 $R/vis/seed${i}_strip_1fps_ref_top.png
  ffmpeg -loglevel error -y -i $REF/seed$i.mp4 -i $R/seed$i.mp4 \
    -filter_complex "[0:v]scale=960:-2[a];[1:v]scale=960:-2[b];[a][b]hstack" -map 1:a? -c:v libx264 -crf 20 \
    -c:a copy $R/sbs/sbs_seed$i.mp4
done
reason="score"
cd $S
source $BASE/python_env/bin/activate
export PYTHONPATH=$S:$W/ttnn:$W/tools LTX_EVAL_THREADS=8 HF_HUB_OFFLINE=1
nice -n 19 timeout 2400 python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir $R --ref-dir $REF \
  --out $R/eval_vs_ref_t48_s2x2 --jobs 3 --vbench-ref \
  --vbench subject_consistency,background_consistency,motion_smoothness,imaging_quality,aesthetic_quality \
  < /dev/null > $R/eval_vs_ref_t48_s2x2.log 2>&1 || echo "ltx_eval rc=$? (1 = QUALITY/BATCH FAIL, scores written)" >> $R/eval_vs_ref_t48_s2x2.log
reason="ok"

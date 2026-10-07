#!/bin/bash
# t235 driver (g15blx02, CPU only, no device): encode #232's kp1 host-noise yuvs (seeds 0-4) on blx01 into mp4
# with the #214 ref settings (x264 crf 12, 24 fps), fetch kp1 + ref mp4s, then ltx_eval batch
# (cand kp1, ref = unoptimized DiffVAE ref, --vbench-ref, 5 dims). Marker: $R/LOCAL.done "<rc> <stage>"
set -eo pipefail
BASE=/home/smarton/fasth3/tt-metal; S=$BASE/tt-project/worktrees/t170; W=$BASE/tt-project/worktrees/t158
R=$BASE/tt-project/data/g15/t235; RM=/var/tmp/fasth3/t235; K=/var/tmp/fasth3/t232/out/kp1; REF=/var/tmp/fasth3/diffvae/ref
mkdir -p $R/kp1 $R/ref
stage=encode
trap 'echo "$? $stage" > $R/LOCAL.done' EXIT
ssh g15blx01 "set -e; mkdir -p $RM; for s in 0 1 2 3 4; do
  ffmpeg -loglevel error -y -f rawvideo -pix_fmt yuv420p -s 1920x1088 -r 24 -i $K/ref_dvx_seed\$s.yuv \
    -c:v libx264 -crf 12 -pix_fmt yuv420p $RM/kp1_seed\$s.mp4; done; ls -la $RM"
stage=fetch
for s in 0 1 2 3 4; do
  scp -q g15blx01:$RM/kp1_seed$s.mp4 $R/kp1/seed$s.mp4
  scp -q g15blx01:$REF/ref_dvx_seed$s.mp4 $R/ref/seed$s.mp4
done
stage=vbench
cd $S
source $BASE/python_env/bin/activate
export PYTHONPATH=$S:$W/ttnn:$W/tools LTX_EVAL_THREADS=8 HF_HUB_OFFLINE=1
nice -n 19 timeout 3600 python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir $R/kp1 --ref-dir $R/ref \
  --out $R/eval --jobs 3 --vbench-ref \
  --vbench subject_consistency,background_consistency,motion_smoothness,imaging_quality,aesthetic_quality \
  < /dev/null > $R/eval.log 2>&1
stage=done

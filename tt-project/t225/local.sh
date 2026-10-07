#!/bin/bash
# t225 local driver (g15blx02, CPU only): wait for the blx01 driver marker, fetch the 10 mp4s + post outputs,
# then VBench both arms (ltx_eval batch: cand = 2d, ref = 1d, --vbench-ref) at reduced priority.
# Marker: $R/LOCAL.done "<rc> <stage>"
set -eo pipefail
BASE=/home/smarton/fasth3/tt-metal; S=$BASE/tt-project/worktrees/t170; W=$BASE/tt-project/worktrees/t158
R=$BASE/tt-project/data/g15/t225; RM=/var/tmp/fasth3/t225
mkdir -p $R/1d $R/2d $R/post
stage=wait
trap 'echo "$? $stage" > $R/LOCAL.done' EXIT
for i in $(seq 600); do
  ssh -o ConnectTimeout=20 g15blx01 test -e $RM/drv/driver.marker && break || sleep 60
done
stage=marker
m=$(ssh g15blx01 cat $RM/drv/driver.marker)
echo "$m" > $R/driver.marker
echo "$m" | grep -q 'stage=done rc=0' || exit 3
stage=fetch
for s in 0 1 2 3 4; do
  scp -q g15blx01:$RM/out/1d_seed$s.mp4 $R/1d/seed$s.mp4
  scp -q g15blx01:$RM/out/2d_seed$s.mp4 $R/2d/seed$s.mp4
done
scp -q "g15blx01:$RM/out/{run.log,post.log,cmp.json,decode_times_1d.json,decode_times_2d.json,still_*.jpg,crop_*.png,MD5SUMS}" $R/post/
scp -q g15blx01:$RM/drv/driver.log $R/post/
stage=vbench
cd $S
source $BASE/python_env/bin/activate
export PYTHONPATH=$S:$W/ttnn:$W/tools LTX_EVAL_THREADS=8 HF_HUB_OFFLINE=1
nice -n 19 timeout 3600 python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir $R/2d --ref-dir $R/1d \
  --out $R/eval --jobs 3 --vbench-ref \
  --vbench subject_consistency,background_consistency,motion_smoothness,imaging_quality,aesthetic_quality \
  < /dev/null > $R/eval.log 2>&1
stage=done

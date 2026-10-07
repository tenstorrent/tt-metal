#!/bin/bash
# t171 post-processing on g15blx02 (CPU only). Fetch blx01 job's 5-seed clips (gen#N = seed N-1, default
# prompt) with their prompt/seed sidecars into the new reference dir, compare each byte-for-byte with #170's
# fast5 seed, fall back to decoded-frame md5 and ltx_eval PCC/PSNR when the files differ, and write a seed-0
# still at t=3 s. Usage: bash post.sh [label]   Marker: $R/POST.done = "<code> <reason>"
set -eo pipefail
BASE=/home/smarton/fasth3/tt-metal; S=$BASE/tt-project/worktrees/t170; W=$BASE/tt-project/worktrees/t158
R=$BASE/tt-project/data/g15/ref_t48_f6b8; F5=$BASE/tt-project/data/g15/t170/fast5/seeds
label=${1:-def5}; REMOTE=g15blx01:/var/tmp/fasth3/t171/res/$label
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
reason="compare"
: > $R/identity.txt
diff_seeds=0
for i in 0 1 2 3 4; do
  if cmp -s $R/seed$i.mp4 $F5/seed$i.mp4; then
    echo "seed$i bytes identical to t170 fast5" >> $R/identity.txt
  else
    a=$(ffmpeg -loglevel error -i $R/seed$i.mp4 -map 0:v -f md5 - | cut -d= -f2)
    b=$(ffmpeg -loglevel error -i $F5/seed$i.mp4 -map 0:v -f md5 - | cut -d= -f2)
    echo "seed$i bytes differ; decoded video md5 $a vs $b ($([ "$a" = "$b" ] && echo same || echo differ))" >> $R/identity.txt
    diff_seeds=$((diff_seeds + 1))
  fi
done
reason="still"
ffmpeg -loglevel error -y -ss 3 -i $R/seed0.mp4 -frames:v 1 $R/seed0_t3s.png
if [ $diff_seeds -gt 0 ]; then
  reason="score"
  cd $S
  source $BASE/python_env/bin/activate
  export PYTHONPATH=$S:$W/ttnn:$W/tools LTX_EVAL_THREADS=8 HF_HUB_OFFLINE=1
  for ref in fast5:$F5 dv145:$BASE/tt-project/baselines/ltx25_1080p_6s/ref_dv145; do
    nice -n 19 timeout 1800 python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir $R \
      --ref-dir ${ref#*:} --out $R/eval_vs_${ref%%:*} --jobs 3 --vbench none < /dev/null \
      > $R/eval_vs_${ref%%:*}.log 2>&1 || true
  done
fi
reason="ok"

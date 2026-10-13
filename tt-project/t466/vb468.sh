#!/bin/bash
# #468 VBench (g15blx02, CPU, no device): waits for chain468b, copies the 6 s A/B mp4s of every seed whose pair
# passed (both arms device-upscaled to 1920x1080, so no host resize), then runs the ltx_eval harness (t421 tree)
# with lofi as candidate and base as reference. Also writes ffmpeg PSNR/SSIM per seed and stills at 1 s / 5.5 s.
set -eo pipefail
RD=${1:?run dir holding chain468b.rc}; D=/home/smarton/fasth3/tt-metal/tt-project/state/t468
for i in $(seq 480); do [ -s $RD/chain468b.rc ] && break; sleep 30; done
[ -s $RD/chain468b.rc ] || { echo "chain468b not done"; exit 20; }
mkdir -p $D/vb/base $D/vb/lofi $D/stills
n=0
for s in 0 1 2 3 4; do
  for a in base lofi; do
    scp -q g15blx01:/var/tmp/fasth3/t466/out_ab6s${s}_$a/fl2va_turbo_1344x768_6s_4fwd.mp4 $D/vb/$a/s$s.mp4 || true
  done
  [ -s $D/vb/base/s$s.mp4 ] && [ -s $D/vb/lofi/s$s.mp4 ] || { rm -f $D/vb/*/s$s.mp4; echo "s$s missing"; continue; }
  n=$((n + 1))
  echo "s$s $(ffmpeg -nostdin -i $D/vb/lofi/s$s.mp4 -i $D/vb/base/s$s.mp4 -lavfi '[0:v][1:v]psnr' -f null - 2>&1 | grep -o 'average:[0-9.inf]*') $(ffmpeg -nostdin -i $D/vb/lofi/s$s.mp4 -i $D/vb/base/s$s.mp4 -lavfi '[0:v][1:v]ssim' -f null - 2>&1 | grep -o 'All:[0-9.]*')" | tee -a $D/psnr6s.txt
  for a in base lofi; do for t in 1 5.5; do
    ffmpeg -nostdin -loglevel error -y -ss $t -i $D/vb/$a/s$s.mp4 -frames:v 1 -vf scale=960:-1 $D/stills/s${s}_${a}_t$t.png
  done; done
done
[ $n -gt 0 ] || exit 21
cd /home/smarton/fasth3/tt-metal/tt-project/state/t421/evaltree
LTX_EVAL_THREADS=8 timeout 5400 nice -n 10 /home/smarton/fasth3/tt-metal/python_env/bin/python -m models.tt_dit.tests.models.ltx.tools.ltx_eval \
  batch --cand-dir $D/vb/lofi --ref-dir $D/vb/base --out $D/vbench --vbench-ref --jobs 2

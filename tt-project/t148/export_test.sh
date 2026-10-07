#!/bin/bash
# t148: PSNR/SSIM cost of the t48 export (libx264 ultrafast/crf20) vs slower presets.
# Source = decoded #171 seed0 frames (already lossy once; proxy for the pre-encode frames).
set -eo pipefail
SRC=${SRC:-/home/smarton/fasth3/tt-metal/tt-project/data/g15/ref_t48_f6b8/seed0.mp4}
T=/tmp/t148; cd $T
ffmpeg -v error -y -i $SRC -an -f rawvideo -pix_fmt yuv420p src.yuv
RAW="-f rawvideo -pix_fmt yuv420p -s 1920x1088 -r 24 -i src.yuv"
for cfg in ${CFGS:-ultrafast:20 veryfast:20 medium:20 slow:20 veryfast:23 slow:18}; do
  p=${cfg%:*}; c=${cfg#*:}; o=enc_${p}_${c}.mp4
  s=$(date +%s.%N); ffmpeg -v error -y $RAW -c:v libx264 -preset $p -crf $c -pix_fmt yuv420p $o; e=$(date +%s.%N)
  m=$(ffmpeg -i $o $RAW -lavfi "[0:v][1:v]psnr=stats_file=psnr_${p}_${c}.log;[0:v][1:v]ssim" -f null - 2>&1 | grep -oE 'PSNR y:[0-9.]+ u:[0-9.]+ v:[0-9.]+ average:[0-9.]+ min:[0-9.]+|SSIM Y:[0-9.]+ .*All:[0-9.]+' | tr '\n' ' ')
  echo "$p crf$c size_MB=$(echo "scale=2;$(stat -c%s $o)/1048576"|bc) enc_s=$(echo "$e-$s"|bc) $m"
done

#!/bin/bash
# On blx03: timings and output match of the t17 A/B (conv145_t17) against baseline job 879 (conv145_t20).
O=~/fasth3/out/ltx25_1080p_6s; A=$O/conv145_t20; B=$O/conv145_t17
for d in $A $B; do echo "== $d"; grep -E "RUN_EXIT|passed|failed" $d/run.log | tail -2
  grep -E "E2E_WALL_S|VAE decode|vae_decode|LTX_TIME" $d/run.log | tail -12 | cut -c1-220; done
for i in 0 1; do
  a=$A/ltx_av_fast_1920x1088_$i.mp4; b=$B/ltx_av_fast_1920x1088_$i.mp4
  echo "gen$i md5(decoded video): $(ffmpeg -v error -i $a -map 0:v -f md5 -) vs $(ffmpeg -v error -i $b -map 0:v -f md5 -)"
  ffmpeg -v error -i $a -i $b -lavfi "[0:v][1:v]psnr" -f null - 2>&1 | tail -1
  ffmpeg -hide_banner -i $a -i $b -lavfi "[0:v][1:v]psnr" -f null - 2>&1 | grep -o "PSNR.*" | tail -1
done
ffmpeg -v error -y -ss 3 -i $B/ltx_av_fast_1920x1088_1.mp4 -frames:v 1 $B/conv145_t17_gen1_t3s.png && echo still=$B/conv145_t17_gen1_t3s.png

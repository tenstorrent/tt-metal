#!/bin/bash
# t195 post-processing on g15blx02 (CPU only). Wait for blx01 job $1 to end, fetch its 5-seed clips
# (gen#N = seed N-1, DEFAULT prompt) with sidecars into data/g15/ref_t48_s2x2, compare byte-for-byte with
# #185's opt-in run of the same schedule (t185_s2x2, job 758), and write a seed-0 still at t=3 s.
# Marker: $R/POST.done = "<code> <reason>"
set -eo pipefail
JOB=$1
BASE=/home/smarton/fasth3/tt-metal
R=$BASE/tt-project/data/g15/ref_t48_s2x2; P=$BASE/tt-project/data/g15/t185_s2x2
REMOTE=blx01:/var/tmp/fasth3/t195/res/ref5
mkdir -p $R
reason="died"
trap 'echo "$? $reason" > $R/POST.done' EXIT
reason="wait"
for _ in $(seq 1 180); do
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes blx01 "tt-device-mcp status -j $JOB 2>&1 | sed -n 's/^Status: *//p'" || echo unreachable)
  case "$st" in queued|running|unreachable|"") sleep 20 ;; *) break ;; esac
done
echo "job $JOB status: $st" > $R/job_status.txt
reason="fetch"
scp -q $REMOTE/run.log $R/run.log
grep -q 'T195_EXIT\[ref5\]=0' $R/run.log || { reason="job_not_ok"; exit 7; }
for i in 0 1 2 3 4; do
  g=$((i + 1))
  scp -q $REMOTE/ltx_av_fast_1920x1088_$g.mp4 $R/seed$i.mp4
  scp -q $REMOTE/ltx_av_fast_1920x1088_$g.json $R/seed$i.json
done
reason="compare"
: > $R/identity.txt
for i in 0 1 2 3 4; do
  if cmp -s $R/seed$i.mp4 $P/seed$i.mp4; then
    echo "seed$i bytes identical to t185_s2x2 (job 758)" >> $R/identity.txt
  else
    a=$(ffmpeg -loglevel error -i $R/seed$i.mp4 -map 0:v -f md5 - | cut -d= -f2)
    b=$(ffmpeg -loglevel error -i $P/seed$i.mp4 -map 0:v -f md5 - | cut -d= -f2)
    echo "seed$i bytes differ; decoded video md5 $a vs $b ($([ "$a" = "$b" ] && echo same || echo differ))" >> $R/identity.txt
  fi
done
reason="still"
ffmpeg -loglevel error -y -ss 3 -i $R/seed0.mp4 -frames:v 1 $R/seed0_t3s.png
reason="ok"

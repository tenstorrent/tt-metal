#!/usr/bin/env bash
# #37: compare the reuse=0 and reuse=1 runs: byte identity of mp4s and S2 latents per gen, S2 timings, a still.
OUT=/home/smarton/fasth3/out/t37
A=$OUT/s2reuse0 B=$OUT/s2reuse1
for L in $A $B; do echo "== $(basename $L): $(grep -E 'passed|failed' $L/run.log | tail -1)"; done
echo "== mp4 md5"
md5sum $A/*.mp4 $B/*.mp4
for L in $A $B; do
  echo "== S2 timings $(basename $L) (per gen: denoise init, steps, total)"
  awk '/generate:.*- Stage 2: /{s=1} s&&/denoise init/{print "  " substr($0, index($0,"denoise init"))}
       s&&/STEP_MS/{sub(/.*STEP_MS=/,"");printf "  step_ms %s\n",$0}
       /Stage 2 denoise: /{s=0} /E2E_WALL_S/{sub(/.*E2E_WALL_S/,"E2E_WALL_S");print "  " $0}' $L/run.log
  grep -E "│ Stage 2 denoise" $L/run.log | sed 's/^/  table: /'
done
echo "== S2 latents (torch.equal per gen)"
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
python - <<EOF
import glob, os, torch
for fa in sorted(glob.glob("$A/lat.gen*.pt")):
    fb = "$B/" + os.path.basename(fa)
    if not os.path.exists(fb):
        print(os.path.basename(fa), "missing in reuse1"); continue
    a, b = torch.load(fa), torch.load(fb)
    for k in ("video", "audio"):
        eq = torch.equal(a[k], b[k])
        d = (a[k] - b[k]).abs().max().item()
        print(os.path.basename(fa), k, tuple(a[k].shape), "equal" if eq else f"DIFF max_abs={d:.3g}")
EOF
for f in $B/ltx_av_fast_*_1.mp4; do ffmpeg -loglevel error -y -ss 3 -i $f -frames:v 1 $OUT/s2reuse1_gen1_t3s.png && echo "still: $OUT/s2reuse1_gen1_t3s.png"; done

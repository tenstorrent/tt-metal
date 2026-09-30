#!/bin/bash
# t19, on g15blx02: pull the conv run's stage-2 latents, mp4s and log from blx03, make 3 s stills, check determinism,
# and (when t16's DiffVAE seeds5 run exists) score full-frame device conv vs device DiffVAE with ffmpeg.
set -e
H=$(cd "$(dirname "$0")" && pwd); D=$H/data; mkdir -p $D/lat $D/mp4 $D/stills
R=g14blx03; RO=fasth3/out/ltx25_1080p_6s
scp -q "$R:/var/tmp/fasth3/t19/lat.gen*.pt" $D/lat/
scp -q "$R:$RO/t19_conv3/ltx_av_fast_1920x1088_seed*.mp4" "$R:$RO/t19_conv3/run.log" $D/mp4/
PY=/home/smarton/fasth3/tt-metal/python_env/bin/python
$PY - $D/lat <<'PYEOF'
import sys, glob, os, torch
d = sys.argv[1]
by_seed = {}
for p in sorted(glob.glob(f"{d}/lat.gen*.pt")):
    x = torch.load(p)
    print(os.path.basename(p), tuple(x["video"].shape), x["flags"])
    by_seed.setdefault(x["flags"]["seed"], []).append((p, x["video"]))
for s, v in by_seed.items():
    # The same seed twice (gen1 replay and the seed0 pass) must give the same latent, or the A/B is racing noise.
    for p, t in v[1:]:
        print(f"seed {s}: {os.path.basename(v[0][0])} vs {os.path.basename(p)} max|diff| {(t - v[0][1]).abs().max().item():.3g}")
    os.symlink(os.path.basename(v[-1][0]), f"{d}/seed{s}.pt") if not os.path.exists(f"{d}/seed{s}.pt") else None
PYEOF
for s in 0 1 2; do
  ffmpeg -loglevel error -y -ss 3 -i $D/mp4/ltx_av_fast_1920x1088_seed$s.mp4 -frames:v 1 $D/stills/conv_seed${s}_t3s.png
  if scp -q "$R:$RO/seeds5/ltx_av_fast_1920x1088_seed$s.mp4" $D/mp4/dv_seed$s.mp4 2>/dev/null; then
    ffmpeg -loglevel error -y -ss 3 -i $D/mp4/dv_seed$s.mp4 -frames:v 1 $D/stills/dv_seed${s}_t3s.png
    echo "seed $s full video, device conv vs device DiffVAE:"
    ffmpeg -hide_banner -i $D/mp4/ltx_av_fast_1920x1088_seed$s.mp4 -i $D/mp4/dv_seed$s.mp4 \
      -lavfi "[0:v][1:v]ssim;[0:v][1:v]psnr" -f null - 2>&1 | grep -E "SSIM|PSNR" | sed 's/^.*\] //'
  fi
done
echo FETCH19_DONE

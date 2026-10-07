#!/bin/bash
# t211 post on blx01 (CPU): latent identity check between the arms, exact-frame stills (frame 72 = 3.0 s at 24 fps).
set -eo pipefail
F=/var/tmp/fasth3; T=$F/t211; R=$T/res
source $F/t48/python_env/bin/activate
python - <<PY | tee $T/latent_check.txt
import glob, hashlib, torch
for c in sorted(glob.glob("$R/conv/s2lat.gen*.pt")):
    d = c.replace("/conv/", "/diffvae/")
    a, b = torch.load(c), torch.load(d)
    same_v = torch.equal(a["video"], b["video"]); same_a = torch.equal(a["audio"], b["audio"])
    md5 = hashlib.md5(a["video"].numpy().tobytes()).hexdigest()
    print(f"{c.split('/')[-1]} video {tuple(a['video'].shape)} identical={same_v} audio identical={same_a} "
          f"max|dv|={(a['video']-b['video']).abs().max().item():.3e} conv_md5={md5} seed={a['flags'].get('seed')}/{b['flags'].get('seed')}")
PY
for a in conv diffvae; do
  for mp4 in $R/$a/ltx_av_fast_*.mp4; do
    ffmpeg -loglevel error -y -i $mp4 -vf "select=eq(n\,72)" -frames:v 1 ${mp4%.mp4}_f072.png
  done
done
ls -la $R/conv $R/diffvae

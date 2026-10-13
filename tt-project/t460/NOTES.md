# t460: FastH3 Turbo gains on one branch, A/B vs t397 on blx01

## Code (published)
origin/ttp/t460-integrate-fasth3-turbo-gains-on-one-bran at dbd6d66048f (local alias ttp/t460-turbo-integrated).
t397 head 8306d3e9c94 plus, cherry-picked with -x: 9acec520340 (t399 tpad fix), 5adb7443150 + bff7d82040c
(test seed/replays/save, 6 s), 5a8c59583a9 (opt-in strip-stitch trace), b0f45278d72 (stream temporal assembly,
from origin t423), 27a36185a2b + 0be01d61f87 (device bicubic 1080p upscale, keyed by canvas).
t422 12aff07c462 = same patch as 8306d3e9c94 (already in base). t410: no code. Left out: #407/#421, #453/#457.
Host tests: 27 passed by exact id (upscale env/target/blocks, trace_release, stitch host tests), clip-1.log run 1740.

## blx01 setup (all under /var/tmp/fasth3/t460)
- /var/tmp/fasth3/t48 (repo + python_env used by every blx01 job) was gone at 2026-10-13 04:34 UTC; no ledger entry.
  New venv: t460/python_env (create_venv.sh from t422/b, py3.10; only the final pre-commit step failed on the broken .git).
- base arm = t422/b (12aff07c462 = t397 + t399 fix + test knobs; built, warm JIT). int arm = same build, cwd and
  PYTHONPATH first = t460/src (models/ tests/ conftest.py pytest.ini from t422/b + int460.patch, Python only),
  MINIMAX_H3_UPSCALE=1920x1080. Import check: vae/upscale load from t460/src, ttnn from t422/b.
- run460.sh <tag> <base|int> <dur> <seed>; specs/ (12 jobs: b/i 6s s0 x2 replays, b/i 10s s0, b/i 6s seeds 1-4);
  every job SAVE=1 (latents + decoded video .pt in out_<tag>), TIMEOUT 600 (unmeasured 6/10 s arms), py 560.
- chain460.sh (detached on g15blx02, run 1740, log tt-project/state/runs/1740/chain460.log; state/chain.log here).
  First job: blx01 broker 077 (b6s0) submitted 04:42 UTC.

## Next step
When `ttp detach --check .../runs/1740/chain460.rc` exits 0: read state/chain.log (drops, tails) and
out_*/run.log timing tables on blx01. Report relative numbers (blx01 AICLK clamp 900 MHz).
Quality: base is 768p, int is device 1080p -> compare int video vs host bicubic(base video) per seed 0-4
(PSNR/PCC), plus latents bit-identity (base vs int latents.pt). Then VBench + stills on 5 seeds, video paths.
Run CPU work on blx01 under ionice -c3 nice, under /var/tmp/fasth3/t460 only.

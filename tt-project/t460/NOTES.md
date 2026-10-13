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

## Results (run 1745, 2026-10-12 22:45 PDT)
Chain 1740 rc 0: all 12 arms PASS on blx01, AICLK clamped at 900 MHz (UMD warning) -> relative numbers only.
Real broker jobs: b6s0 077, i6s0 078, b10s0 079, i10s0 080, b6s1 083, i6s1 085, b6s2 087, i6s2 089,
b6s3 091, i6s3 093, b6s4 095, i6s4 097. Even ids 084-098 were no-op resubmits (0.1 s, exited on PASS):
chain460.sh's wait-grep matched 'health' in a normal broker gate message after a successful queue.
No drops.

Timing (test's own table, last replay; base = t397 768p out, int = integrated, device 1080p out):
| run      | arm  | Total   | Denoise/step | VAE decode | VAE enc | Text  |
| 6s s0 r1 | base | 10.169  | 1.833 | 1.522 | 0.470 | 0.706 |
| 6s s0 r2 | base | 10.483  | 1.876 | 1.600 | 0.439 | 0.739 |
| 6s s0 r1 | int  | 11.895  | 1.876 | 2.844 | 0.634 | 0.723 |
| 6s s0 r2 | int  | 11.071  | 1.852 | 2.286 | 0.578 | 0.601 |
| 10s s0   | base | 15.355  | 3.003 | 1.920 | 0.565 | 0.696 |
| 10s s0   | int  | 14.982  | 2.978 | 2.107 | 0.130 | 0.694 |
| 6s s1-4 mean | base | 9.581 | 1.824 | 1.171 | 0.333 | 0.659 |
| 6s s1-4 mean | int  | 9.600 | 1.826 | 1.256 | 0.315 | 0.586 |
VAE decode with device 1080p: +0.085 s (6 s, s1-4) and +0.19 s (10 s) vs t397's 768p decode; t397 needs a host
upscale on top for 1080p (not in its total). i6s0 was the first int job (cold upscale programs: 407 s wall vs
300 s) and its decode stayed slow in replay 2 (2.29 s); the seed-0 base arm was also slower (1.52-1.60) than
s1-4 (1.17), so seed-0 6 s decode numbers are noisy. Denoise per step unchanged (same DiT path).
Totals vary +-0.4 s with VAE Encode/Text (host) jitter.

Quality (q460.py on blx01, CPU): latents bit-identical (max|d| 0) on all 6 pairs. int 1080p video vs host torch
bicubic of base 768p yuv: Y PSNR 55.23-55.33 dB, PCC 0.99995; YUV PSNR 54.43-54.66 dB, PCC 0.99992 (rounding).
Stills: tt-project/state/t460/still_<tag>_f79_base-bicubic_vs_int.png (left base+bicubic, right int): identical.
Videos: tt-project/state/t460/mp4/{base,int}-6s-s{0..4}.mp4, {base,int}-10s-s0.mp4.
VBench: vb460.sh (detached run 1745 vb460b), int vs base1080 (ffmpeg bicubic crf 8) -> state/t460/vbench.

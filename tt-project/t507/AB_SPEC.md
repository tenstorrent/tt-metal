# t507 device A/B: LTX25_S2_ANCESTRAL (default off) vs on

Code: branch `ttp/t507-ltx-2-5-opt-in-stage-2-ancestral-euler-t`, commit a22dc027ee2 (on top of
origin/ttp/t48-ltx25-integrated 1c5a2127400). Not on t48 or ltx-rt yet: push only after this A/B.

## What changes
`LTX25_S2_ANCESTRAL=1` makes stage 2 (trace_key `s2`) use ancestral Euler (eta 1, s_noise 1,
noise generator seed = seed + 20000, video then audio draws), as upstream
`ltx_pipelines.distilled` does. Stage 1 is unchanged (ancestral, seed + 10000). `s1_ref`/`s2_ref`
(reference-conditioned) stay plain Euler, as before. The env is read per denoise call, so one
process can run both arms. Default (unset or 0) is the old code path, bit-identical.

On device, the S2 Euler step then runs eager (the traced Euler tail is skipped when ancestral);
the DiT trace is untouched and the noise is uploaded per step, never baked into a trace.
Expected speed cost: two host randn + uploads and a few eltwise ops per S2 step (2 noisy steps).

## Job (one A/B pair, one broker job)
- Box: first of blx03 / blx01 / g15blx02 whose broker health check passes. blx03 via
  `tt-project/harness/templates/blx03-runner/blx03-enqueue.sh <spec>`; blx01/g15blx02 via
  `device-lint.sh` + `chain_job`. One project job per box.
- Mesh: full 4x8 (the standard LTX-2.5 e2e test opens it).
- Weights: local copies only (`LTX25_ROOT`, `LTX25_VIDEO_VAE` under /var/tmp/fasth3), never /mnt/MLPerf.
- Config: 1088x1920, 145 frames, SEED=0, NO_PROMPT=1 default prompt, untraced (LTX_TRACED=0)
  so each arm is one gen. Same process, same seed, same clock:
  arm A `LTX25_S2_ANCESTRAL=0` -> `/var/tmp/fasth3/t507/s2anc_off_seed0.mp4`,
  arm B `LTX25_S2_ANCESTRAL=1` -> `/var/tmp/fasth3/t507/s2anc_on_seed0.mp4`.
  The stock test runs one gen per process, so either (a) a small driver that builds the pipeline
  once and calls `generate` twice, flipping the env between calls, or (b) two pytest runs of
  `models/tt_dit/tests/models/ltx/test_pipeline_ltx25_distilled.py` inside the job script.
  Use (a) if (b) does not fit the timeout.
- Timeout: unmeasured, so 600 s. If both arms do not fit, split into one job per arm (same box,
  back to back), never a bigger number.
- Log check: arm B must log `ancestral Euler on-device (eta=1.0) noise_seed=20000` in stage 2
  (and `noise_seed=10000` in stage 1 in both arms); arm A must have no stage-2 ancestral line.
  Quote both arms' `Stage 2 denoise:` times.

## Quality (host, after the job; CPU only)
```
source ~/fasth3/tt-metal/python_env/bin/activate
python -m models.tt_dit.tests.models.ltx.tools.ltx_eval video \
  --ref s2anc_off_seed0.mp4 --cand s2anc_on_seed0.mp4 --out /var/tmp/fasth3/t507/q --vbench-ref
```
PCC/PSNR here measure how far the sampler change moves the output, not a regression: the arms use
different stochastic samplers, so the default gates (PCC 0.99, PSNR 30 dB) are expected to FAIL.
The verdict comes from VBench (cand vs ref, no dim down by > 0.01) and the first/middle/last
stills and cmp PNGs. If in doubt, run seeds 1-4 the same way (one A/B pair per job) and use
`ltx_eval batch`. Upstream reference sampling is the stated goal, so "on" wins if it is not
visibly worse; then push it to ltx-rt as a default-off opt-in, or propose default-on to the user.

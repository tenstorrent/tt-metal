# Wan VAE conv3d: testing perf, correctness, and video (old vs new config)

Notes-to-self for re-running the VAE conv3d evaluation. The "new config" =
conv3d HiFi2 (bf16) / HiFi4 (fp32) + the re-swept blockings baked into
`_BLOCKINGS`. It is the **default** on this branch — no env vars needed.

## What "old" vs "new" means now

The experiment env toggles were removed once the new config was adopted
(commit `72b8616aa78`). So switching configs is done via git, not env vars.

- **New config (default):** just run on the current branch (`HEAD`).
- **Old config (pre-change production):** restore the two config files from the
  baseline commit `eeda2ff7e25` (the commit just before this work), run, then
  restore:

  ```bash
  FILES="models/tt_dit/utils/conv3d.py models/tt_dit/models/vae/vae_wan2_1.py"
  git stash --                       # if you have local edits
  git checkout eeda2ff7e25 -- $FILES # -> OLD config (HiFi4 + old blockings)
  # ... run a test ...
  git checkout HEAD -- $FILES        # -> back to NEW config
  ```

  Only those two files determine the conv3d config (fidelity + blockings).

## Setup (every shell)

```bash
cd /path/to/tt-metal
source python_env/bin/activate
# Device-aware runner (flock + dispatch timeout + auto-reset). Prefer this over bare pytest.
#   scripts/run_safe_pytest.sh <test> [pytest args...]
```

## 1) VAE decode perf + correctness (end-to-end, 2x4)

`test_wan_decoder_production_blocking` builds the decoder, decodes a 480p latent,
compares against the diffusers torch golden (PCC + relative RMSE = correctness),
and logs `tt time taken` (perf). Uses whatever `_BLOCKINGS` + fidelity rule the
checked-out code has.

```bash
scripts/run_safe_pytest.sh \
  models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py::test_wan_decoder_production_blocking \
  -k "480p_t7_cached and 2x4_h0_w1" -s --timeout=0
```

- Pass/fail thresholds in the test: `MIN_PCC = 0.999`, `MAX_RMSE = 0.046`.
- For a clean A/B (perf + accuracy), run once on NEW (HEAD), once on OLD
  (checkout `eeda2ff7e25` as above). Same seed -> only the conv3d path differs.
- Note: this test times a single decode (includes JIT compile). For a cleaner
  device-bound number, warm up once then time again, or use the per-op sweep
  baseline below.

## 2) Per-op conv3d perf (sweep tooling)

Trace-timed, table-driven. Lives in `bruteforce_conv3d_sweep.py` (runs on a 1x1
submesh; reproduces any mesh's per-chip shape).

```bash
# Baseline: measure every _BLOCKINGS shape with its current blocking -> CSV
CONV3D_BASELINE_CSV=conv3d_baseline_perf.csv \
  scripts/run_safe_pytest.sh \
  models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py -k "collect_baseline" -s --timeout=0

# Sweep one shape (id from the table, e.g. h4w8_192x192_k333_T66_H94_W82)
scripts/run_safe_pytest.sh \
  models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
  -k "sweep_all and <shape-id>" -s --timeout=0

# Sweep everything
scripts/run_safe_pytest.sh \
  models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py -k "sweep_all" -s --timeout=0
```

Search-space knobs (env): `CONV3D_SWEEP_HW_PRODUCT` (default 32, `none` for full),
`CONV3D_SWEEP_MAX_T_BLOCK` (default 8), `CONV3D_SWEEP_MAX_COMBOS` (default 500),
`CONV3D_SWEEP_TRACE_ITERS` (default 10). Sweep fidelity is HiFi2 (matches the
model default); see `MATH_FIDELITY` in the file.

To bake new winners into `_BLOCKINGS`: re-sweep -> read `best_*` columns from the
result CSV -> edit the table values (this is what the adoption commit did).

## 3) Full pipeline -> 480p video (2x4, 2 links)

Generates `wan_t2v_832x480_0.mp4` in the CWD. `NO_PROMPT=1` is required (else it
prompts on stdin and hangs). Fixed prompt + seed=42, so denoising is identical
run-to-run; only the VAE decode differs between configs.

```bash
NO_PROMPT=1 scripts/run_safe_pytest.sh \
  models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py::test_pipeline_inference \
  -k "bh_2x4sp1tp0 and resolution_480p" -s --timeout=0
# rename so old/new don't overwrite each other:
mv wan_t2v_832x480_0.mp4 wan_t2v_480p_<old|new>.mp4
```

- Weights are cached at `~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers`.
- Video export needs `imageio_ffmpeg` in the venv (bundles ffmpeg; no system
  ffmpeg). If missing, the test silently skips export — install with:
  `/usr/bin/pip install --target python_env/lib/python3.10/site-packages imageio_ffmpeg`
- ~40 denoising steps, ~4-5 min/run after compile. The test also runs a CLIP
  gate (mean >= 36); the mp4 is written before that assert, so it exists even if
  CLIP fails.

## Old-vs-new A/B recipe (any of the above)

```bash
FILES="models/tt_dit/utils/conv3d.py models/tt_dit/models/vae/vae_wan2_1.py"
# NEW:
<run test>                                  # on HEAD
# OLD:
git checkout eeda2ff7e25 -- $FILES
<run test>
git checkout HEAD -- $FILES                 # restore
```

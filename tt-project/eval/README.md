# FastH3 quality eval

CPU-only checks for fl2va runs. Nothing here touches the TT device except `gen_seeds.sh`, which
queues the generation on the broker.

Run layout (from `models/tt_dit/tests/models/minimax_h3/test_fasth3_baseline_minimax_h3.py`), one
dir per config: `seedN.mp4`, `seedN.wav`, `seedN_latents.pt`, `seedN_frames_u8_every16.npy`
(lossless), `seedN_{first,mid,last}.png`, `seedN_timings.json`.

## Setup (once)

    tt-project/eval/setup_env.sh        # CPU venv at /home/smarton/fasth3/tt-metal/.venv-eval
    PY=/home/smarton/fasth3/tt-metal/.venv-eval/bin/python

## 1. Generate 5 seeds (device, via broker)

    tt-project/eval/gen_seeds.sh tt-project/baselines/dense BASE_SECONDS=10 BASE_HEIGHT=1088 BASE_WIDTH=1920
    tt-project/eval/gen_seeds.sh tt-project/runs/vsa09     BASE_SECONDS=10 BASE_HEIGHT=1088 BASE_WIDTH=1920 BASE_VSA_SPARSITY=0.9

Seeds 0-4 by default (`BASE_SEEDS`), keyframes from `tt-project/baselines/keyframes`, code from the
main checkout (`REPO=` for a built worktree), `DRY_RUN=1` prints the command. Artifacts land in
`OUT/<tag>/`, e.g. `OUT/fl2va_1088p_10s_dense_50steps/`.

## 2. One-shot check of a candidate

    tt-project/eval/evaluate.sh REF_RUN_DIR CAND_RUN_DIR [OUT_DIR]   # FULL=1, SEEDS=0,1

Runs 3 → 4 → 5 below and prints a VBench delta table. Scores are cached per run dir
(`vbench_partial.json`), so the reference is scored once.

## 3. PCC / PSNR (first pass)

    $PY tt-project/eval/compare.py REF_RUN_DIR CAND_RUN_DIR [--seeds 0,1] [--mp4] [--json out.json] [--strict]
    $PY tt-project/eval/compare.py ref.mp4 cand.mp4          # or .npy / .pt / .wav pairs

Per seed: latent PCC (`video_rows`, `audio_rows`), pixel PCC/PSNR on the lossless every-16th-frame
npy (all mp4 frames with `--mp4`, or when no npy exists), worst-frame PSNR, audio PCC/SNR, and the
wall-clock speedup from the timings json. Defaults: pixel PCC ≥ 0.99, PSNR ≥ 30 dB, latent PCC ≥ 0.99;
`--strict` exits 1 on a miss. A miss is a prompt to look, not a verdict: sparse attention or a
different reduction order moves pixels without making the video worse. Use 4 and 5 on 5 seeds.

## 4. VBench (partial by default, `--full` optional)

    $PY tt-project/eval/vbench_eval.py RUN_DIR_OR_MP4... [--full | --dims a,b] [--ref-json old.json] [--json out.json]

Partial: subject_consistency, motion_smoothness, aesthetic_quality, imaging_quality. Full adds
background_consistency, temporal_flickering, dynamic_degree, overall_consistency (prompt read from
the timings json, or `--prompt`). VBench's other dims need its own prompt suite and do not apply.

Speed settings, on by default: videos re-encoded to 512 px short side (`--short-side 0` = native),
aesthetic/imaging score every 4th frame (`--frame-stride 1` = all), DINO batched (same score).
`--stock` = unmodified VBench code. Only compare scores taken at the same settings; the summary json
records them and `--ref-json` warns on a mismatch. Summary `dims` are VBench's normalized means
(imaging /100); `per_video` holds raw per-video scores (imaging on 0-100).

## 5. Visual review

    $PY tt-project/eval/review.py REF_RUN_DIR CAND_RUN_DIR [MORE...] --out OUT/prefix [--names a,b] [--seeds 0,1,2,3,4]

Writes `prefix_sheet.png` (one row per seed × run, 6 evenly spaced frames, PSNR vs the first run in
the row label) and `prefix_grid.mp4` (runs side by side, one row per seed, 640 px tiles, full length).

## Runtime per check (g15blx02 CPU, 1080p 10 s = 241 frames, shared box under load)

TIMINGS

## Self-test

    cd tt-project/eval && $PY -m pytest -q -p no:cacheprovider -c /dev/null --confcutdir . test_eval.py

`make_fixture.py OUT` builds synthetic baseline-shaped ref/cand dirs (zoom-pan over a still, cand =
ref + known noise) for trying the scripts without a device run.

## Disk

venv ~1.8 GB (`.venv-eval`, git-excluded). VBench weights ~3.5 GB in `~/.cache/vbench`,
`~/.cache/clip`, `~/.cache/torch/hub` (some were already there). Staged low-res copies go to
`/tmp/fasth3_vbench` (override with `--out`); delete when done.

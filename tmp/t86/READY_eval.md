# READY_eval (#86): 5-seed 4x8 eval pack for the non-exact candidates

Status: ready to launch. NOT launched. It needs full 4x8 (32-chip) runs, which the user has barred
on both boxes. Launch only after the user allows 4x8 runs. Box: blx03 (g14blx03) or g15blx02,
through that box's own tt-device-mcp broker, one project job at a time.

Branch: `ttp/t95-eval-pack-launchable` (t48 7606e48a8bf + LTX_VAE_EXACT_SHARD commit d22cef25303 + this pack). expapprox (rejected, #81) and vaetrace (rejected, #87) were dropped; the LTX_VIDEO_VAE_TRACE code is not on t48.

## What it settles

| config | env | question |
|---|---|---|
| baseline | (t48 defaults; x264 ultrafast/crf20 export) | reference timings + conv-decoder clips for A/B |
| gate | LTX_FUSE_GATE_ON_DEVICE=1 | -0.2 s est.; non-exact vs LTX_FUSE_GATE=0 |
| adaln | LTX_FUSE_NORM_ADALN=1 | -1.9 %/block |
| export_veryfast | LTX_EXPORT_PRESET=veryfast | old export (veryfast/crf23): settles the ultrafast default |
| lofi | LTX_VAE_CONV_FIDELITY=LoFi | LoFi up-block conv3d: -177 ms decode on 2x4 (job 354), PSNR min 45 dB; needs 5-seed VBench + visual before default |
| exact_shard | LTX_VAE_EXACT_SHARD=1 | exact-shard rebalance after upsample: -51.9 ms decode, bit-identical on 2x4 (job 454); 4x8 1080p split unchecked on device; expect PCC 1.0 vs baseline |
| all | gate + adaln + lofi | combined |

t48 7606e48a8bf has the LTX_VAE_CONV_FIDELITY commit the lofi arm needs.

The production conv VAE decode runs eager; tracing it gave no speedup (#87), so it is not in the pack.

Per config, one pytest run (`-k bh_4x8sp1tp0_ring`) does:
- gen#0 (capture)
- gen#1 plus 2 extra replays on fresh prompts (encoder on the path)
- 5 seed replays (seeds 0-4) on the default prompt → `ltx_av_fast_1920x1088_seed<N>.mp4`

## Setup (once per box; needs its own build, neighbor_pad is C++)

    ssh g14blx03 'BR=ttp/t95-eval-pack-launchable W=/home/smarton/fasth3/t95 bash -s' < tmp/t60/blx03_setup60.sh
    # ready when the setup log shows: SETUP60_DONE rc=0

Weights and caches: the existing /var/tmp/fasth3 LTX-2.5 / Gemma caches are reused. No new caches.

## Launch (only once 4x8 is allowed)

Dry run first (no broker, writes to /tmp/eval86_dry):

    ssh g14blx03 'DRY_RUN=1 bash /home/smarton/fasth3/t95/tmp/t86/eval_pack.sh'

Real run, detached on blx03:

    ssh g14blx03 "mkdir -p /var/tmp/fasth3/eval86; ALLOW_4X8=1 setsid nohup bash /home/smarton/fasth3/t95/tmp/t86/eval_pack.sh > /var/tmp/fasth3/eval86/pack.log 2>&1 &"

- `ONLY=baseline,lofi` runs a subset.
- Configs whose run.log already has `RUN_EXIT=0` are skipped, so a relaunch resumes.
- Each job goes through `tmp/blx03/submit.sh` (exit 75 = our other job active, retried every 120 s), with JOB_TIMEOUT=2700.
- The pack stops at the first failed config.
- Job IDs: `/var/tmp/fasth3/eval86/jobs.txt`.

Done check (for retry_when):

    ssh g14blx03 test -e /var/tmp/fasth3/eval86/PACK_DONE    # content: ok | failed:<cfg> job=<id> | refused

Stop procedure: if a drop starts during one of our jobs, run `tt_device_job_kill` on our queued
or running jobs on every box, kill the pack loop (`pgrep -af eval_pack.sh`, then kill that PID)
and report it.

## Evaluate (on g15blx02, off-device)

    rsync -a g14blx03:/var/tmp/fasth3/eval86/ ~/fasth3/out/eval86/
    python3 tmp/t86/post_eval.py pack ~/fasth3/out/eval86

Per config this:
- links `clips/seed<N>.mp4`
- runs `ltx_eval batch` against `tt-project/baselines/ltx25_1080p_6s/ref_dv145` (partial VBench, stills f000/f072/f144) into `eval_ref/`
- runs the same with `--vbench none` against `baseline/clips` into `eval_ab/`
- writes `timing.json`

Then it writes `pack_summary.json` and `pack_summary.md`, one row per config:
- e2e, encode, S1, upsample, S2, VAE (+ decode/output split), audio and export medians (fresh gens and seed gens)
- load time, DRAM peak and min contiguous free
- PCC/PSNR vs ref and vs baseline
- VBench means, with FAIL when a dim drops more than 0.01 below the ref

`--no-eval` gives timings only.

## Decision rules

- Exact candidates must show PCC 1.0 vs baseline. Any non-1.0 is expected for gate/adaln/lofi.
- Non-exact wins pass if VBench stays within 0.01 of the ref on every dim, and PSNR vs baseline shows no outlier seed.
- When in doubt, look at the 5-seed stills in `eval_ab/` (charter rule: metric misses alone are not proof).
- The ref was decoded with DiffVAE and the pack uses the conv decoder, so judge PCC/PSNR mainly vs `baseline`, not vs the ref.
- export_veryfast vs baseline settles the ultrafast default: if VBench/PSNR are equal, ultrafast stays.

## Runtime and disk

Device:
- load ~12 min
- gens: gen#0 ~50 s, plus 8 replays at ~10-20 s each
- ~15-18 min per config, 6 configs ≈ 1.5-1.8 h of 4x8 device time (6 separate jobs)

Eval on g15blx02: VBench ~3 min/clip, about 5-10 min per config, ~45 min total.

Disk:
- ~90-130 MB of mp4 plus logs per config, under 1 GB in /var/tmp/fasth3/eval86 on blx03
- the same again in ~/fasth3/out/eval86 on g15blx02, plus stills (~50 MB)

Risks:
- DRAM headroom with the gate/adaln folds: DIFFVAE_MEM_LOG=1 logs it.

## Cleanup

- blx03: `rm -rf /var/tmp/fasth3/eval86 /tmp/eval86_dry` once the summary is copied back.
- The /home/smarton/fasth3/t95 worktree+build on blx03: remove only after the branch is pushed and nothing else needs it.
- g15blx02: keep pack_summary.* and stills; drop the clips once a decision is recorded.

## Files

- `tmp/t86/configs.txt`: config label + env
- `tmp/t86/run_eval.sh`: one config (run48.sh + seeds/fresh prompts/stage timers); DRY_RUN=1 prints only
- `tmp/t86/eval_pack.sh`: detached submit/poll loop on blx03; ALLOW_4X8=1 required
- `tmp/t86/post_eval.py`: `parse <run.log>` and `pack <root>`

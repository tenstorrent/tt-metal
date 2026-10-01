# READY (blx03): t48 integrated LTX-2.5 e2e, 1080p/145f, seed 0, 3 timed gens

STATUS: PREPARED, NOT RUN. This job opens the full 4x8 mesh (32 chips, `bh_4x8sp1tp0_ring`). The user's
2026-09-30 22:10 rule bars full-mesh runs on blx03 (only 1 tray / 2x4 submesh, one at a time), and
g14blx03-device is paused since the 2026-10-01 02:10 reboot. Submit only after the user allows full-mesh runs again.
The mesh is opened whole by the test fixture; this job never opens a bare (2,4) mesh. Any submesh variant must
open the full mesh and use `create_submesh`.

## What is on the branch

Branch `ttp/t48-ltx25-integrated` (same commit as `ttp/t48-integrate-all-ltx-2-5-wins-on-one-branch`),
based on t36 `16ba9a383dc` (LTX-2.5 port + blx03 setup), with:

| win | commit | effect on gen #1 (t20 job 879 baseline) |
|-----|--------|------------------------------------------|
| t20 conv decoder default | 0533827a419 | VAE 11.77 s (DiffVAE) -> 0.72 s; already in the 8.76 s baseline |
| t40 Gemma encode trace captured after gen #0 | 9e336c44b71 | encoder 1.91 s -> ~0.2-0.3 s for a new prompt |
| t13 yuv420p/x264 export: ultrafast crf 20, zero-copy frames, AAC beside video | eee3baf7c0d | host bench: video encode 0.65 -> 0.15 s, export 0.35-0.40 -> 0.19 s |
| t18 video encode on a worker under the device audio decode | 63902277007 | export 0.9 -> 0.3 s on g15blx02 (job 610, mp4 byte-identical) |
| t44/t66 `LTX_VAE_FOLD_TIME_PAD` (default on; `=0` turns it off) | 1968790b040 | -44 ms per conv decode at 1080p/145f, bit-identical (#46, blx03 2x4 jobs 000/008) |
| t58 fused YUV unpatch permute drops the p=1 axis | b9312ff3061 | output path 208.7 -> 38.5 ms per decode (about -170 ms), bit-identical (#62, blx03 2x4 job 030) |
| t60/t72 `LTX_VAE_FOLD_W_MASK` (default on; `=0` turns it off): neighbor_pad_async `logical_w` zeros the W pad columns | e21bcbad8d9 (C++) | -69 ms per conv decode at 1080p/145f, bit-identical (#65, blx03 2x4 jobs 033/034) |
| t8 ltx_eval harness | t8 tip | eval tooling only |

Conflicts resolved: t13 and t40 both moved the Gemma trace capture; t40's `capture_trace()` (guarded by
`_trace_captured`) is kept. t13 and t18 both rewrote the yuv export; the merged `YuvVideoExport` keeps the worker
thread (t18), the zero-copy frame wrap (t13), and encodes AAC in `finish()` while the video worker may still run (t13).

t60's neighbor_pad_async `logical_w` is a C++ change (op, kernels, nanobind), so t48 no longer runs on blx03's t36
build: the t48 worktree needs its own Release build, and `run48.sh` sets `TT_METAL_HOME` to it. The Python passes
`logical_w` to neighbor_pad even with the fold off, so an old build fails at the first conv either way.

## One-time setup on blx03 (CPU only, ~2.5 GB worktree + Release build, a few min)

```bash
ssh g14blx03 'BR=ttp/t48-ltx25-integrated W=/home/smarton/fasth3/t48 bash -s' < tmp/t60/blx03_setup60.sh
```

Ready when `ssh g14blx03 tail -1 ~/fasth3/t48-setup.log` prints `SETUP60_DONE rc=0`. New kernels JIT-compile into
`$FASTH3_DATA/cache/tt-metal-cache` (/var/tmp) on the first run.

If blx03 cannot fetch from GitHub: on g15blx02
`git -C ~/fasth3/tt-metal bundle create /tmp/t48.bundle 16ba9a383dc..ttp/t48-ltx25-integrated && scp /tmp/t48.bundle g14blx03:/tmp/`,
then on blx03 `git fetch /tmp/t48.bundle ttp/t48-ltx25-integrated` and run the setup script with `bash -s -- FETCH_HEAD`.

## The job (one broker job, ~5-8 min warm; budget 1800 s)

```bash
ssh g14blx03 "~/fasth3/tt-metal/tmp/blx03/submit.sh 1800 bash /home/smarton/fasth3/t48/tmp/blx03/run48.sh t48_e2e"
```

`run48.sh`: 4x8 ring, 1080p (1920x1088) / 145 frames, `SEED=0`, conv decoder, traced. Gens: #0 (warm-up and trace
capture, plus the Gemma encode capture at its end; not a timing number), then #1, #2, #3 timed, each on a new prompt
(`LTX_FRESH_PROMPTS=1`, so the text encoder is on the measured path). `LTX_TIME_STAGES=1` logs per-stage times.
Output: `blx03:~/fasth3/out/t48/t48_e2e/{run.log, ltx_av_fast_1920x1088_{0,1,2,3}.mp4}`.

Optional second job, only after the first passes (fold A/B with both folds off, same everything else):
`ssh g14blx03 "~/fasth3/tt-metal/tmp/blx03/submit.sh 1800 bash /home/smarton/fasth3/t48/tmp/blx03/run48.sh t48_nofold LTX_VAE_FOLD_TIME_PAD=0 LTX_VAE_FOLD_W_MASK=0"`.
Expect bit-identical frames: `cmp` the two runs' mp4s for gens 1-3; any difference means a fold is not exact on device.

Check: `ssh g14blx03 tt-device-mcp status -j <id>`; pass = run.log has ` passed` and `RUN_EXIT[t48_e2e]=0`.
Stop all device work and report at the first chip drop (`ssh g14blx03 tt-smi -ls` count, or a fabric/PCIe error in the log).

## Timings to read

```bash
ssh g14blx03 "grep -iE 'E2E_WALL_S|encoder|stage [12] denoise|latent upsample|vae decode|audio decode|video export|total \(compute\)' ~/fasth3/out/t48/t48_e2e/run.log"
```

Expected gen #1-#3 (vs t20 job 879 gen #1 at E2E 8.76 s):

| stage | t20 (s) | expected t48 (s) |
|-------|---------|------------------|
| text encode (new prompt) | 1.91 | 0.2-0.3 |
| S1 denoise | 2.28 | 2.28 |
| upsample | 0.15 | 0.15 |
| S2 denoise | 2.50 | 2.50 |
| VAE decode (conv) | 0.72 | 0.72 (~0.6 with both folds: -44 ms T pad, -69 ms W mask) |
| audio decode | 0.37 | 0.37 (video encode runs under it) |
| export | 0.8 | 0.1-0.2 |
| E2E_WALL_S | 8.76 | ~6.4-6.7 |

The goal is < 7 s for a 1080p 6 s clip; this run checks whether the integrated branch gets there.

## Quality check (on g15blx02, no device)

Gen #1 uses the same prompt ("red paper boat", first fresh prompt) and seed as t20's gen #1, and the denoise and
decode code is unchanged, so only the x264 settings differ (veryfast crf 23 -> ultrafast crf 20):

```bash
scp g14blx03:fasth3/out/t48/t48_e2e/ltx_av_fast_1920x1088_1.mp4 /tmp/t48_gen1.mp4
cd ~/fasth3/tt-metal/tt-project/worktrees/t48 && source ~/fasth3/tt-metal/python_env/bin/activate
python -m models.tt_dit.tests.models.ltx.tools.ltx_eval video \
  --ref ~/fasth3/tt-metal/tt-project/baselines/t20/ltx_av_fast_1920x1088_1.mp4 --cand /tmp/t48_gen1.mp4 \
  --out /tmp/q48 --vbench-ref
```

Expect PSNR >= 40 dB and PCC >= 0.99. Look at the stills in /tmp/q48 and show the user a still
(`ffmpeg -ss 3 -i /tmp/t48_gen1.mp4 -frames:v 1 /tmp/t48_gen1_t3s.png`). Clean up /tmp/q48 afterwards.

## Cleanup after the runs

Copy the mp4s and run.log to `tt-project/baselines/t48/` on g15blx02, then on blx03:
`rm -rf ~/fasth3/out/t48 ~/fasth3/t48-setup.* && git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/t48`.

## Known latency: first request pays the encode-trace capture

The Gemma encode trace is captured once, at the end of the first request (gen #0), after its mp4 is written
(`pipeline_ltx_distilled.py`, `open_trace_gate()` + `capture_trace()`). That adds ~1.6 s (4x8) to gen #0's
call, not to its output time. Every later request replays the trace. Time gen #1+ for the steady-state number.

## Export preset

Default x264 export is `ultrafast` / crf 20. `LTX_EXPORT_PRESET=veryfast` restores the old veryfast / crf 23;
`LTX_EXPORT_CRF` overrides the crf of either preset.

## Before a PR: remove these force-tracked files

Keep `tmp/READY_48.md` and `tmp/blx03/{READY_blx03.md,env.yaml,run48.sh,submit.sh,NOTES.md}` until the t48 e2e run
is done, then remove them too. Remove now-unneeded files before any PR:

- `NOTES.md` (root)
- `tmp/READY_22.md`, `tmp/READY_26.md`
- `tmp/block_env.yaml`, `tmp/blx03/run25.sh`
- `tmp/chunk_ab.sh`, `tmp/cmp.sh`, `tmp/cmp_blk.py`, `tmp/done.sh`, `tmp/drive6.sh`, `tmp/e2e.sh`
- `tmp/na_kernel_cmds.txt`, `tmp/na_kernel_compile.sh`

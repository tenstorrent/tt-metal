# t288: AdaLN precomputed vs on device, FastH3 HyperFlow 8-step, 4x8 blx03

## 2026-10-10 ~06:10 UTC (run 1292)
- Code: ttp/t288-adaln-ab = a99af1c7dea + cherry-pick acebc7d39e2 (test import fix, cc3b6de2dc7)
  + f7266236d90 (MINIMAX_H3_ADALN_PRECOMPUTE=0 keeps AdaLN on device under HyperFlow; the turbo
  test skips the two-time embedder check under the table). Python only.
- `ttp push --own` refused: its push check runs LTX tests (test_vae_ltx_*_ref.py) that do not
  exist on this H3 branch. Branch not on origin yet. Applied on blx03 as patches instead:
  ~/fasth3/t286 detached at 03848fa4df (same tree as f7266236d90; build from #286 reused).
- HyperFlow adapter: HF videorebirth/hyperflow -> blx03 /var/tmp/fasth3/models/hyperflow
  (2.8 GB, sha256 matches hyperflow.json, READ_OK). Header has hyperflow sigmas (9 pts),
  gate 0.25, shifts 12/3.
- Driver /var/tmp/fasth3/t288/run288.sh <tag> <on|off> (copy here). fl2va, 5 s, 1344x768,
  NFE 8, seed 0, kf /var/tmp/fasth3/t209/kf_first.png, caches dit-h3hf + tt-metal-cache-h3hf
  (shared with #286). Off arm reuses transformer_resident_adaln cache; on arm writes a new
  transformer_precomputed_adaln cache (cold: expect on-a to stop at the warm deadline, on-b warm).
- Queued (FIFO): t288-off-a, t288-on-a, t288-on-b, t288-on-c, t288-off-b. ONCE guard per arm
  (out_on/PASS, out_off/PASS), so extra jobs exit at once after a pass. off-b re-runs only if
  off-a failed.
- AICLK on blx03 is clamped at 900 MHz: numbers are relative only.

## Next step
Wait on probe.sh t288-off-b. Then: done markers of all five; out_{on,off}/PASS; grep
"MINIMAX-H3 PERFORMANCE RESULTS" table from /var/tmp/fasth3/t288/out_{on,off}/run.log;
PSNR/PCC of out_on vs out_off mp4 (ffmpeg psnr + numpy on decoded frames); still frames;
copy small artifacts to tt-project/t288/. Cleanup: dit-h3hf/minimax-h3/transformer_precomputed_adaln
if not kept, hyperflow model if no follow-up needs it.

## 2026-10-10 05:10 PDT (run 1359): blx03 runs all failed, blx03 down, move refused
- blx03 serial runner (900 MHz clamp), every arm failed on the #286 warm deadline (400 s), none
  reached denoise, so there are no timings or videos:
  - t288-off-a broker 853: 395.9 s call, "warm deadline 400 s passed in Initializing bucket buffers".
  - t288-on-a broker 855: 399.1 s call, "warm deadline 400 s passed in Warming prompt encoder keyframe layouts".
  - t288-on-b broker 857: 411.9 s call, same bucket-buffer deadline.
  - t288-on-c broker 860: DROP 06:34:37 UTC (status=killed, chips 12 and 8, trays 1-4), then
    broker-kill 06:47:43 UTC (chips unknown); T288_EXIT=137. Our job; blx03 later went out of service.
- Removed our leftovers from the blx03 runner: queue/...t288-off-b.job and running/t288-on-c.{job,state}
  moved to /var/tmp/fasth3/runner/parked/t288-moved-blx01/ (nothing else touched).
- blx01 refused (coordinator 05:02 PDT, #330): MiniMax-H3 is 135 GB (blx03), dit-h3hf cache 152 GB;
  blx01 footprint already ~140 of 150 GB. No weights copied.
- Branch ttp/t288-adaln-ab pushed to origin at f7266236d90 (ttp push --own passed this time).

## Next step (when blx03 is back: tt-project/state/ready/g14blx03.READY)
The cold test does not fit 600 s at the 900 MHz clamp: warmup alone passes 400 s. Before requeueing,
cut warm work so one arm finishes in ~400 s: e.g. init only the 1344x768/5 s bucket buffer instead of
all 26 buckets, and skip keyframe-layout warmups the fl2va 5 s run does not use (or run at full AICLK).
Then requeue off then on (on needs its precomputed-AdaLN cache: a cache-only job first, then the timed run).

## 2026-10-10 05:10 PDT (run 1370): still blocked on blx03
- g14blx03.READY is stale (04:36 UTC, before the outage). blx03 broker at 12:05 UTC: "device HELD
  (degraded): device is dirty and unverified: gate/post-job: 8/32 chip", health gates 884/886/888/892
  failed, bridge-reset chips 8-15 failed. Wait probe: tt-project/t288/probe_blx03.sh (exits 0 once the
  broker is active and not holding the device).
- No device work, no copies to blx01 (coordinator 05:02 PDT). Branch unchanged at f7266236d90 on origin.
- Before requeueing, the warm-only-the-used-bucket flags (coordinator decision 05:04 PDT) must land on
  ttp/t288-adaln-ab; the cold test does not fit 600 s at the 900 MHz clamp otherwise.

## 2026-10-10 05:15 PDT (#350, run 1369): warm-scope flags, three short jobs (CPU only, no device)
- Code: ttp/t288-adaln-ab 22fcacb6b19 (on f7266236d90). Patch for blx03:
  tt-project/t288/patches/0001-minimax-h3-add-opt-in-warm-scope-for-single-request-.patch.
  Pipeline kwargs `warm_rungs`, `warm_canvases` (default None = full envelope, as before). Turbo test flags:
  - `MINIMAX_H3_TURBO_WARM_RUNGS=41984`: bind only that rung plus the top rung (2 of 26).
  - `MINIMAX_H3_TURBO_WARM_USED_LAYOUTS=1`: warm only the keyframe layouts of the run's canvas and keyframe
    (kf_first.png is 1344x768, the same canvas): 3 of 28 layouts (empty, 1 kf, 2 kf).
  - `MINIMAX_H3_TURBO_CACHE_ONLY=1`: build without warmup (writes the weight cache), build/load the
    precomputed AdaLN table, then stop: no generation.
  Output does not change: on 4x8 nothing is traced, and a rung outside the warm set is only logged
  ("outside warm_rungs") and compiled by the untimed priming call. CPU checks: py_compile; the new
  served_keyframe_layouts equals the old one at alignments 32-1024; 61 device-free tests pass
  (packing, scheduler, conditioning); the turbo test collects with the flags set; pre-commit clean.
- run288.sh: `<tag> <on|off> [cache]`; sets the flags above; refuses code without them (grep), weights
  under /mnt, on a non-local fs (findmnt) or behind symlinks; keeps setsid + EXIT/TERM/INT group kill.
  lint.sh --device passes. Old specs off-a/off-b/on-a/on-b/on-c removed; new specs off, on-cache, on.
- Per-job wall at the 900 MHz clamp (estimates; -t = estimate +50%, at most 600):
  - t288-off: about 300 s. Basis: job 850 (t286, same canvas and rung, 4 rungs, 3 layouts, VAE warm
    skipped): warmup done at 248 s, process wall 283 s. Here: 2 rungs instead of 4 (each 8-23 s: -16 to
    -40 s); 8 forwards instead of 4 in the priming and timed calls (2 x 4 x 1.47 s/step: +12 s).
    -t 450; T286_WARM_DEADLINE_S=360, T286_WARM_LATE_S=360 (warmup must end by 360 s so the two
    generations, about 50 s, plus teardown fit). Job 853 (full warm) reached the bucket walk at about
    325 s, so the cut is the 230 s layout walk -> about 35 s, and 26 rungs -> 2.
  - t288-on-cache: about 90 s warm. Both caches already exist on blx03 (transformer_precomputed_adaln,
    06:11 UTC; minimax-h3-adaln/97be3f17....adaln.pt, 06:24 UTC); jobs 857/860 loaded that cache in 13 s.
    Cold: transformer cache write 180 s or less (855), plus the host table build, which has not been
    measured. Unmeasured, so -t 600.
  - t288-on: about 300 s, like off (cache load 13 s against about 10 s). -t 450, same deadlines.
- If a job hits the 360 s deadline: the layouts or rungs were not cut (check "[t288] cmd" and the env
  dump in run.log for MINIMAX_H3_TURBO_WARM_*), or the box is slower than 850's run. Do not raise -t.

## Next step (#288, when blx03 is back: tt-project/state/ready/g14blx03.READY fresh, probe_blx03.sh exits 0)
1. On blx03 (setup, not a device job): `git -C ~/fasth3/t286 apply` the patch above (copy it to
   /var/tmp/fasth3/t288/patches/ first), check `git -C ~/fasth3/t286 diff --stat` shows the 3 files.
   Copy run288.sh to /var/tmp/fasth3/t288/run288.sh.new, then `mv` it into place.
2. Enqueue in this order, one at a time through the serial runner (blx03-enqueue.sh <spec>):
   specs/off.txt (t288-off), specs/on-cache.txt (t288-on-cache), specs/on.txt (t288-on). Wait on t288-on.
3. Results: out_off/run.log and out_on/run.log "MINIMAX-H3 PERFORMANCE RESULTS" tables, plus
   "outside warm_rungs" lines (there should be none); PSNR/PCC of out_on vs out_off mp4; still frames.

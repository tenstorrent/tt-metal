# t220: LTX-2.3 8-bit e2e, BH 4x8, 1088x1920 145f, 5 seeds (user #148/#151)

8-bit in production = ltx_server tiers medium/fast (`LTX_QUALITY`, expanded by ltx-rt
`models/tt_dit/utils/ltx.py:apply_quality_env`): `LTX_QUANT=all_bf8_lofi` (bf8 DiT linear weights +
activations, LoFi). medium = 6 S1 + 1 S2 steps (the LTX_FAST bundle); fast = 3 S1 + 1 S2. high = bf16, 8+3.
Measuring medium (scene-preserving 8-bit tier). Galaxy worker env for bf8 tiers adds LTX_ATTN_FABRIC_AGMM=0.

Tree: blx03 ~/fasth3/t220 = ltx-rt b9f8587ce6c (= live tt-metal-ltx-rt-b HEAD) + 4843ab693a (test: LTX_E2E_SEEDS,
E2E_WALL_S log), branch ttp/t220-ltx23-8bit (local on blx03). Release build there (setup.sh).
Caches: /var/tmp/fasth3/cache/dit-ltx23 (copy of /home/sulphur/tt_dit_cache gemma/ltx-2.3/upscaler; the bf8 DiT
is filled by our fill job), JIT /var/tmp/fasth3/cache/tt-metal-cache-ltx23. Checkpoint/Gemma read from /home/sulphur/hf.
#217's blx01 driver (pid 3599339) killed before it submitted anything.

Jobs (blx03 serial runner): t220-fill-r1 (seed 0 capture+replay, fills bf8 cache + JIT), then t220-time-r1
(seeds 0-4: gen0 capture, gen1 seed0, gen2-5 seeds 1-4; out /var/tmp/fasth3/t220/out_time).
Tray-2 incident 2026-10-07 19:49 UTC (bridge-reset chips 8-15, broker job 417/418) -> enqueue after 20:20 UTC.

## 2026-10-07 20:25 UTC (attempt 1, standard wake)
- Setup copy rc=21: the live service's cache has root-owned unreadable subdirs (bf16/bf8 transformer, VAE,
  connectors, feature_extractor, upscaler). Copied OK: Gemma (44 GB, size matches) and the readable part of
  ltx-2.3-22b-distilled-1.1 (1.4 GB). Removed the empty dirs cp left behind so they read as cache misses.
  => the fill job also builds bf8 DiT + VAE + connectors + upscaler from /home/sulphur/hf checkpoints.
  Caches are per module, so if t220-fill-r1 hits the 600 s cap, a t220-fill-r2 resumes; then time-r1.
- Both boxes held at 20:21 UTC: blx03 tray 2 (chips 8-15) bridge-reset loop since 19:48 UTC (broker jobs 415-428,
  ltx-host job 415 killed by chips leaving PCIe); blx01 chip 11 off the bus (broker 844-846, after our-project
  t219 job 843). Not our jobs.
- Wait: `bash tt-project/t220/probe_ready.sh` (0 = blx03 or blx01 clear of holds/resets for 30 min).
- Next: blx03 clear -> `tt-project/harness/templates/blx03-runner/blx03-enqueue.sh tt-project/t220/spec-fill.txt`.
  Only blx01 clear -> needs a /var/tmp/fasth3/t220 build + cache there first (not done).

## 2026-10-07 21:50 UTC (attempt 1, standard run 2)
- blx03 clear since its 21:07 UTC power-cycle (broker 450-453); ltx-host 454 ran clean. probe_ready=0.
- Enqueued spec-fill.txt then spec-time.txt on the blx03 runner. First submit failed (no ENV, broker looked for
  python_env under WORKDIR); added ENV=~/fasth3/runner/env.yaml (run220.sh overrides its TT_METAL_HOME/cache).
- t220-fill-r1 = broker job 455, started 21:47:51 UTC. t220-time-r1 queued behind it.
- Wait: `ssh g14blx03 'bash ~/fasth3/runner/probe.sh t220-time-r1'`. Fill may hit 600 s: if fill status=failed
  with a pytest timeout, the time job continues filling; if time also times out, enqueue t220-time-r2.
- Results: blx03 /var/tmp/fasth3/t220/out_time/run.log (E2E_WALL_S lines + LTX timing tables), videos there.
- 21:48-21:50 UTC: fill-r1 (broker 455) and time-r1 (457) failed in 16 s, not drops: b9f8587ce6c's test passes
  image_conditioning= that LTXPipeline.__init__ no longer takes. Fixed on blx03 ~/fasth3/t220 as commit d791f4e949
  (drop the kwarg; production leaves RUN_I2V unset = default 1, same as us). patch.py updated to match.
- Requeued: t220-fill-r2 = broker 459 (started 21:52 UTC, building VAE/upsampler/bf8 DiT caches), t220-time-r2 queued.
- Next wake: read done/t220-time-r2.done. If time-r2 timed out while still filling, enqueue t220-time-r3.
  Then parse out_time/run.log: E2E_WALL_S gen=2..6 (seeds 0-4 warm; gen1 = capture), timing tables, warmup.

## 2026-10-07 22:12 UTC (attempt 1, standard run 3)
- DROP 1 (this config on blx03): 21:57 UTC, blx03, broker job 459 (t220-fill-r2, OUR job), chips 8,9,12,13
  (tray 2) left PCIe. Broker recovery (glx_reset + bridge resets) still looping at 22:11 UTC; chip 9 bridge
  reset failing (broker 472/473), health-gate 474 running. Not reset by us.
- Runner is dead (lock free); t220-fill-r2 stays in running/ (runner resumes it first on restart),
  t220-time-r2 in queue. Probe: `bash tt-project/t220/probe_ready.sh` (blx03 or blx01 30 min clear).
- Next wake: if blx03 clear -> `ssh g14blx03 bash ~/fasth3/runner/runner-start.sh` (reruns fill-r2 = rerun after
  drop 1). If fill-r2 drops again on blx03 -> skip blx03 (2 drops), move to blx01 (needs /var/tmp/fasth3/t220
  build + cache first). If only blx01 clear: blx01 setup (setup.sh paths -> /var/tmp/fasth3) is the next step.

## 2026-10-08 00:45 UTC (attempt 1, standard run 4)
- Runner (pid 22962, started 22:14 UTC) is dead: it waited through a broker upgrade (22:53-23:30 UTC), then blx03
  had another tray-2 incident: chips 8,10-15 off PCIe, broker 483-485 health-gate/bridge-reset failed ~00:30 UTC,
  power-cycle 487 at 00:39 UTC, hold 488 ended and broker restarted 00:41 UTC. Not our job (runner was not running
  a job); not counted as a drop of this config. Config drops on blx03 so far: 1 (job 459, 21:57 UTC).
- fill-r2 still in running/ (resumes first), time-r2 queued. Nothing submitted this run.
- probe_ready.sh missed power-cycle/hold rows; fixed. New probe_blx03.sh = blx03 only, 30 min clean.
- Next wake: `ssh g14blx03 bash ~/fasth3/runner/runner-start.sh`, then wait on
  `ssh g14blx03 'bash ~/fasth3/runner/probe.sh t220-time-r2'`.

## 2026-10-08 02:45 UTC (#251, blx01; blx03 skipped after 2 drops per config)
- Setup on blx01 (all under /var/tmp/fasth3/t220, nothing in /home): src = copy of blx03 ~/fasth3/t220 (tree
  d791f4e949 = ltx-rt b9f8587ce6c + test patch, Release build; COMMIT file, no .git), caches copied from blx03
  (ltx-2.3 VAE/connectors/audio, bf8 DiT 23G, upscaler, JIT 1.4G) + gemma cache from blx01's
  /home/sulphur/tt_dit_cache (read only). venv /var/tmp/fasth3/t48/python_env. Run: run220.sh (= run220_blx01.sh),
  broker env env.yaml (= env_blx01.yaml). CPU import check OK. Footprint ~63 GB on blx01 /var/tmp (199G free after).
- Fill = broker job 901 (submitted 02:38:52 UTC, -t 600, seed 0 gen0 capture + gen1 replay), out_fill/run.log.
- Driver drv251.sh (blx01 pid 1390395, own session) waits for 901; if completed, submits the timed job
  (T220_SEEDS=1,2,3,4, -t 600) once no hold/upgrade/smarton job is running or queued, waits, writes
  /var/tmp/fasth3/t220/drv251.done (FILL_NOT_OK / TIME_NOT_SUBMITTED / TIME_DONE job=.. status=..), log drv251.done.log.
- Next wake: read drv251.done. FILL_NOT_OK: check out_fill/run.log (timeout while JIT compiling -> submit the
  timed job by hand; drop -> rerun fill once). TIME_DONE: parse out_time/run.log E2E_WALL_S gen=1..5 (gen0 = capture).
  After a drop mid-timed run, rerun with T220_SEEDS = seeds not yet saved (ltx_av_fast_1920x1088_<gen>.mp4).

## 2026-10-08 03:00 UTC (#251 attempt 1, standard wake)
- Fill job 901 (blx01) = pytest timeout at 570 s, NOT a drop or compile error. All weight caches hit (only
  vae_enc was built, 27 s); the copied blx03 JIT cache gave 0/3148 hits (prewarm skipped 704 entries as
  foreign-tree), so everything compiled cold. Warmup finished (audio decode done 02:48:30) 1 s before the
  timeout. JIT cache on blx01 is now 6.0 GB, so a rerun should get much further.
- drv251b.sh (blx01 pid 1525001, own session): waits until no smarton/hold/upgrade job runs or is queued
  (t249 job 905 was running), submits fill2 (seed 0, -t 600), waits; if completed submits time
  (seeds 0-4 in one process: T220_SEEDS=1,2,3,4, -t 600), waits. Marker /var/tmp/fasth3/t220/drv251b.done
  (FILL_NOT_SUBMITTED / FILL_NOT_OK / TIME_NOT_SUBMITTED / TIME_DONE job=.. status=..), log drv251b.done.log.
- Next wake: FILL_NOT_OK -> read out_fill2/run.log (2nd timeout => restructure: skip audio warmup or split;
  drop => rerun once, 2 drops => skip). TIME_DONE -> parse out_time/run.log E2E_WALL_S (gen0 = capture,
  gen1 = seed 0, gen2-5 = seeds 1-4); timeout -> rerun with seeds not yet saved.

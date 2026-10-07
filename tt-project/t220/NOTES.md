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

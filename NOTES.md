# t114 notes (conv3d blockings for the 1080p 4x8 per-chip shapes)

## Finding before any device work
- In pad mode (default, no LTX_VAE_EXACT_SHARD), 544x960 on 2x4 has the SAME per-chip shards as 1088x1920 on 4x8
  (latent 9x8 per chip, then 18x16, 36x32, 72x64). So #100 (job 458-467) already swept the 1080p 4x8 pad-mode
  shapes for s2_res, s3_res, s4_res, s1_up. Only pad-mode s3_chg was never swept.
- What is new is LTX_VAE_EXACT_SHARD=1: from s1 on, the shards are 17x15, 34x30, 68x60 (inputs 19x17, 36x32,
  70x62). They look up the SAME _BLOCKINGS keys (keys are the exact dims, from _compute_ltx_decoder_dims), which
  were timed on the padded shards. If exact-mode winners differ, the table needs an exact-shard override (the
  keys collide), e.g. a dict that get_conv3d_config checks when the VAE runs with exact_shard.

## Code (102daa3eea0, pushed)
- bruteforce_conv3d_sweep_ltx.py: _SWEEP_LAYERS_LTX25_544P_145F_HALO_EXACT, test ids exact_<layer>; JSON
  out exact_<layer>_<Cin>x<Cout>.json. CPU test test_conv3d_sweep_halo_cpu.py: 15 pass.
- tmp/blx03/t114/{driver114,run114,stage114}.sh (copied from #100's scripts).
- blx03 ~/fasth3/t48 now at 7e25dc0dbad (detached). No rebuild: 83c11ee2b34..7e25dc0dbad is Python only.

## Device run (blx03)
- Driver launched 2026-10-03 08:17 UTC: /var/tmp/fasth3/t114/src/tmp/blx03/t114/driver114.sh. At launch the
  broker HELD the device (5/32 chips off the bus after another tenant's job 470, ltx-host, 08:07 UTC). Not ours.
  The driver waits up to 2 h for broker health, then runs one broker job per layer:
  exact_s2_res exact_s3_res exact_s4_res exact_s1_up exact_s3_chg s3_chg.
- Marker `T114_DRIVER_DONE <stage> <rc>` in g14blx03:/var/tmp/fasth3/t114/driver.log. rc 9 = drop/reboot during
  OUR job -> stop ALL device work on every galaxy, report. rc 8 = broker never healthy -> relaunch.
  If blx03 reboots, the driver dies without a marker: relaunch with
  `tt-project/harness/templates/blx03-launch.sh t114 /var/tmp/fasth3/t114/src/tmp/blx03/t114/driver114.sh`
  (finished layers have results/<layer>_done and are skipped).
- 08:29 UTC blx03 rebooted (broker host rung after the 08:07 tenant drop) while the driver was still waiting for
  health; no job of ours had run. Relaunched 08:33 UTC (old log: driver.log.prev-reboot0829). First job: 484
  (exact_s2_res).
- Results: /var/tmp/fasth3/t114/results/*.json, per-layer log run114_<layer>.log.

## Next
1. Read the JSONs. Accept a winner only if C_in_block == the table's (keeps the decode bit-identical) and
   best_us <= 0.97 * table_us; check output_check.
2. If exact winners differ from the table: add an exact-shard override for those keys (opt-in with
   LTX_VAE_EXACT_SHARD), CPU test, then one traced decode A/B on blx03 2x4 (pattern: #100 runab100.sh/ab100.py
   on ttp/t100-t93-5-..., add LTX_VAE_EXACT_SHARD=1), md5 vs table.
3. Commit keys only where the gain >= noise; push; clean /var/tmp/fasth3/t114/src.

---

# t48 notes: all LTX-2.5 wins on one branch

Branch ttp/t48-ltx25-integrated (= ttp/t48-integrate-all-ltx-2-5-wins-on-one-branch), base t36 16ba9a383dc.
Merged: t20+t40 (9e336c44b71, includes 0533827a419), t13 (eee3baf7c0d), t18 (63902277007),
t44 tip (1968790b040 + its A/B harness), t8 ltx_eval harness. Python-only diff against t36.

Conflicts:
- pipeline_ltx_distilled.py: t13 and t40 both capture the Gemma encode trace after gen #0. Kept t40's
  open_trace_gate() + capture_trace() (guarded by _trace_captured). t13's open_trace_gate(capture_prompt=) was removed in t55 (no caller).
- utils/video.py: t18's YuvVideoExport (worker-thread video encode) + t13's zero-copy frame wrap and start_encoding;
  the AAC encode runs in finish() before joining the worker, so it overlaps the video encode as in t13.
  test_yuv_export_encodes_audio_alongside_video now gates the video worker on the audio encode starting
  (fails if finish() encodes audio after the join; checked).
- test_ltx_export_latency.py: gemma -> gemma3 import path.

CPU tests (python_env, PYTHONPATH=worktree): export/trace/eval/cache/ltx set (13 files) 78 passed, 8 skipped;
13 pre-existing failures in test_ltx_euler_tail.py and test_ltx_embedding_cache_identity.py (they read
models/tt_dit/encoders/gemma/, renamed to gemma3); same 13 fail on the t36 base tree.
Fold CPU reference (--noconftest): 5 passed. The 78 include the ltx_eval harness (8) and the 13 export/trace tests.

Device: not run (blx03 paused; full-mesh barred by the 22:10 rule). Ready job: tmp/READY_48.md, tmp/blx03/run48.sh.
Next: when the user allows full-mesh runs on blx03, follow tmp/READY_48.md (setup, one job, timings, ltx_eval vs t20).

t113: folded t100 47aecb9bdd7 (halo sweep harness + CPU test) and 118ed6de1f4 (conv3d _BLOCKINGS (4,8):
s4_res (128,64,6,4,8), s1_up (128,64,5,2,16); bit-identical, traced decode 519.7 -> 506.2 ms, blx03 job 469, 544x960/145f).

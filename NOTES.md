# #155 (current): see tt-project/t155/NOTES.md; specs t155-build, t155-j1, t155-j2 queued on the blx03 runner.

# #122 bisect of the job-484 hang (combos 141,142,145-150)

- Driver: g14blx03:/var/tmp/fasth3/t115/src/tmp/blx03/t115/driver115.sh, launched 2026-10-06 02:55 UTC, pid in
  /var/tmp/fasth3/t115/driver.pid, WAIT_PIDS=28542 (#119's driver). Log /var/tmp/fasth3/t115/driver.log, marker
  `T115_DRIVER_DONE <stage> <rc>`; outcomes /var/tmp/fasth3/t115/outcomes.txt; per-combo logs run115_<tag>.log.
- rc 0 all done; 6 = hang and broker not healthy 30 min later (stop); 7 = submit failed; 8 = broker never healthy
  (relaunch the same way; finished combos have results/<tag>_done). Driver gone without marker = blx03 reboot:
  relaunch with `ssh g14blx03 "WAIT_PIDS= setsid nohup bash <driver> > /var/tmp/fasth3/t115/driver.out 2>&1 < /dev/null &"`.
- 2026-10-06 03:02 UTC blx03 rebooted (broker power-cycle/glx_reset recovery, not our job; 8/32 chips off PCIe
  before it). Driver 79490 died before its first submit. Relaunched 03:15 UTC, pid 37098, after adding a
  health() check that waits while any other smarton job is running or queued (t133 job 251 was queued).
- 2026-10-06 03:42:42 UTC blx03 rebooted again (broker recovery jobs 257-268: glx_reset health gates failed,
  bridge-reset chips 8-15 (tray 2) failed, power-cycle with 32/32 chips off PCIe; device back for tenants 03:45 UTC).
  No job of ours ran; t133 job 251 (project) completed at 03:45. Driver 37098 died. Relaunched 03:48 UTC, pid 21066
  (old log driver.log.prev-reboot0342).
- 141 (64,64,3,8,8): PASS, job 269 (03:48-03:50 UTC). 142 (64,64,3,16,4): job 270 submitted 03:50 UTC.
- Next: read outcomes.txt, fill tt-project/t114/BUG.md, log drops (UTC, job id, chips) in the hand-off,
  then rm -rf /var/tmp/fasth3/t115/src on blx03.

---

# t115 notes (off-device triage of the job 484 hang, branch ttp/t115-conv3d-hang-triage)

- Suspect: 143 (64,128,6,8,8) and 144 (64,128,6,16,4). They are the only combos in 141-150 whose L1 prefetch
  shard does not fit (shard 102400/110592 B, only 91136 B left after the other CBs), so the factory falls back
  to the direct reader. They are also the first fallbacks with T_out_block>1, with the largest output block
  (12x4 tiles, fp32 partials). Code reading only; no combo has run on its own.
- Separate bug: halo mode plus that fallback drops the halo without any error (the direct reader never reads
  halo_buffer). Guard 5bce3778127: TT_FATAL in conv3d_program_factory.cpp (not compiled, disk rule) +
  prefetch_shard_fits() in the sweep harness; halo sweeps drop non-fitting blockings before launch.
- 5 LTX _BLOCKINGS entries have no shard and would hit the TT_FATAL. Re-pick them before folding the guard
  into t48: (4,8,1024,1024,(3,3,3),22,10,8), (4,8,1024,1024,(3,3,3),22,5,4), (4,8,128,1024,(3,3,3),22,5,4),
  (4,8,128,1024,(3,3,3),21,5,4), (2,4,128,128,(3,3,3),147,136,120). See test_ltx_table_blockings_without_prefetch_shard.
- Bug draft: tt-project/t114/BUG.md.

## Bisect (only once the user clears device work)
1. `bash tmp/blx03/t115/stage115.sh` (git archive of this branch to g14blx03:/var/tmp/fasth3/t115/src).
   It runs on blx03's ~/fasth3/t48 build (no guard; the harness filter alone keeps 143/144 out).
2. `tt-project/harness/templates/blx03-launch.sh t115 /var/tmp/fasth3/t115/src/tmp/blx03/t115/driver115.sh`
   and use the retry_when it prints. Plan check without side effects: `DRY_RUN=1 bash tmp/blx03/t115/driver115.sh`.
3. One broker job per blocking (141,142,145-150), full mesh then create_submesh(2,4). Marker
   `T115_DRIVER_DONE` in g14blx03:/var/tmp/fasth3/t115/driver.log. rc 9 = drop during our job: stop ALL
   device work, report. rc 6 = hung job: that blocking is the culprit; stop and report.
4. All 8 pass -> 143/144 are implicated; do not run them on purpose.

---

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

- 2026-10-03 08:33-08:42 UTC: job 484 (exact_s2_res, OUR job) HUNG. Combos 1-140 ran (table 13597 us; best so far
  (64,256,1,8,8) 13464 us, -1.1%, under the 3% bar). After [140/300] no output for 300 s; broker killed it (exit
  130). Post-job health gate: active-eth core heartbeat FROZEN (incident
  /var/lib/tt-device-broker/health/incidents/20261003T084256Z_unhealthy_484 on blx03). Broker held the device,
  glx_reset (job 486), fabric-check OK (488), device back for tenants at 08:44 UTC. Driver exited rc 9
  (T114_DRIVER_DONE exact_s2_res 9). Per the rules: ALL device work stopped, reported, waiting for the user.
- The hang is in combos 141-150 (CPU-reproduced order, tmp script in the handoff): (64,64,3,8,8) (64,64,3,16,4)
  (64,128,6,8,8) (64,128,6,16,4) (64,32,3,4,4) (64,32,3,8,2) (64,128,5,4,4) (64,128,5,8,2) (64,128,7,4,4)
  (64,128,7,8,2) on C_in=C_out=512, halo input (1,73,34,30,512), no masks. Fields are (Cin,Cout,T,H,W) blocks.
- No layer finished; results/ is empty. /var/tmp/fasth3/t114 on blx03 is 133 MB (src copy + logs), kept for a resume.

## Resume (only once the user clears device work again)
- Do not rerun the same combo list blind: first bisect 141-150 with one short job that times them one by one with
  a per-combo print before each launch (or exclude them), so a hang names the exact blocking.
- Consider SWEEP_MAX_COMBOS=140 for exact_s2_res: 1-140 already cover the near-table neighbours.

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

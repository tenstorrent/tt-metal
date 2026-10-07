# t170 — t95/t164 eval pack on blx01 (follow-up of #164)

Branch ttp/t170-... = ttp/t164-eval-pack-g15 @ a4b6a835d1a (+ t170 scripts in tmp/t170).
Box: blx01 (g15blx01), everything under /var/tmp/fasth3/t170 (nothing under blx01 /home).
- tree/: hardlinked /var/tmp/fasth3/t48 (bf7db12a149) models/ + the 8 python files that differ in a4b6a835d1a
  (OVERLAY_COMMIT). Build, kernels, JIT cache: t48 tree + /var/tmp/fasth3/cache (job 621's, warm).
- Default warmup (no LTX_WARMUP_T2V_ONLY/ENCODERS cuts: those dropped tray 3 on blx01 in jobs 625/640).
- driver.sh: pack (configs.txt) -> post.py PCC/PSNR vs own baseline -> pick 1-2 -> baseline5 + <cfg>5
  (seeds 0-4, default prompt) -> post.py. Marker /var/tmp/fasth3/t170/driver_pack/DRIVER.done; pid in driver.pid.
  Results /var/tmp/fasth3/t170/res/<label>/, summaries res/summary_configs{,5}.md.
- Timeouts: baseline 300 s; knobs = 1.5 x measured baseline wall + 120 (<= 600); one retry at 600 on timeout.

## Run log
- 2026-10-06 22:35 UTC: driver started on blx01 (pid 203047). Baseline = broker job 665.
- 22:39 UTC: baseline job 665 OK: process wall 184 s (warmup 85.5 s), 0 JIT compiles, gen#0 34.258 s,
  gen#1 6.017 s, gen#2 6.041 s. Knob limit -> 396 s.
- Wake probe: ssh blx01 test -e DRIVER.done, or the driver pid is gone (reboot/power-cycle kills it: then
  relaunch with ADOPT=<label>:<job> for a job still queued/running in the broker, PRIOR_DROP=<label> if it dropped).
- Phase-2 scoring on g15: tmp/t170/score_g15.sh <labels...> (detach), marker data/g15/t170/SCORE.done.

## Next (after DRIVER.done)
- scp the mp4s of baseline5/<cfg>5 (and baseline + winner) to g15 tt-project/data/g15/t170/ (mp4s only),
  run VBench + per-seed PCC/PSNR vs baselines/ltx25_1080p_6s/ref_dv145 (ltx_eval batch --vbench-ref) on g15,
  look at stills, write the table + recommendation, commit, land on ttp/t48-ltx25-integrated.
- 22:50:39 UTC DROP: blx01 tray 3 (chips 16-23) left PCIe in adaln job 669 (ours, 66 s in, default warmup).
  Broker recovered; host rebooted ~22:57 and killed the driver. Done: baseline 665, exact_shard 666, lofi 667, gate 668.
- 23:01 UTC: driver relaunched (pid 14047) with PRIOR_DROP=adaln; adaln = job 685, -t 450 (TO_KNOB default,
  the relaunch did not re-read the baseline wall; still under the cap). One more adaln drop skips it.
  Tray 3 now dropped under both warmup configs (625, 640, 669): if it keeps dropping, move the pack to exabox.
- Pack done (all 8 configs rc=0: 665-668, 685-688); summary at res/summary_configs.md. Phase-2 pick: exact_shard
  (the only config beating baseline with PCC 1.0; gate/adaln/all faster but PCC 0.95-0.98, PSNR 21-28 dB).
  baseline5 OK job 689 (160 s).
- 23:27:50 UTC DROP: blx01 chips 24-31 (broker: tray [3]) left PCIe in exact_shard5 job 690 (ours, 164 s in).
  Broker recovered (705 hold ended 23:37:58); host rebooted ~23:34, killing the driver.
- 23:41 UTC: driver relaunched (pid 62236) with PRIOR_DROP=exact_shard5; exact_shard5 = job 707 (-t 450).
  A second drop skips it. Next: on DRIVER.done, tmp/t170/score_g15.sh baseline5 exact_shard5 detached on g15.
- 00:00 UTC: DRIVER.done "0". Phase-2 scoring done (data/g15/t170/SCORE_p2a.done): baseline5 and exact_shard5
  clips are byte-identical (md5); vs ref_dv145 PCC 0.9931 mean (worst 0.9816), PSNR 34.25 dB mean (worst 31.72);
  VBench subj 0.888 / bg 0.922 / img 0.563 / motion 0.984 (ref 0.893/0.926/0.553/0.985).
  Warm e2e gen1-5: baseline5 5.995 s mean, exact_shard5 5.939 s mean (-56 ms, every seed faster; VAE 0.56 -> 0.51).
- Visual check of pack stills (data/g15/t170/stills/grid_g12.jpg, gen1 boat + gen2 fisherman): gate/adaln/all look
  as clean as baseline; they diverge in trajectory (pose, birds), no artifacts. The driver's phase-2 gate
  (PSNR >= 25 on gen2) dropped them on metrics alone, so one more phase-2 job: fast5 = exact_shard + gate + adaln
  (lofi/agmm left out: no speed gain, lower PCC). Code commit a3216e1486d flips LTX_VAE_EXACT_SHARD default on.
- 00:05 UTC: driver bug: broker server.log rotated at midnight -> health() saw no event; now reads server.log.1 too.
  Relaunched TAG=fast PHASE2_LIST=configs5b.txt (pid 221453); fast5 = broker job 708, -t 400.
  g15: tmp/t170/score_after_fast.sh detached (ttp detach t170-score-fast) waits for driver_fast/DRIVER.done,
  then score_g15.sh fast5 -> data/g15/t170/SCORE.done.
## Next (on wake)
- Read res/summary_configs5b.md on blx01 (fast5 timing + PCC vs baseline5) and data/g15/t170/score.log (VBench).
  Look at data/g15/t170/fast5/vbench/seed*_cmp_f072.png vs baseline5. If fast5 VBench ~= baseline5 and no visual
  regression: also flip LTX_FUSE_GATE_ON_DEVICE / LTX_FUSE_NORM_ADALN defaults (code commit), else exact_shard only.
- Write RESULTS.md (tables, recommendation, video paths + still, drop log), land code on ttp/t48 via a -land branch.
- 2026-10-07 light wake: fast5 scored BATCH FAIL vs ref_dv145 (PCC 0.874 mean, worst 0.674; PSNR 22.2 dB, worst 18.1);
  VBench subj 0.888 (= baseline5). blx01 res/summary_configs5b.md missing (check driver_fast res dir for fast5 timing).
  Needs judgment: view data/g15/t170/fast5/vbench/seed*_cmp_f*.png, decide gate/adaln defaults.
- 2026-10-07 00:25 UTC standard wake: visual 5-seed check of fast5 clean (trajectory drift only); VBench = baseline5.
  Flipped LTX_FUSE_GATE_ON_DEVICE / LTX_FUSE_NORM_ADALN defaults on (2aae332c37b, test_ltx_fused_defaults.py).
  Landed a3216e1486d + 2aae332c37b on ttp/t48-ltx25-integrated as f6b806516cc. Results: tmp/t170/RESULTS.md.
  Left on blx01: /var/tmp/fasth3/t170 (tree hardlinks + res mp4s) — cleanup pending (followup).

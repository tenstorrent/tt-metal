# t140 — eval pack, 4x8 1080p (blx03)

Branch ttp/t140-eval-pack-4x8. Overlay staged at g14blx03:/var/tmp/fasth3/t140/src (REV file), C++ build from blx03 ~/fasth3/t48.
Collect-only of test_pipeline_distilled -k bh_4x8sp1tp0_ring passes (1/8).

Driver: /var/tmp/fasth3/t140/src/tmp/t140/driver.sh, launched via tt-project/harness/templates/blx03-launch.sh t140.
- Waits for the t136 and t141 drivers to finish (or 2 h idle), then runs tmp/t140/configs.txt, one broker job per config,
  each after the blx03 health gate. Job ids: /var/tmp/fasth3/t140/jobs.txt. Per-config output: /var/tmp/fasth3/t140/<label>/.
- Done marker: `T140_DRIVER_DONE` in /var/tmp/fasth3/t140/driver.log. Relaunching skips configs with T140_EXIT=0.

Next step after the marker:
1. rsync g14blx03:/var/tmp/fasth3/t140/<label>/ (run.log, mp4, png, broker slices) to tt-project/t140/.
2. `python3 tmp/t140/post.py tt-project/t140` -> summary.md/json (warm e2e, stages, PCC/PSNR vs baseline).
3. 5-seed phase: write a second configs file (baseline + 1-2 best + any PCC/PSNR drop) with LTX_E2E_SEEDS=0,1,2,3,4,
   relaunch with CONFIGS=<file>. Compare per seed vs baseline.
4. Table + recommended default set + videos/stills; commit, push, clean /var/tmp/fasth3/t140 bulk.

Run 2 (2026-10-06 03:12 UTC): blx03 rebooted 03:02 and killed the first driver before any t140 job. Relaunched at
1016c3aac47 (waits for every other smarton job, not only t136/t141; pair cutoff 30 min idle). blx03 pid 31022.
If the probe wakes and the log has no DONE line, the driver died (reboot): relaunch it the same way; it resumes.

Drops logged:
- 2026-10-06 02:17-02:19 UTC, blx03 job 212 (smarton, t136 run_ab.sh), chips 8-15 (tray 2) off PCIe, chip 15 UNHEALTHY;
  job 216 (t141) abandoned; resets 214/215 failed, health-gate 217 glx_reset; blx03 rebooted ~02:26 UTC. No t140 job was running.
- 2026-10-06 ~03:02 UTC: blx03 rebooted (cause not seen; no t140 job running). Killed t140 driver pid 16601.
- 2026-10-06 03:10 UTC: blx03 job 246 (smarton, t141 e2e) killed -9; chips 8-15 (tray 2) left the bus; broker post-job gate
  escalated to glx_reset. No t140 job running.

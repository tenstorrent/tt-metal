# t141 notes (5-seed confirm of #138's 6.238 s warm 4x8 e2e)

Branch ttp/t141-e2e-5seed. 9f2b28b7663 = t48 c4409b1fa24 + test-only LTX_E2E_SEEDS knob (one warm replay gen
per listed seed). blx03 t48 tree is checked out detached at 9f2b28b7663; the driver puts it back to c4409b1fa24.

Device job: driver blx03:~/fasth3/t141drv/driver.sh (copy: tmp/t141/driver.sh), log /var/tmp/fasth3/t141/driver.log,
job script /var/tmp/fasth3/t141/run_e2e.sh (copy: tmp/t141/run_e2e.sh), output /var/tmp/fasth3/t141/out
(run.log, ltx_av_fast_1920x1088_{0..5}.mp4 + _t3s.png). gen#0 = capture (seed 0, default prompt);
gen#1..#5 = warm seeds 0..4 with FRESH_LTX_PROMPTS[0,1,2,0,1]. gen#1 matches t138 gen#1 and t20 gen#1.

Done marker: grep T141_DRIVER_DONE /var/tmp/fasth3/t141/driver.log ("e2e 0" = clean; 9 = drop/reboot -> rerun;
8 = broker never healthy; 7 = submit refused).

Drops seen:
- 2026-10-06 02:19:44 UTC, broker job 212 (smarton, task t136, not ours), chips 8-15 (tray 2, incl. chip 12)
  left the PCIe bus. Our first submit (job 216, 02:20:19) raced the recovery; withdrawn 02:20:46 (never ran).
  Driver fixed to re-check health before every submit attempt; relaunched 02:21:16.
- Same incident continued: tray-2 bridge-resets 214/215/219 failed, health-gate 217 failed, broker power-cycled the
  box (job 220, 02:26:22 boot) -> killed our waiting driver (no job of ours was running). Startup health gate passed
  02:29:19 (32 chips, fabric OK). Driver relaunched 02:29 -> broker job 224 running since 02:30:09.
- 2026-10-06 02:31:46 UTC, broker job 224 (OURS, t141 e2e), chips 8-15 (tray 2, incl. chip 12) left the PCIe bus
  during warmup stage 1 (~1.5 min in). Broker killed it (-9); bridge-resets 226/227 and health-gate 228 failed,
  broker escalating. Drop #1 for this config (216 never ran). Logs kept as /var/tmp/fasth3/t141/*.224 and out_224/.
  Driver relaunched 02:35:25 (waits for health, then submits). Second drop in a row of this config -> skip per rule.

Next (after DONE e2e 0):
1. scp run.log + mp4/png to tt-project/t-e2e/t141/ on g15blx02 (gzip run.log).
2. E2E_WALL_S per gen#1..#5 -> median/min/max; stage tables.
3. ltx_eval (g15blx02, python_env): gen#1 vs tt-project/baselines/t20/ltx_av_fast_1920x1088_1.mp4 (PCC/PSNR +
   VBench ref) and vs t138 gen#1; VBench (partial) on all 5 warm clips; compare dims to t20 gen#1's VBench.

# #37 S2 prompt reuse A/B on blx03 — NOTES

Deviation from spec (judgment call): no t22 checkout on blx03. The blx03 tree ~/fasth3/tt-metal (t36, 16ba9a383d)
already contains 5d993cd2f7c, has a build and warm caches, so the A/B runs there via tmp/blx03/run25.sh.
No new build or cache. Conv decoder (LTX25_DIFFVAE=0, the main decode path) keeps each job ~3 min.

Driver: blx03:~/fasth3/t37/drive37.sh (copy in tmp/t37/), detached, log drive37.log, job ids in ~/fasth3/t37/jobs.
It waits for no other smarton broker job, submits s2reuse0 (LTX_S2_PROMPT_REUSE=0), waits, then s2reuse1,
then runs compare37.sh -> ~/fasth3/t37/compare.txt and writes ~/fasth3/t37/DONE.
Outputs: blx03:~/fasth3/out/t37/{s2reuse0,s2reuse1}/ (run.log, mp4 per gen, lat.gen*.pt), still s2reuse1_gen1_t3s.png.

Pass: mp4 md5 and S2 latents equal per gen, and S2 (denoise init "prompt" ms + STEP_MS, table "Stage 2 denoise")
faster with reuse=1. gen#0 is the capture gen; compare replays gen#1/gen#2 for timing.

Next step on resume: ssh g14blx03 cat ~/fasth3/t37/compare.txt; copy the still + one mp4 back to tmp/t37/;
if not identical set LTX_S2_PROMPT_REUSE default to 0 on the t22 branch and push; then rm -rf blx03:~/fasth3/out/t37
(keep nothing there) and ~/fasth3/t37.

## 2026-09-30 20:46 UTC (after reboot)
blx03 rebooted 20:25 UTC; the driver died before submitting anything (no jobs file, empty log). Nothing measured.
blx03 broker: device HELD (degraded), 8/32 chips off PCIe after our t41 job 904 was broker-killed (chips left PCIe);
auto-recovery (bridge-reset, glx_reset) failing. Did NOT relaunch the driver: no autonomous submitter while the
device is flaky and a pause may come. On resume: check steer/pause, confirm broker not HELD and no smarton job,
then relaunch `setsid nohup bash ~/fasth3/t37/drive37.sh > ~/fasth3/t37/drive37.log 2>&1 &` on blx03.

## 2026-09-30 20:51 UTC (woken, still held)
blx03 broker still HELD (degraded) 25 min after reboot: 8/32 chips off PCIe, bridge-reset and glx_reset keep
failing (jobs 906-912). No job submitted, driver not relaunched. Waiting on the broker hold to clear
(retry_when checks that "HELD" is gone from the RUNNING section of `tt-device-mcp status 1` on blx03).

## 2026-09-30 20:59 UTC (woken by a false probe, still degraded)
The old probe fired because the broker's RUNNING line changed from "HELD" to "hold-deadline-escalate: glx_reset".
The device is not back: `lspci -d 1e52:` shows 24 of 32 chips; chips 8-15 (one tray) stay off PCIe; bridge-reset
and glx_reset keep failing (jobs 906-914). Nothing submitted, driver not relaunched.
New probe: 32 chips on PCIe and no broker job in RUNNING (see result.json retry_when).

## 2026-09-30 21:15 UTC (device back, driver relaunched)
blx03 broker power-cycled at 21:05 UTC (job 918); all 32 chips back, startup + fabric-check passed (920, 921).
Relaunched drive37.sh detached on blx03. It waits for project job 922 (another task, blx03_ab.sh) to finish,
then submits s2reuse0, then s2reuse1. Check: `ssh g14blx03 cat ~/fasth3/t37/jobs ~/fasth3/t37/drive37.log`;
done when ~/fasth3/t37/DONE exists. Then follow "Next step on resume" above.

## 2026-09-30 22:10 UTC (reference done, reuse=1 not run, blx03 unstable)
Reference s2reuse0 = job 931, completed (180 s). Its log is tmp/t37/job931_s2reuse0.log (copied over
blx03 ~/fasth3/out/t37/s2reuse0/run.log, which job 937 had overwritten). Hashes in tmp/t37/s2reuse0.md5.
S2 denoise (table, replays gen1/gen2): 2.53 s / 2.54 s. S2 steps 818-828 ms. S2 prompt upload 35-36 ms of the
47-49 ms denoise init: this is the most prompt reuse can save per request (~1.4% of S2).
E2E wall gen1 8.758 s, gen2 7.108 s (conv decoder).

Driver bug (mine): it parsed "^JOB ID:" but `tt-device-mcp run-bg` prints "Job <id> queued". The id stayed empty,
so it resubmitted s2reuse0 after 931 (job 932) and again while the device was held (937). Both 932 and 937 were
broker-killed when chips 8-15 left PCIe, both during the first S1 steps (strided all-gather warm-up); 932 led to
a power cycle at 22:00 UTC. With t41's job 904 that is 3 of the last 4 LTX jobs on blx03 dropping the same tray.
Fixed in tmp/t37/drive37.sh (and on blx03): parses "Job N queued", treats HELD/recover/reset/power as busy, skips
s2reuse0 when its latents exist. NOT relaunched: waiting for a user call on whether blx03 is fit for more jobs.

On resume (only once the user OKs more blx03 jobs): confirm `lspci -d 1e52: | wc -l` = 32 and no broker hold,
then on blx03 `setsid nohup bash ~/fasth3/t37/drive37.sh > ~/fasth3/t37/drive37.log 2>&1 &` (runs s2reuse1 only).
Reference outputs (70 MB) stay in blx03 ~/fasth3/out/t37 until then; delete that and ~/fasth3/t37 afterwards.

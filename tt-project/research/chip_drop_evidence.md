# Chip drops on g15blx02 and g14blx03, 2026-09-30

For the platform team. All times are UTC (both hosts' clocks are set to UTC; the broker log timestamps match `last -x`).
Sources: `/var/log/tt-device-broker/*.log` and `tt-device-mcp status` (read-only), `last -x`, and task results #33, #36, #37, #41 in tt-project.
Kernel logs (dmesg, journal: AER, link-down, MCE) are **not readable** by our user, so the "fatal processor error" is **unverified**.

"Ours" means a FastH3 project job (owner smarton, submitted from ~/fasth3). `ltx-host` is the live LTX service.
Every LTX job listed ran the full 32-chip 4x8 mesh (`bh_4x8sp1tp0_ring`), 1080p, LTX-2.x distilled pipeline.

## g15blx02

| Time (drop / reboot) | Job | Owner | Chips off PCIe | What the run was doing | Ours? |
|---|---|---|---|---|---|
| 00:23 / 00:57 | 427 | ltx-host | 8-15 (tray 1) | live service, 10s request, denoise step 2/3 | no |
| ? / 00:13 | 412 | ltx-host | not checked | live service (per #33) | no |
| – / 07:17 | none | – | not checked | no job running (ltx-host 491 queued) | no |
| 12:10 / 12:19 | 577 | smarton (t13) | 0-7 (tray 0) | pytest `test_pipeline_distilled` 4x8; stage 1 denoise done, entering stage 2 denoise | yes |
| 13:42 / 13:51 | 598 | smarton (t18) | 0-7 (tray 0) | `ab.sh async`; VAE encoder loaded, start of stage 1 denoise | yes |
| 14:04 / no reboot | 600 | smarton (t24) | 20 (tray 2) | `job.sh`; VAE decode done at 14:01, audio-decode warmup, drop came as the job hit its 600s timeout. Recovered by glx_reset | yes |
| – / 15:01 | none | – | not checked | no job; our job 638 ended 2 min before and its health check passed (32/32 chips at 14:59). Looks like a host crash | no |
| ~15:14 / 15:23 | 640 | smarton (t10) | 3-7 (tray 0; exact list not in log) | `run25.sh dv145` (DiffVAE, 145 frames); VAE encoder loaded, start of stage 1 denoise | yes |
| 15:36 / 16:00 | 642 | smarton | 0-7 (tray 0) | `ab.sh process`; stage 1 denoise done, entering stage 2 denoise | yes |
| 16:27 / 16:36 | 689 | smarton | 0-7 (4-7 first seen) | `run25.sh dv153` (DiffVAE, 153 frames); first 2-step denoise done (first step 49s = compile), entering next stage. Not in VAE yet. User reported this as "09:22" = 09:22 PDT | yes |
| – / 18:33, 19:18, 20:41, 21:08 | none of ours | – | not checked | project device pause from 16:03; no project submissions. Other tenants' activity **not checked** | no |

Note: the g15blx02 broker is not reachable at 22:10 (`tt-device-mcp status` fails), so jobs could only be read from the log files.

## g14blx03

| Time (drop) | Job | Owner | Chips off PCIe | What the run was doing | Ours? | Recovery |
|---|---|---|---|---|---|---|
| 00:28, 05:32, 05:43, 06:59, 07:11, 07:54, 09:15 | 664, 704, 708, 737, 748, 767, 786 | ltx-host | 8-15 (tray 1) each time | other tenant's LTX jobs, before we used blx03 | no | not checked |
| 09:32 | 810 | smarton (not a project job) | 30 | `bh-mod set chip_limits.voltage_margin=25`; not from ~/fasth3, **origin unverified** | no | host boot at 09:22 precedes it |
| 20:18 | 888 | smarton (t37) | 8-15 | `run25.sh s2reuse0` (conv VAE, S2 prompt reuse off); start of a denoise stage | yes | host boot 20:25 |
| 20:37 | 904 | smarton (t41) | 8-15 | `job.sh 2`, plain eager LTX-2.5 4x8; stage 1 denoise step 7/8 | yes | bridge reset, SBR, tray re-power, mesh reset (jobs 906-910) all failed; power cycle held off (last one 1103s earlier); mesh held degraded until power cycle, host boot 21:05 (job 919) |
| 21:53 | 932 | smarton (t37) | 8-15 | `run25.sh s2reuse0`; 1080p denoise step 1/2 | yes | power cycle, host boot 22:00 |
| 22:05 | 937 | smarton (t37) | 8-15 | same config as 932; right after denoise init, before step 1 | yes | reset sweep failed; power cycle held off (682s since last); mesh degraded at 22:09 |

## Pattern

- Every drop with a job attached happened during a full 4x8 LTX run at 1080p, in denoise or at a stage boundary. None happened inside VAE/DiffVAE decode. One (600) came after decode, at timeout.
- Drops take out a whole tray. g15blx02: mostly tray 0 (chips 0-7) under our jobs, tray 1 under the live service. blx03: always tray 1 (chips 8-15), for the live service in the morning and for us in the evening.
- No single script or config repeats: pytest, ab.sh, job.sh, run25.sh (dv145, dv153, conv) all appear.
- Two g15blx02 reboots (07:17, 15:01) had no job running, and the live service hit the same failure before our project started.
- Our read: the fault is the trays/host under heavy 4x8 load, not a specific workload. Our job volume raised how often it happened. Not provable without kernel logs.

## Could not verify

- Kernel-side cause (AER, PCIe link-down, MCE): no dmesg/journal access.
- Chip ranges for g15blx02 reboots at 00:13, 07:17, 15:01 and 18:33-21:08.
- Exact chip list for job 640 (log shows `[3,4,5,6,7]` in the kill line only).
- Who ran job 810 on blx03.

# blx03 (g14blx03) drop at ~07:22 UTC, 2026-10-01

Task #63, read-only. Checked 07:30-07:36 UTC. blx03 was reachable (ping, ssh from g15blx02) and had been up since 07:26.
No device jobs submitted, nothing reset or restarted.
Sources on blx03: `last -x`, `journalctl -b -1` (incl. `-k`, read with `sudo -n`), BMC SEL (`ipmitool sel elist`, read-only),
`/var/log/tt-device-broker/*.log` (jobs 994, 995, 997-999, 000, 002-007, server.log),
`/var/lib/tt-device-broker/health/{health_events,auto_recovery}.jsonl` and `incidents/20261001T0720*`.
All times UTC.

## Answer

- **Our job 000 did not cause it.** It was queued at 07:21:50, about 2 minutes **after** tray 1 (chips 8-15) had already dropped.
  It waited behind the broker's reset and never touched the device before the reboot.
- **What dropped the tray:** the `ltx-host` job 995, a tray stress run (`~/tray-stress/ltx_stress.py`, 3 x 10s 1080p i2v,
  full 4x8 mesh, `HOST_FMAX=1150`). It was submitted from e11tscale01 (172.27.31.21), not from our project.
  Tray 1 went down about 60s into the load, during the Gemma text-encode warmup.
- **What took the box off the network:** not a crash. The broker's own recovery reset the tray, failed, and then
  **power-cycled the host on purpose at 07:23:56** (`tray-down-no-window`).
- **Underlying cause:** the weak tray 1 on blx03, the same tray as every earlier blx03 drop. It fails under heavy full-mesh load.
- **Confidence:** high that our job was not the trigger, and high that the reboot was the broker's power cycle.
  Medium on the root cause. No AER or MCE was logged: this platform handles AER in firmware, and the SEL shows only
  the tray re-POST, nothing on the PCIe side.

## Timeline

| Time | Event | Source |
|---|---|---|
| 07:14:05-07:16:01 | **Our job 994** (t43, full mesh + create_submesh(2,4)): done, exit 0. Post-job gate: 32/32 chips, heartbeat OK | 994 log |
| 07:18:39 | **ltx-host job 995** starts: `ltx_stress.py`, `HOST_FMAX=1150`, `ngen=3 length=10s res=1080p-portrait i2v=True`, mesh `bh_4x8sp1tp0_ring` | 995 log, journal (ssh from 172.27.31.21) |
| 07:18:50-07:19:40 | Telemetry: all 32 chips at aiclk 1150, heartbeats advancing | incident telemetry_trace.json |
| **07:19:40-07:19:42** | **SEL: `UBB1_ASIC0..7_Stat FRB2/Hang in POST failure Asserted`.** Tray 1 (chips 8-15) is re-POSTing | BMC SEL 6e53-6e5b |
| 07:19:46-48 | Same 8 deasserted | SEL 6e5c-6e64 |
| 07:19:50 | Telemetry: chips 8-15 read 0xFFFFFFFF; chip 0 still fine | telemetry_trace.json |
| 07:19:54 | metrics exporter still counts 32 PCIe devices | journal |
| 07:19:59 | Job 995 last line: `gemma text-encode: still working, 20s elapsed` | 995 log |
| 07:20:00 | Broker: `chip_dead` 8-15, removed from the bus | incident 20261001T072000Z_chip_dead_none |
| 07:20:02 | Broker kills job 995 (-9, `MCP killed: chips left PCIe`). Post-job gate: `24 chip(s) in sysfs, expected 32` | 995 log |
| 07:20:02 / 07:20:05 | Bridge resets (jobs 997, 998) fail: `no_bridge` | health_events.jsonl |
| 07:20:05 | `TRAY_DOWN_NO_WINDOW (8/32 off the bus, trays [1]) ... host_at_risk: true` | 995 log, health_events |
| 07:20:37 | Broker runs `tt-smi -glx_reset` (job 999) | journal, 999 log |
| 07:20:42-44 | SEL: UBB1 ASIC0-7 FRB2 asserted again (tray re-powered by the reset, never came back) | SEL 6e66-6e6d |
| 07:21:29 | `tt-health: PCI 24/32 — rescanning bus` → `ALERT rescan left 24/32: link-level drop` | journal |
| **07:21:50** | **Our job 000** (`run44.sh 0`, #46 fold A/B arm 0) submitted → `QUEUED (WAITING FOR THE DEVICE: BROKER DEVICE OP IN FLIGHT: DEVICE RESET: TT-SMI -GLX_RESET)` | server.log |
| 07:22:49 | glx_reset fails after 132s: `PanicException: Didn't find all 32 chips` | health_events (reset_done rc=1) |
| 07:22:56 | `back-to-back reset sweep issued — settling 60s, then one verify` | server.log |
| 07:23:29-37 | Health probe again 24/32 | journal |
| 07:23:48 | Last journal line of boot -1 (no shutdown sequence) | journal |
| **07:23:56** | **Broker: `auto_power_cycle_request` `tray-down-no-window: every reset type ran back-to-back and the mesh did not verify`** | auto_recovery.jsonl, health_events |
| 07:23:58 | SEL: `Power Unit PowerState Power off/down Asserted` | SEL 6e6f |
| 07:24:37 | SEL: power back on | SEL 6e72 |
| 07:26:24 | Host boot (`last -x`) | last -x |
| 07:28:10-07:29:16 | Broker starts, 32/32 chips, eth heartbeat OK, fabric check OK (61s), hold ended | server.log, jobs 003-007 |
| 07:29:16-07:30:20 | Broker **re-queued our job 000 from before the crash and ran it**: passed, exit 0. Post-job gate 32/32 healthy | 000 log, server.log `QUEUE-RESTORE requeued 1 job(s)` |
| 07:35 | Queue empty, nothing running | `tt-device-mcp status` |

## Key excerpts

Broker server.log:
```
07:21:50 | INFO | -> QUEUED (WAITING FOR THE DEVICE: BROKER DEVICE OP IN FLIGHT: DEVICE RESET: TT-SMI -GLX_RESET (~60S)): job_id=000, owner=smarton
07:28:12 | INFO | QUEUE-RESTORE requeued 1 job(s) that outlived the last broker
07:28:13 | WARNING | CLEAN-GATE holding job 000 — device degraded: ... boot attributed to a broker-fired power-cycle recorded at 2026-10-01T07:23:56.320288.
07:29:16 | INFO | JOB_RUNNER starting job_id=000
```
Job 995 (ltx-host):
```
STRESS|07:18:40.010|start ngen=3 length=10s res=1080p-portrait weights=sulphur i2v=True gap=0.0
STRESS|07:18:40.150|host fmax 1150 MHz applied to 32 chips; failures=[]
07:19:59 ... gemma text-encode: still working, 20s elapsed
STATUS: killed  EXIT CODE: -9
[health-gate/post-job] heartbeat: UNHEALTHY — 24 chip(s) in sysfs, expected 32 (chips dropped off the bus)
[health-gate/post-job] TRAY_DOWN_NO_WINDOW (8/32 off the bus, trays [1]) — firing every reset type back-to-back ...
```
BMC SEL:
```
6e53 | 10/01/2026 | 07:19:40 | Processor UBB1_ASIC0_Stat | FRB2/Hang in POST failure | Asserted
...  (ASIC1-7 by 07:19:42)
6e6f | 10/01/2026 | 07:23:58 | Power Unit PowerState | Power off/down | Asserted
6e72 | 10/01/2026 | 07:24:37 | Power Unit PowerState | Power off/down | Deasserted
```
auto_recovery.jsonl:
```
{"action": "power-cycle", "reason": "tray-down-no-window: every reset type ran back-to-back and the mesh did not verify", "boot_id": "ee0e0c60-...", "at": "2026-10-01T07:23:56.320288"}
```
Kernel (`journalctl -b -1 -k` from 06:00): only `bridge window` messages, all from the broker's PCI rescans (07:20:02, 07:20:05, 07:21:29, 07:22:49, 07:23:29).
No AER, MCE, pciehp link-down, thermal or tt-kmd error. Boot log: `_OSC: platform does not support [... AER ...]`, `GHES: APEI firmware first mode`.
So PCIe errors go to the BMC, and the SEL has none for this event. Incident telemetry: `aer_*` = 0, `therm_trip_count` = 0 on surviving chips.

## Compared with earlier incidents

| Incident | Our job? | What the device did | Why the host went down |
|---|---|---|---|
| 02:10, job 989 (ours, bare 2x4 mesh) | yes, running | fabric init failure on device 1 | **Real host crash**: SEL `Bus Correctable error` storm 02:12:39-42, then `Processor #0x88 Uncorrectable machine check exception` at 02:13:42. No broker power cycle logged for that boot. |
| 07:14, job 994 (ours, full mesh + create_submesh(2,4)) | yes | passed, 32/32 after | - |
| **07:19-07:23, job 995 (ltx-host stress)** | **no**: ours (000) was only queued | tray 1 re-POST (FRB2), chips 8-15 off PCIe | **Broker power cycle** after the reset ladder failed |
| 07:29, job 000 (ours, full mesh + create_submesh(2,4)), after reboot | yes | passed, 32/32 after | - |
| Earlier blx03 boots 21:57, 22:41, 23:30 (2026-09-30) | - | tray down | also broker power cycles (`auto_recovery.jsonl`) |

So the 07:22 event is a different kind from 02:10. The 02:10 one was an MCE host crash during our bare-2x4 fabric init.
The 07:22 one was a tray-1 drop under the stress workload, followed by the broker's planned power cycle.
Both full-mesh + create_submesh(2,4) jobs we ran today (994, 000) finished clean, with healthy post-job gates.

## Other findings

1. **The 1150 MHz clock cap did not stop the drop.** The `ltx_stress.py` docstring says tray 1 "never [dropped] in six runs with the clock ceiling at 1000/1150 MHz".
   This run had `HOST_FMAX=1150` applied (telemetry aiclk 1150) and dropped after ~60s. Our `run43.sh`/`run44.sh` also call `~/tray-stress/hostfmax.py 1150`.
   That cap lowers the risk at best, it is not a guard.
2. **The broker re-runs queued jobs after a reboot.** Job 000 ran at 07:29:16, after the crash and just before the user's 07:30 stop order.
   It was harmless (passed, chips healthy), but a "stop" only takes hold once our queued jobs are killed.
   As of 07:35 nothing of ours is queued or running.
3. Unrelated noise: some other smarton session (not ours, cwd `/home/smarton/tmp`) polls `ipmitool sdr get TEMP_UBB*/Power_UBB*` about once a second.
   `tt-telemetry.service` crash-loops on `Unknown option: --enable-otel`. Neither touches the chips.

## Could not verify

- Why tray 1 re-POSTed (tray power/VRM or the ASICs themselves). The SEL only says FRB2/Hang in POST. Tray power readings at 07:19:40 are not stored anywhere we can read.
- Who launched the stress job from e11tscale01. It runs as `ltx-host` from a smarton ssh session, outside `~/fasth3`.

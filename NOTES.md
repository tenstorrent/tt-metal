# t119 — re-time re-picked conv3d blockings (2x4 submesh, blx03)

## Status 2026-10-04 18:17 UTC: STOPPED, chip drop during our job
- Driver: /var/tmp/fasth3/t119/src/tmp/blx03/t119/driver119.sh (staged from a4c75dc718f), log /var/tmp/fasth3/t119/driver.log on g14blx03.
- First step ref544 = broker job 978 (submitted 18:14:54, after ~23 h of idle device; no pre-job gate ran).
  hostfmax 1150 ran (exit 0), then the mesh open hit `MMIO per-op timeout` on chip 12 (PCIe 0000:45:00.0) in
  LocalChip::start_device at 18:15:04. No kernel ever ran. Job exit 1 after 13 s.
- Broker post-job gate: chip 12 FELL OFF THE PCIe BUS (tray 2); BMC tray sweep then showed trays 1/3/4 down too.
  Incident: /var/lib/tt-device-broker/health/incidents/20261004T181509Z_unhealthy_none (kernel AER NonFatalErr on root port 40:01.5 at 18:15:06).
- Per the stop rule, the driver stopped (rc 9) and nothing else was submitted. No reference was recorded; no arm was timed.

## Resume (only after the user clears device work)
1. `bash tmp/blx03/t119/stage119.sh` (from this worktree), then from the project root:
   `tt-project/harness/templates/blx03-launch.sh t119 /var/tmp/fasth3/t119/src/tmp/blx03/t119/driver119.sh`
   (move the old driver.log aside first; the retry_when greps `_DRIVER_DONE`).
2. Steps (one broker job each): ref544, the 4 s0ups arms (+ a baseline repeat), ref1080, the 3 s4res arms.
3. Read `T119_CONV`/`T119_HIT` (was the key hit?), `VAE_REF` (gate: >=40 dB overall, >=35 dB seams), and traced times from
   /var/tmp/fasth3/t119/run_*.log. Compare against #100's 506.2 ms (1150 MHz clamp). Commit winners to t48.
- Possibly worth dropping the hostfmax clamp on the resume run, or running it as a separate job: it is the only thing that touched every chip just before the open. Not proven to be the cause.

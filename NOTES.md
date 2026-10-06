# t119 — re-time re-picked conv3d blockings (blx03)

## 2026-10-06: resumed as full 4x8 jobs (#127, user update #111)
Scope change, from reading conv3d._BLOCKINGS: on the production 4x8 at 1080p/145f only ONE of the 5 re-picks is
on the path: (4,8,128,1024,(3,3,3),21,5,4) ups_initial, which is the LATENT UPSAMPLER's initial conv (S1 latent
17x30 -> 5x4 per chip), not the VAE decoder. The other 4 are T=22 (153f) or 2x4-only (s4_res 147,136,120).
So the old s0ups arms on the VAE decode test would never have hit their key, and s4res is not production on 4x8.
New jobs (tmp/blx03/t119/test_t119_4x8.py, one broker job each, full (4,8) mesh, no submesh):
- vae: halo-off reference decode at 1088x1920 (recorded to /var/tmp/fasth3/vae_ref), then the t48 default
  decoder eager + traced, VAE_REF gate vs that reference (mesh 4,8 seams). Gives the 4x8 production decode time.
- ups:A/B x3: latent upsampler, t48 (128,128,1,2,4) vs each of (64,128,3,2,4), (128,64,3,2,4), (128,128,3,2,2);
  per arm: forward_device and initial_conv timing, PCC/PSNR vs fp32 diffusers, PSNR in +-2 px bands at chip seams.
Driver: driver119.sh (STEPS, drop -> one rerun, second drop skips). Results: /var/tmp/fasth3/t119/run_*.log,
grep T119_UPS / T119_VAE / VAE_REF / T119_CONV.


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

Drop log (2026-10-06): 02:31:46 UTC, broker job 224 (smarton, t141 run_e2e.sh, not t119): chips 8-15 (tray 2) left the PCIe bus. glx_reset at 02:40:52 UTC left 32/32 off-bus; host power-cycle held off. The t119 driver (pid 28540, started 02:42:51) waits for health and has submitted nothing yet.
Next run: grep T119_UPS / T119_VAE / VAE_REF / T119_HIT / T119_EXIT in /var/tmp/fasth3/t119/run_*.log on blx03, pick winners, commit to t48 if anything changes.

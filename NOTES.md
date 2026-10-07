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

## 2026-10-06 08:47 UTC: moved to the blx03 serial runner (#153)
Legacy driver119 (pid 28540) died in the 03:42 power-cycle and is gone. Staged src at /var/tmp/fasth3/t119/src
matches 4e3d3aee651 (run119.sh, test_t119_4x8.py, ab119.py diffed equal). Queued on the runner, in order
(positions 8-11 behind t140 x4, t155 x2, t136):
t119-vae-r1, t119-ups1-r1, t119-ups2-r1, t119-ups3-r1 (specs as in tt-project/state/runs/603/migration.md #127).
Wait on: ssh g14blx03 'bash ~/fasth3/runner/probe.sh t119-ups3-r1'. On wake with no marker: ssh g14blx03 bash ~/fasth3/runner/runner-start.sh.
Retries get a new ID (-r2) with the same CONFIG. Results: done/<ID>.done (log=), grep T119_UPS / T119_VAE / VAE_REF / T119_HIT.
Device state at queue time: broker in hold-deadline-escalate (glx_reset); tray 2 (chips 8,9,12,13) dropped in every
t140 job 05:22-08:46 UTC (broker jobs 311, 331, 351, 371, 389), each followed by a power-cycle. Expect a long wait.

## 2026-10-07 10:36 UTC: moved to blx01 (#127 update 10:33)
blx03: all four t119 runner specs (vae, ups1, ups2, ups3 -r1) moved from queue/ to /var/tmp/fasth3/runner/parked
(all four were still queued; parking only ups3 would have let the other three run twice).
blx01: /var/tmp/fasth3/t127 (driver.sh, run127.sh, setup.sh, test copy); reuses #135's lean t48 tree
/var/tmp/fasth3/t48 @ bf7db12a149 (test copied into its tmp/t127). Full (4,8) mesh, LTX_CONV3D_BLOCKING_MESH=4,8.
The vae job is dropped: #135 (jobs 773/774) already timed the 4x8 traced VAE decode (545.5 ms) on this tree, and
the only re-picked key on the 4x8 path is ups_initial (latent upsampler), so the VAE blockings are not under test.
Jobs (one A/B each, arms base/cand/base): u1 vs 64,128,3,2,4; u2 vs 128,64,3,2,4; u3 vs 128,128,3,2,2.
u1 = broker job 775, 87.7 s, rc 0:
  base 128,128,1,2,4: full med 144.5 ms, initial_conv med 0.661 ms, PSNR 48.48 (seams 48.07/48.20)
  64,128,3,2,4:       full med 141.2 ms, initial_conv med 0.671 ms, PSNR 48.40 (seams 47.99/48.12)
  base again:         full med 137.0 ms, initial_conv med 0.710 ms
  -> initial_conv is <0.5% of the upsampler; arm deltas are inside the base's own drift.
Wait: ssh blx01 test -e /var/tmp/fasth3/t127/driver.marker ; results grep T119_UPS /var/tmp/fasth3/t127/run127_u?.log
(summary.txt). Then: delete /var/tmp/fasth3/t127 and /var/tmp/fasth3/t48/tmp/t127 (nothing else was created).

## #127 result (blx01 4x8, LTX_CONV3D_BLOCKING_MESH=4,8, t48 tree bf7db12a149, 2026-10-07)
Broker jobs 775 (u1), 776 (u2), 779 (u3), all rc 0, no drops. Arms base/cand/base, medians.

| job | arm (ups_initial) | initial_conv med ms | upsampler med ms | PSNR / seam rows / cols dB | output vs base |
|---|---|---|---|---|---|
| u1 | base 128,128,1,2,4 | 0.661 / 0.710 | 144.5 / 137.0 | 48.48 / 48.07 / 48.20 | - |
| u1 | 64,128,3,2,4 | 0.671 (+0.01) | 141.2 | 48.40 / 47.99 / 48.12 | differs |
| u2 | base | 0.630 / 0.653 | 225.8 (noisy) / 138.5 | 48.48 / 48.07 / 48.20 | - |
| u2 | 128,64,3,2,4 | 0.575 (-0.07) | 187.2 | same | identical |
| u3 | base | 0.728 / 0.655 | 143.9 / 137.9 | same | - |
| u3 | 128,128,3,2,2 | 0.713 (~0) | 138.2 | same | identical |

Base repeats vary 0.63-0.73 ms on initial_conv and 137-226 ms on the upsampler, so every
delta is inside noise (best: u2 -0.07 ms on a <1 ms layer, ~0.01% of e2e). No winner;
t48 keeps 128,128,1,2,4. VAE arm not rerun: #135 already timed 4x8 VAE decode on this tree.

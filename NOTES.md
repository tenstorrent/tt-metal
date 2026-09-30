# t19 notes: conv vs DiffVAE A/B on real LTX-2.5 latents (seeds 0,1,2)

Branch ttp/t19-conv-vs-diffvae-ab-ltx25 (base 3c13c445c79 = t16's LTX_SEEDS commit on the t36 line).
Scripts: tmp/t19/ (committed with -f; tmp/ is gitignored).

Device (blx03): python-only worktree ~/fasth3/t19 (detached 3c13c445c7; build_Release, build, runtime, _ttnn.so symlinked
to ~/fasth3/tt-metal). Detached driver ~/fasth3/t19/tmp/t19/drive19.sh, log ~/fasth3/drive19.log (JOB=<id>, ends DRIVE19_DONE).
It waits for drive16.sh (t16 seeds5, DiffVAE seeds 0-4, same commit, same default prompt) and any other smarton job, then queues
`W=~/fasth3/t19 bash tmp/blx03/run25.sh t19_conv3 LTX25_DIFFVAE=0 LTX_SEEDS=0,1,2 LTX_FRESH_PROMPTS=0 LTX_DUMP_LATENTS=/var/tmp/fasth3/t19/lat`.
Dumps: lat.gen0 (capture, seed 0), gen1 (replay, seed 0), gen2..4 = seeds 0,1,2. Output mp4s: ~/fasth3/out/ltx25_1080p_6s/t19_conv3/.

Next, once `ssh g14blx03 grep -q DRIVE19_DONE fasth3/drive19.log`:
1. Check the job passed (run.log ends " passed").
2. `tmp/t19/fetch.sh`: latents, mp4s, 3 s stills, determinism check, and full-frame device conv vs t16 device DiffVAE (ffmpeg PSNR/SSIM).
3. Look at stills, pick a face crop (T=3, 14x20 latent = 448x640 px) and a texture crop (T=3, 8x16) per seed.
   `tmp/t19/crop.sh <seed> face t0 h0 w0 3 14 20` (~1 min each on CPU). Then `python tmp/t19/sanity.py <crop_dir> <conv mp4> t0 h0 w0`.
4. Write FINDINGS.md, copy mp4s/stills to tt-project/baselines/t19/, clean up: blx03 /var/tmp/fasth3/t19,
   blx03 out/ltx25_1080p_6s/t19_conv3, blx03 worktree ~/fasth3/t19 (git worktree remove), local tmp/t19/data (latents, crop .pt).

## 2026-09-30 20:45 (after g15blx02 reboot)
Driver never submitted: ~/fasth3/drive19.log is empty and no drive19 process survives. No /var/tmp/fasth3/t19 dumps exist.
blx03 broker is HELD (degraded) since 13:37 PT: t41's job 904 was broker-killed with "chips left PCIe", 8/32 chips
(8-15) off the bus, bridge-reset and glx_reset attempts (jobs 906-910) failed. No submission while it is held.
t16's ref_dv145 has no latents (raw/ holds only run.log), so the conv run is still needed for the CPU crop A/B.
Next: once `tt-device-mcp status 1` on blx03 shows no HELD row and no other smarton job, relaunch drive19.sh
(setsid nohup, log ~/fasth3/drive19.log). drive16.sh is done, so busy() only waits on other smarton jobs.

## 2026-09-30 13:50 PT (attempt wake, probe passed but device still HELD)
blx03 still HELD (degraded): 8/32 chips off the bus, broker glx_reset 910/912 failed. Nothing submitted.
drive19.sh busy() now also waits while a broker recovery row runs or the latest [broker]hold row is HELD
(and no longer waits on drive16). Relaunched detached on blx03: pid 122727, log ~/fasth3/drive19.log.
It submits by itself once the hold ends and no other smarton job is running or queued.
Wake check: DRIVE19_DONE in the log, or the driver is gone (e.g. blx03 rebooted) -> check state, relaunch if needed.

## 2026-09-30 14:12 PT (both hosts rebooted)
g15blx02 and blx03 rebooted ~14:05 PT. blx03 broker power-cycled, hold 919 ended, startup 920 saw all 32 chips, fabric-check 921 passed.
No dumps exist yet. Relaunched drive19.sh detached on blx03 (pid 14880, log ~/fasth3/drive19.log). Same next steps as above.

## 2026-09-30 (run 178 wake)
Woke on probe, but g14blx03 is unreachable (ping 100% loss, ssh "No route to host"): likely down or rebooting again.
Driver state, dumps and broker state unknown. Nothing submitted, nothing changed on blx03.
Next: once blx03 answers ssh, check ~/fasth3/drive19.log and `tt-device-mcp status 1` (HELD?). If DRIVE19_DONE, follow steps 1-4.
If the driver is gone and no dumps exist, relaunch drive19.sh detached (setsid nohup, log ~/fasth3/drive19.log).

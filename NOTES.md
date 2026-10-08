# t243 notes (notes branch only)

Device A/B of DIFFVAE_S5_PACKED_LANES=1 (b8c2b3403f8, branch ttp/t242-..., pushed to origin) on blx01.

- Step 0: t242 push done (b8c2b3403f8 on origin/ttp/t242-diffvae-next-lever-fused-stage-5-blocks-).
- Scaffolding: tt-project/t243/ (copied to blx01 /var/tmp/fasth3/t242/drv). decode242.py = t240's decode240.py
  (matches the t48 @94958f7acee base; t241's decode241 needs afed2273e01) + md5 of device-noise decodes.
- 2026-10-08 01:30 UTC: driver.sh submitted blx01 broker job 886 (-t 330), then died with its ssh session
  (no marker). 01:32 UTC: score.sh (RESUME_JOB=886) waits for 886 and scores; writes drv/driver.marker.
- Next: `ssh g15blx01 cat /var/tmp/fasth3/t242/drv/driver.marker`; read drv/driver.log (timings, cmp lines),
  out/cmp_{def,packed}.json vs #214 refs. Pass: PCC >= 0.9999, PSNR within 0.5 dB of 55.1-55.6 dB.
  If 886 ended in a drop (broker-kill/reboot...), rerun driver.sh (launch with `ssh -n ... setsid -f nohup`).

## Result (job 886, completed rc=0; read 2026-10-08 light wake)
- def: 3.771/3.774 s (mean 3.772). packed: 3.510/3.509 s (mean 3.510) -> -0.262 s (-7%).
- vs #214 host-noise refs: def PCC 0.999956/0.999957, PSNR 55.56/55.12; packed PCC 0.999956/0.999957, PSNR 55.51/55.07 (-0.05 dB).
- packed vs def: PCC 0.999958, PSNR 55.7/55.3. Decision: PASS, faster -> flip default on, land on t48, video + still.
- No drops.

## Land (2026-10-08 standard wake)
- Branch ttp/t243-land = origin/t48 @94958f7acee + cherry-pick of b8c2b3403f8 (03b11f85782) + default flip 34a571c5f47
  (DIFFVAE_S5_PACKED_LANES on unless =0; unit test test_packed_lanes_default_on fails without the flip).
- First `ttp push --detach` failed (exit 4) only because I switched the worktree branch while its checks ran
  (test_denoise_trims.py vanished mid-run); it passes on the land tree. Rerun: state/pushes/t243-20261008-013902-514950.json.
- Media: blx01 /var/tmp/fasth3/t242/out/packed/packed_seed{0,1}.mp4 (+ _t3s.jpg stills), side-by-side def vs packed
  frame 72 in tt-project/t243/media/ (frame PSNR 53.3 dB, no visible difference).

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

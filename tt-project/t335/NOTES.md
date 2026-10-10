# t335: conv VAE math fidelity A/B (lever 2 of t332/FINDINGS.md)

The decoder is t48-ltx25-integrated f6547442b30 (Python overlay on the blx01 lean build bf7db12a149) with the
production 4x8 blocking. The weights are the **real LTX-2.5 conv VAE**
(/var/tmp/fasth3/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors, which landed 06:29 under #235). Its
decoder has the same config and the same 86 key shapes as the 2.3 VAE, so it loads directly. The latents are the
5 saved LTX-2.5 latents from diffvae/latents (1x128x19x34x60), giving 145x1088x1920 output.

## Running (2026-10-10)
- Driver on blx01: /var/tmp/fasth3/t335/drv335.sh. Detach dir /var/tmp/fasth3/t335/detach; log t335/drv335.log;
  marker t335/drv335.done.
- Pair 1, `run335.sh hifi2 default HiFi2`: broker job 374 (-t 600; each arm runs in its own process with a 280 s
  timeout). It was submitted at 06:31:54 UTC. The driver was restarted once (to move its files off blx01 /home)
  and adopted 374 (ADOPT=hifi2=374) instead of submitting it again.
- Pair 2, `run335.sh lofi default LoFi`: submitted by the driver after pair 1.
- When a job ends without its T335_EXIT line, the driver counts it as a drop: it waits for broker health and reruns
  the job once. A second drop skips that pair.

## Reading the results
- Timings: out_<tag>/run.log, `[t335] ARM <arm> decode_ms min= median=` lines. Copy: run_<tag>_job<J>.log.
- Quality: t335/cmp.txt has `CMP <arm> vs default seedN: identical= pcc= psnr= ...` lines. Gate: PSNR >= 45 dB and
  PCC >= 0.999 on all 5 seeds. If the md5s in the DECODE lines are identical across arms, the A/B is invalid.
- Timings are relative only: blx01 may be at its 900 MHz clamp (#230).

## Next
If an arm passes the gate: encode mp4s from out_<tag>/<arm>/seed*.yuv (yuv420p 1920x1088 at 24 fps), run VBench
on both arms, and take a still (frame 72). Then delete /var/tmp/fasth3/t335 on blx01.

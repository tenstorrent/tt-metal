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

## Result (2026-10-10, jobs 374 and 377 on blx01, both T335_EXIT=0, no drops)
- **The default is already HiFi2, not HiFi4.** In vae_ltx.py the conv3d compute config uses HiFi4 only for float32
  weights on Blackhole and HiFi2 otherwise; the decoder runs in bf16, so the "HiFi4 default" premise of t332 lever 2
  was wrong. The HiFi2 arm is md5-identical to default on all 5 seeds. That is expected, not a cache artifact: each
  arm ran in its own process, and the LoFi arm changes the md5 through the same env knob.
- Decode (145x1088x1920, 4x8, blx01, clock possibly clamped, so relative only), median of 5 seeds x 2 reps:
  default 591.0 / 590.5 ms (job 374 / 377), HiFi2 594.6 ms, LoFi 585.1 ms. LoFi is 5.4 ms (0.9%) faster, close to
  the run-to-run noise (+-4 ms). conv3d is limited by data movement (t332), so lowering the math fidelity barely helps.
- Quality of LoFi vs default (RGB): PCC 0.999915-0.999916, PSNR 45.28-46.32 dB, so it passes the gate (>= 45 dB,
  >= 0.999), but only just: luma PSNR 43.84-44.88 dB, worst frame 44.71 dB, max abs error 39-52 (out of 255).
  It looks visually identical (results/seed0_f72_default_LoFi_diffx8.jpg: default | LoFi | |diff| x8).
- Verdict: **reject lever 2.** HiFi2 is a no-op and LoFi gives under 1% for a marginal quality pass. VBench was
  skipped on purpose: it could not make a gain of about 5 ms worth adopting.
- Files: results/ (cmp.txt, run logs, driver log, stills). LoFi seed0 mp4: blx01
  /var/tmp/fasth3/t335_keep/seed0_LoFi.mp4. blx01 /var/tmp/fasth3/t335 (8.7 GB) was deleted.

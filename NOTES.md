# t241 — DiffVAE: traced decode (noise as trace input) A/B

Branch ttp/t241-diffvae-next-lever-overlapped-uint8-pull. Base t48 @94958f7acee (3.772 s, blx01 job 882).

## Lever choice
The pixel pull is already uint8 YUV with async DMA from all chips (utils/yuv_d2h.py), ~98 ms in the t240 profile,
so an overlapped pull can save <= ~50 ms. Tracing is the bigger lever (host dispatch of thousands of ops).

## Code (afed2273e01)
- stage5.forward takes `x_bands` (embedded noise) from the caller; `stage5.noise_bands()` draws it (device randn or host x_t).
  Caller bands are never freed inside (`_brick_activation(release_input=False)`), since they are trace inputs.
- decode_device(raw, x_bands, t, h, w): trace key (c,t,h,w), no seed -> one capture per shape serves every seed,
  and the traced decode accepts host noise (scorable against #214 refs).
- `DiffVAEDecoder.stage5_grid(t,h,w)` (checked on CPU: 1080p latent (19,34,60) -> (145,272,480)).
- DIFFVAE_TRACED=1 (opt-in): forward decodes eagerly on the first call at a shape, traced after.

## Device job
blx01 driver /var/tmp/fasth3/t241/drv/driver.sh (copied in tt-project/t241/), broker job 884 (started 2026-10-08 00:58 UTC),
-t 330, trace_region_size 400 MB. Arms def (eager) and traced, seeds 0,1 timed twice, host-noise seeds 0,1.
Marker: /var/tmp/fasth3/t241/drv/driver.marker. Log: driver.log; out: /var/tmp/fasth3/t241/out (run.log, cmp_*.json).

## Next step
Read driver.log + cmp_def/cmp_traced/cmp_traced_vs_def.json. Expect traced host-noise yuv identical to def.
If faster and identical/neutral: flip DIFFVAE_TRACED default to 1 (keep the eager first call), land code commits
on ttp/t48-ltx25-integrated via a -land branch (`ttp push --detach`), notes via `ttp push --own --detach`.
If capture fails (trace region too small / OOM), read the traceback in out/run.log.

## Result job 884 (2026-10-08 01:05 UTC, blx01, no drops)
Eager (default): 3.776/3.772, repeat 3.774/3.773 s -> mean 3.774 s.
Traced (DIFFVAE_TRACED=1): first 4.414 (capture) / 3.793, repeat 3.790/3.800 s -> warm ~3.795 s, +20 ms slower.
Quality: traced output md5-identical to eager (device noise and host noise). Host-noise vs #214 refs: PCC 0.999956/0.999957, PSNR 55.56/55.12 dB.
Decision: tracing gives no gain (dispatch is not the bottleneck; device-bound). Stays opt-in, default off, not landed on t48.
Next lever: fused stage-5 blocks / 2-D stage-5 split default (DIFFVAE_S5_2D) per PLAN.md; device-side kernel time is what remains.

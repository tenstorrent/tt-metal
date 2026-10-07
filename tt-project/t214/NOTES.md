# t214: DiffVAE reference latents + unoptimized decodes (PLAN.md section 4 steps 1-3)

Box: blx01 (g15blx01). blx03 was in a tray-2 reset loop and has no 5e4e0cd643a tree. Data: /var/tmp/fasth3/diffvae on blx01.
Scripts here are deployed to blx01:/var/tmp/fasth3/diffvae/scripts/. Driver: driver.sh (setsid nohup), log driver.log, marker driver.marker.

- Job A run_lat.sh: t48 5e4e0cd643a overlay (/var/tmp/fasth3/t208/tree), defaults (S2 2 steps, conv 2.3 VAE), DEFAULT prompt,
  seeds 0-4 warm replays, LTX_DUMP_LATENT -> latents_raw. map_latents.py picks each seed's dump -> latents/seed{s}.pt.
  No t48 change needed: LTX_DUMP_LATENT already exists, so nothing was pushed.
- Job B run_dec.sh/decode_ref.py: DiffVAEOptions.production(), 4x8 FABRIC_1D_RING, host noise
  torch.randn(Generator().manual_seed(seed)) injected into stage5, yuv output -> ref/ref_dvx_seed{s}.yuv (1920x1088 yuv420p),
  decode_times.json (one warm-up decode first, then timed per seed). Resumes per seed.
- Media: ref/*.mp4 (libx264 crf 12), stills at 3 s (*_t3s.jpg), stills of job A conv mp4s, MD5SUMS.

Timeouts: -t 600 for both (unmeasured). Size to measured +50% on reruns.

Next step on wake: read driver.marker/driver.log; copy latents + decode_times.json + logs + stills to
tt-project/baselines/ltx25_1080p_6s/diffvae_latents/ on g15blx02; copy latents/yuv/mp4/stills to blx03 /var/tmp/fasth3/diffvae/.

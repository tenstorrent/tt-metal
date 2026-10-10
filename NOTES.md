# t365 notes (conv VAE: make the #362 unpatch fusion reach the default LTX-2.5 e2e path)

## Finding (step 1)
- Standard e2e (test_pipeline_ltx_distilled.py::test_pipeline_distilled, LTX_VERSION=2.5 LTX25_DIFFVAE=0) passes
  output_path, so pipeline_ltx_distilled decodes with output_type="yuv" (LTX_YUV_EXPORT defaults 1).
- LTXVideoDecoder.forward takes the fused YUV branch only if LTX_FUSE_YUV_OUTPUT=1 or LTX_TRACE_YUV_OUTPUT=1; both
  defaulted 0 and neither the test nor the pipeline sets them. So e2e ran the unfused path: reshape + 8D permute +
  reshape to BCTHW, then fast_device_to_host_yuv. The #362 fusion (LTX_VAE_FUSE_UNPATCH) never ran in e2e.

## Change (step 2)
- Branch ttp/t365-... reset onto origin/ttp/t48-ltx25-integrated 20b40f459a9.
- Code commit 53da67e9c61: LTX_FUSE_YUV_OUTPUT defaults to 1; test_vae_ltx_fuse_unpatch_ab.py arms over
  LTX_FUSE_YUV_OUTPUT too (arm label y<0|1>r<0|1>) and asserts the decoder's flags match.
- Side effects: warm-up decode now uses yuv (pipeline_ltx._warmup_decode keys on fuse_yuv_output); defer_yuv
  (opt-in LTX_AUDIO_OVERLAP=1) is ignored on the fused path (returns the array; callers accept that, like DiffVAE);
  multi-host and non-(0,1) mesh axes fall back to the unfused path as before.

## A/B (step 3), blx01
- Driver /var/tmp/fasth3/t365/drv365.sh (copy here): bundle -> worktree /var/tmp/fasth3/t365/b at 53da67e9c61,
  Release build, then one broker job -t 240 run365.sh (arm y0 then y1, LTX_VAE_FUSE_UNPATCH=1, real 2.5 conv VAE,
  diffvae/latents/seed0-4, md5/PCC/PSNR). Started 2026-10-10 ~19:38 UTC via
  `ttp detach --remote g15blx01 --dir /var/tmp/fasth3/t365/detach drv365`.
- Probe: `ttp detach --check --host g15blx01 /var/tmp/fasth3/t365/detach/drv365`.
- Results: /var/tmp/fasth3/t365/drv365.done, run_job<id>.log, arm{0,1}_job<id>.log.

## Next step
1. Read drv365.done + run_job<id>.log: identical per seed, decode_s min per arm.
2. Pass -> `ttp checks`, then land: `git switch -c ttp/t365-land origin/ttp/t48-ltx25-integrated`,
   cherry-pick 53da67e9c61, `ttp push --detach`.
3. Clean blx01: `git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t365/b; rm -rf /var/tmp/fasth3/t365`.

## Result (blx01 broker job 477, 2026-10-10 19:38-19:41 UTC, no drops)
- y0 (old default, unfused) vs y1 (new default, fused YUV + unpatch): md5-identical on all 5 seeds
  (pcc 1.000000, psnr inf, maxabs 0). decode_s min 0.6587 -> 0.5380 s (-121 ms, -18%), 900 MHz clamp, relative only.
- Log: tt-project/t365/run_job477.log.

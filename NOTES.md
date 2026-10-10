# t362 notes (conv VAE: fuse conv_out unpatchify into RgbToYuv)

## State (2026-10-10 12:00 PDT)
- Code commits on ttp/t362-conv-vae-fuse-conv-out-4x4-unpatchify-in (base origin/ttp/t48-ltx25-integrated ba3636df5ee):
  - a4996c6a3d3 rgb_to_yuv: read patchified input; fuse the LTX VAE conv_out unpatchify (flag LTX_VAE_FUSE_UNPATCH, default 0)
  - 695a102dc1c ltx vae: add the conv_out unpatchify fusion A/B test
- Deviation from the spec: conv_out's 3-channel groups are not tile-aligned and RgbToYuv needs T innermost, which
  the conv3d writer cannot emit (T is its outer loop: 2-byte scattered writes). So RgbToYuv's reader reads conv_out's
  (1,T,H/4,W/4,48) output directly (input_patch_size=4), staging each frame's patch pages and scattering them into its
  T-stick scratch. The reshape + permute are gone either way; clip moves onto conv_out's output (elementwise, same bits).
- blx01 driver: /var/tmp/fasth3/t362/drv362.sh (copy in tt-project/t362/), started with
  `ttp detach --remote g15blx01 --dir /var/tmp/fasth3/t362/detach drv362`. It builds 695a102dc1c in /var/tmp/fasth3/t362/b,
  then one broker job (-t 600) run362.sh: unit test test_rgb_to_yuv_patch_input.py (bit-exact, 3 shapes incl. the
  145x68x60 per-chip shape), then arm 0/1 of test_vae_ltx_fuse_unpatch_ab.py (real 2.5 conv VAE, diffvae/latents/seed0-4,
  LTX_FUSE_YUV_OUTPUT=1), then md5/PCC/PSNR.
- Results: /var/tmp/fasth3/t362/drv362.done (DONE job=<id>:<status>), run_job<id>.log, arm{0,1}_job<id>.log, unit_job<id>.log.

## Job 459 (blx01, 2026-10-10 19:05-19:08 UTC): failed in arm 1, no drop
- unit test PASSED (3 shapes); arm 0 (unfused) ran: decode_s 0.5759 0.5606 0.5695 min 0.5606 (900 MHz clamp, relative only).
- arm 1 died: TT_FATAL rgb_to_yuv_device_op.cpp:60 'Padded input is not supported (logical [1,145,68,60,48] vs padded
  [...,64])'. conv3d rounds C_out up to a tile. The reader already reads pages at aligned_page_size and uses only the
  first 48 channels, so b5d8b7f567e relaxes the check (patchified input may pad the last dim) and adds a padded unit case.
- Logs moved to /var/tmp/fasth3/t362/old459/.

## Rerun (started 2026-10-10 ~19:20 UTC)
- drv362.sh now builds b5d8b7f567e (git checkout in the existing t362/b worktree, incremental build).
- `ttp detach --remote g15blx01 --dir /var/tmp/fasth3/t362/detach drv362b`; probe:
  `ttp detach --check --host g15blx01 /var/tmp/fasth3/t362/detach/drv362b`.

## Next step
1. Read drv362.done and run_job<id>.log (T362_EXIT, identical=True per seed, AB decode_s min per arm).
2. If identical on all 5 seeds: separate commit flipping LTX_VAE_FUSE_UNPATCH default to 1 (vae_ltx.py, and A/B test default).
3. `ttp checks`; land: `git switch -c ttp/t362-land origin/ttp/t48-ltx25-integrated`, cherry-pick the code commits
   (not the notes commit), `ttp push --detach`.
4. Clean blx01: rm -rf /var/tmp/fasth3/t362 after `git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t362/b`.

## Job 466 (blx01, 2026-10-10 ~19:14-19:18 UTC): passed, no drop
- unit test PASSED (4 cases incl. padded-channel 64). Both arms md5-identical on seeds 0-4 (pcc 1.0, psnr inf, maxabs 0).
- decode_s min: arm 0 (unfused) 0.6214, arm 1 (fused) 0.5156: -106 ms (-17%), same job, 900 MHz clamp, relative only.
  Arm 0 was 0.5606 in job 459, so the gain vs that is ~45 ms, matching the #349 profile (45.9 ms). Noise is large.
- 0f2ca651c33 flips LTX_VAE_FUSE_UNPATCH default to 1. Landing: ttp/t362-land (cherry-picks a4996c6a3d3 695a102dc1c
  b5d8b7f567e 0f2ca651c33 onto origin/ttp/t48-ltx25-integrated), ttp push --detach.

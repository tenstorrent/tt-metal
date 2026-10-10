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

## Next step
1. Read drv362.done and run_job<id>.log (T362_EXIT, identical=True per seed, AB decode_s min per arm).
2. If identical on all 5 seeds: separate commit flipping LTX_VAE_FUSE_UNPATCH default to 1 (vae_ltx.py, and A/B test default).
3. `ttp checks`; land: `git switch -c ttp/t362-land origin/ttp/t48-ltx25-integrated`, cherry-pick the code commits
   (not the notes commit), `ttp push --detach`.
4. Clean blx01: rm -rf /var/tmp/fasth3/t362 after `git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t362/b`.

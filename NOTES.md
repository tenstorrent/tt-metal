# t337 notes (LTX-2.5 conv VAE)

Done
- blx01: /var/tmp/fasth3/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors, 1452269922 B,
  sha256 685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d (matches HF LFS etag, repo rev 2356ce76915d).
- Upscaler x2 1.0: HF sha256 eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8 is the same blob t48 already
  uses (MLPerf snapshot 28dac7acdc). No fetch.
- 2.5 conv VAE vs 2.3 monolith VAE: same config, 170 tensors, all byte-identical. Only difference: no `vae.` key prefix.
  The f6547442b30 loader then fails: #333 job 375 (blx01) died with "ValueError: missing Torch state keys: conv_in.weight, ...".
- Code 63e54d98036 (vae_key_map + LTX25_VIDEO_VAE resolution, 2.3 fallback only via LTX25_VAE_FALLBACK_23=1, unit test
  models/tt_dit/tests/unit/test_ltx25_conv_vae.py 5 passed 1 skipped) landed on origin/ttp/t48-ltx25-integrated with ttp push.

Running (2026-10-10 06:38 UTC)
- blx01 driver /var/tmp/fasth3/t337/drv337.sh (ttp detach, pid 2908428): builds t337/b at 63e54d98036, then broker jobs
  (-t 570, one at a time) conv25 then conv23 (md5 check). Log drv337.log; marker drv337.done; job logs run_<arm>_job<id>.log;
  videos out_<arm>/ltx_av_fast_*.mp4 + _t3s.png.
- Probe: ttp detach --check --host blx01 /var/tmp/fasth3/t337/drv337

Left
- Quote conv25 timing table verbatim (+ commit, command, job id), video + still, compare conv25 vs conv23 md5/decode time.
- Verify blx03 copy /var/tmp/fasth3/ltx25_vae/ltx-2.5-video-vae-conv-bf16.safetensors once .part is gone (still .part at
  06:39, 957 MB); never touch the .part.
- Cleanup blx01: t337/b, t337/jit, fetch.sh, cmp.*, vae.log (keep model files). Report footprint.

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

Run 1 (drv337, done 07:03 UTC): both arms FAILED on the script's own pytest --timeout=540 (no drop).
- conv25 job 382: cold JIT (0/2929 hits) + cold weight load (236 s, no TT_DIT_CACHE_DIR); timed out in warmup. The 2.5
  conv VAE config loaded fine from the split file (no key error).
- conv23 job 384: warmup 391 s, of it the audio-decode warmup 262 s (vocoder kernels still cold). gen#0 finished
  (table: VAE decode 1.80 s, Total 34.90 s, clamp) and the gen#1 replay was cut off. gen#0 video kept at
  out_conv23_r1/ltx_av_fast_1920x1088_0.mp4 md5 1340be4d394e1a37575d6cc275a4a113.
- No leftover processes after either job. blx01 /var/tmp/fasth3 footprint 137 G (no room for a DiT cache).

Run 2 (drv337b, done 07:51 UTC, blx01, 900 MHz clamp: relative only). Both PASSED, no drop, no leftover processes.
- Commit 63e54d98036 (ttp/t48-ltx25-integrated), standard test unmodified (test_md5 d9a26aaf...), command:
  pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled[blackhole-bh_4x8sp1tp0_ring-True]
  with LTX_VERSION=2.5 LTX25_DIFFVAE=0 (8+3, 1088x1920, 145 f, 24 fps, seed default). Driver: notes/t337/run337.sh.
- conv25 job 391 (LTX25_VIDEO_VAE = /var/tmp/fasth3/ltx25_vae/... 2.5 split file): replay Total 7.33 s, VAE decode 0.60 s.
- conv23 job 394 (2.3 monolith): replay Total 7.31 s, VAE decode 0.58 s.
- All four mp4s md5 1340be4d394e1a37575d6cc275a4a113 (bit-identical, as expected: the weights are byte-identical).
- Kept video: blx01 /var/tmp/fasth3/t337/out_conv25/ltx_av_fast_1920x1088_1.mp4, still notes/t337/conv25_t3s.jpg (sane).
- #333 job 388 (VAE 0.70 s) ran on a different commit; same-commit A/B above shows no decode-time difference.
- Cleanup done 07:56 UTC: removed t337/b (worktree of t48, 2.9G), t337/jit (6.1G), extra outputs, fetch/cmp scratch.
  t337 now 14M; blx01 /var/tmp/fasth3 133G; df / 52%. Model files kept.
- blx03 copy: left alone per update #243 (separate task).

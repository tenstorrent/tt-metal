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

Running (2026-10-10 07:07 UTC)
- drv337b on blx01 (ttp detach --remote, pid 3141740): verified the user's copy /var/tmp/fasth3/ltx25_vae/... (size+sha OK,
  07:07:42), then conv25 (LTX25_VIDEO_VAE = that copy) and conv23, -t 570, now with a warm JIT. Waits behind #333 job 386.
  Marker drv337b.done. Probe: ttp detach --check --host blx01 /var/tmp/fasth3/t337/drv337b

Left
- Quote conv25 timing table verbatim (+ commit, command, job id), video + still, compare conv25 vs conv23 md5/decode time.
- Verify blx03 copy /var/tmp/fasth3/ltx25_vae/ltx-2.5-video-vae-conv-bf16.safetensors once .part is gone (still .part at
  06:39, 957 MB); never touch the .part.
- Cleanup blx01: t337/b, t337/jit, fetch.sh, cmp.*, vae.log (keep model files). Report footprint.

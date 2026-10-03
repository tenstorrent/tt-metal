# t97 notes (LTX_VAE_EXACT_SHARD)

- Code: d22cef25303 (opt-in `_reshard_exact_hw` after each depth-to-space; CPU test
  test_vae_ltx_exact_shard_ref.py 8 passed). A/B harness: deb30e3191e + cdef9a014ea (driver health fix).
- blx03 job 454 (submitted 2026-10-03 06:17 UTC): run97.sh -> test_vae_ltx_exact_shard_ab.py on a 2x4 submesh
  of the full mesh, 544x960/145f, LTX_CONV3D_BLOCKING_MESH=4,8, fused YUV, eager. One warmup per arm, then
  3 interleaved timed decodes per arm (pad, exact).
- blx03 dropped at ~05:02-05:16 UTC during ltx-host job 438 (not ours); the broker recovered and verified
  health at 05:16. Per the 07:38 rule I retried.
- Logs on g14blx03: /var/tmp/fasth3/t97/driver.log (marker `T97_DRIVER_DONE <stage> <rc>`; rc 9 = drop
  during OUR job -> stop all device work and report), /var/tmp/fasth3/t97/run97.log (`AB arm=` lines with
  min and md5, `AB_CMP exact_vs_pad identical=... delta_min_ms=`), yuv_{pad,exact}.pt.
- Next: read run97.log. Pass = identical=True and delta_min_ms < 0 (expected -40 to -50 ms vs 445 ms).
  Then delete /var/tmp/fasth3/t97/{src,yuv_*.pt} on blx03, push the branch (no PR), write result.json.

## Result (job 454, 2026-10-03 06:18 UTC, exit 0, no chip drop)
- pad   decode_s 0.5166 0.5158 0.5225, min 0.5158
- exact decode_s 0.4672 0.4701 0.4639, min 0.4639
- delta min -51.9 ms (-10.1%); YUV identical=True, max_abs_diff=0, md5 18e86950fb2021b62023ca2ad1073d41 both arms.
- Absolute times are above the 445 ms #96 baseline: UMD logged AICLK clamped at 1150 MHz (expected 1350) on this run.
  The A/B is interleaved on the same run, so the delta stands; expect about -44 ms at 1350 MHz if it scales with clock.
- Cleaned /var/tmp/fasth3/t97/{src,yuv_*.pt} on blx03 (logs kept, 44 KB).
- Next: make LTX_VAE_EXACT_SHARD the default once a 4x8 1080p eval is allowed (the 4x8 path is untested on device).

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

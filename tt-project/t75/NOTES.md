# t75 — 2x4 conv decode check of t48 tip a613d669eef (all default folds stacked)

blx03 broker job **038** (submitted 2026-10-01 11:13 UTC, timeout 1300 s): `bash ~/fasth3/t75/run75.sh`.
Log: `g14blx03:~/fasth3/t75/run75.log` (also `/var/log/tt-device-broker/2026-10-01_111346_038.log`).
Status: `ssh g14blx03 tt-device-mcp status -j 038`.

Harness: test_vae_ltx_fold_time_pad_ab.py from the t48 tree (~/fasth3/t48, built at a613d669eef), full 4x8 mesh
opened then create_submesh(2,4), 544x960/145f, real latent lat.gen0.pt, LTX_FUSE_YUV_OUTPUT=1, hostfmax 1150,
1 warmup + 3 timed decodes per arm. Same settings as #46 / #65 (run60.sh).
- arm def: tree defaults -> /var/tmp/fasth3/t75/yuv_t1w1.pt
- arm ref: LTX_VAE_FOLD_TIME_PAD=0 LTX_VAE_FOLD_W_MASK=0 + overlay vae_ltx_pre58.py (t48 vae_ltx.py with #58 reverted)
  -> yuv_t0w0.pt
Pass: `T75_CMP ... identical=True`, `T75_EXIT=0`, def min decode ~2.05 s or less.
Prior numbers: #65 t1w0 2.2915 s, t1w1 2.2224 s (jobs 033/034, pre-#58); #58 output path -170 ms (job 030).

Next: grep `AB arm=\|T75_` in the log, check broker log for drops, copy log here, then
`rm -rf ~/fasth3/t75 /var/tmp/fasth3/t75` on blx03.
If a chip drop/reboot/fabric failure hit during job 038: stop all device work, kill our queued jobs, report.

## Result (2026-10-01 11:20)
Job 038 completed, exit 0, runtime 138.5 s. Post-job broker health gate: healthy, no drops/reboot (uptime 3:10).
- def (all folds + #58): decode 2.0525 / 2.0558 / 2.0554 s, min 2.0525 s
- ref (all off, pre-#58): 2.3374 / 2.3401 / 2.3302 s, min 2.3302 s
- YUV output bit-identical (max_abs_diff=0, shape 145x816x960 uint8)
Saves 278 ms/decode (-11.9%). Log: run75_job038.log. blx03 t75 dirs removed.

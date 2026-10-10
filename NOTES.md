# t373 notes (conv VAE: measure and cut the non-device tail)

## State (2026-10-10 14:30 PDT)
- Code: e823d13b945 = t48 90ed8257bac + timing_tree spans in vae_ltx.py forward (latent upload, decode (device),
  unpatch + rgb->yuv (device)) and yuv_d2h._yuv_planar_d2h (yuv readback (3 planes), yuv host assemble (C++ planar
  concat | torch fallback)). Opt-in: TT_DIT_STAGE_TIMING=1; LTX_PERF_BREAKDOWN=2 expands the VAE decode row in the
  standard table; TT_DIT_STAGE_LOG=1 prints per-span ms. Host test: models/tt_dit/tests/unit/test_yuv_d2h_timing.py (2 pass).
- blx01 driver (detached 14:27 PDT): /var/tmp/fasth3/t373/drv373.sh (copy in tt-project/t373/).
  Builds Release in /var/tmp/fasth3/t373/b, then broker jobs -t 570: w = plain standard run (cold JIT fill),
  m = same with timing env. Logs: /var/tmp/fasth3/t373/{drv373.log,build.log,run_<tag>_job<id>.log}; marker drv373.done.
  retry_when: ttp detach --check --host g15blx01 /var/tmp/fasth3/t373/drv373

## Run 1441 (2026-10-10 14:55 PDT)
- drv373 jobs 523 (w) and 525 (m) both failed on pytest-timeout 540 s: cold JIT. 523 compiled through warm-up stage 1
  (step 1 took 94 s). 525 (JIT 58% hits) finished warm-up in 426 s and was killed 8 s into gen#0. No drop: fabric
  checks 524/526 passed after each. Not a code fault: no timing lines reached (gen#0 never finished).
- JIT cache t373/jit should now be full. drv373b (copy: tt-project/t373/drv373b.sh; ttp detach --remote g15blx01
  --dir /var/tmp/fasth3/t373, pid 824901, 21:54Z) reuses the build and queues m then w at -t 570 behind other
  tenants. Lint rc 0 for both. retry_when: ttp detach --check --host g15blx01 /var/tmp/fasth3/t373/drv373b
  Marker: /var/tmp/fasth3/t373/drv373b.done; logs run_m_job<id>.log, run_w_job<id>.log.

## Run 1445 (2026-10-10 15:30 PDT)
- drv373b done: job 535 (m, timing) and 538 (w, plain) both completed at e823d13b945, blx01 900 MHz clamp (relative only).
  mp4 md5 1340be4d394e1a37575d6cc275a4a113 for all four gens (timing spans bit-identical). No drops.
  Warm gen#1 VAE split (ms, job 535): upload 14.8 | device decode 420.8 | unpatch+yuv device 6.5 | readback 30.9 |
  host assemble (C++ planar concat, HAS_CPP_PLANAR_CONCAT True) 30.3 | TOTAL 509.9. Plain table (538 gen#1): VAE 0.50 s,
  Audio 0.47 s, Total 7.23 s. Excerpts: tt-project/t373/results/job535_m_excerpt.txt, job538_w_excerpt.txt.
- Biggest host slices: readback 31 and assemble 30 ms. wide_rows (ad2792e70b0) needs T<=32 (we have T=145 per shard),
  so it does not apply. Deferred assembly already existed (LTX_AUDIO_OVERLAP=1, opt-in) but the fused-YUV default
  branch ignored defer_yuv. fe19a6634ee threads defer through the fused path (+unit test); bf45db27371 makes
  LTX_AUDIO_OVERLAP default on (=0 restores eager). Unit tests: test_yuv_d2h_timing.py + test_yuv_video_export.py 9 pass.
- drv373c (pid on blx01, ttp detach --dir /var/tmp/fasth3/t373): checks out bf45db27371 in t373/b (py-only diff, build
  reused), jobs dm (timing) then dw (plain headline), -t 570 each. Marker drv373c.done; logs run_dm_job*.log, run_dw_job*.log.
  retry_when: ttp detach --check --host g15blx01 /var/tmp/fasth3/t373/drv373c

## Next step
1. Read drv373c.done; dw md5 must be 1340be4d394e1a37575d6cc275a4a113; compare VAE row dw vs 538 (0.50 s) and dm decode
   TOTAL vs 535 (509.9 ms); check Total and "Video export:" lines (assembly now on the export thread, overlapping audio).
2. Keep if VAE row drops >= 15 ms and md5 same: land fe19a6634ee + bf45db27371 on t48 via a -land branch + ttp push --detach.
   Headline = dw table verbatim.
3. Clean /var/tmp/fasth3/t373 on blx01 at the end (git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t373/b).

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

## Next step
1. Read run_m_job*.log: the VAE decode breakdown rows and the TT_DIT_STAGE_LOG ms lines for gens #1/#2
   (upload / device decode / yuv device / readback / host assemble / other) and "[t373] planar concat" line.
2. Port from ad2792e70b0 only the part that the biggest host slice maps to (wide_rows writer -> device yuv/readback;
   deferred readback -> readback; AVX2 planar concat -> host assemble). A/B in one job, md5 must match, keep if >= 15 ms.
3. Land code commits on t48 via a -land branch + ttp push --detach; headline = plain standard run quoted verbatim.
4. Clean /var/tmp/fasth3/t373 on blx01 at the end.

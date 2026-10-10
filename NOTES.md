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

## Next step
1. Read run_m_job*.log: the VAE decode breakdown rows and the TT_DIT_STAGE_LOG ms lines for gens #1/#2
   (upload / device decode / yuv device / readback / host assemble / other) and "[t373] planar concat" line.
2. Port from ad2792e70b0 only the part that the biggest host slice maps to (wide_rows writer -> device yuv/readback;
   deferred readback -> readback; AVX2 planar concat -> host assemble). A/B in one job, md5 must match, keep if >= 15 ms.
3. Land code commits on t48 via a -land branch + ttp push --detach; headline = plain standard run quoted verbatim.
4. Clean /var/tmp/fasth3/t373 on blx01 at the end.

# t209 — H3 (fasth3 fl2va) 10 s e2e stage timings on blx01

Code: ttp/t209-h3-e2e @9fa26939b35 = fasth3-opt + harness knobs (BASE_VSA_RING_GATHER, BASE_LOAD_ONLY). Pushed.
Config (fastest recorded with OK quality): 10 s, 768x1344 native + upscale 1920x1080, VSA sparsity 0.9,
vsa_ring_sdpa fused_kv gather, 50 steps, 4x8 ring fabric (no mesh graph descriptor override; t161 ran the same way).

## blx01 layout (all under /var/tmp/fasth3)
- t209/b: detached worktree of t48 clone at 9fa26939b35, own Release build (setup209.sh, log t209/build.log)
- t209/driver.sh: build -> fill (BASE_LOAD_ONLY, weight cache, up to 3 tries) -> jit (2 steps, cold) -> time (2-step warmup + seeds 0,1)
  Each step is one <=600 s broker job (-t 600), after the t161/t159 health checks; one drop rerun, second drop skips.
- Logs: t209/driver.log (job ids, drops), t209/out_<tag>/run.log, out_<tag>/fl2va_768p_10s_vsa0.9_ring_fused_kv_50steps/
- Marker: t209/driver.marker "T209_DRIVER_DONE stage=.. rc=.. jobs=.."
- New caches: cache/dit-h3opt (~131 GB expected), cache/tt-metal-cache-h3opt. Remove after the task (followup).

## Started
2026-10-07 18:12 UTC, driver pid 3093335. No device job submitted yet at hand-off (git fetch + build first).

## Next step on wake
`ssh g15blx01 cat /var/tmp/fasth3/t209/driver.marker /var/tmp/fasth3/t209/driver.log`.
rc=0: read out_time/.../seed*_timings.json (+ out_jit for cold/load), copy one mp4 + mid png to tt-project/t209/, report.
Failure: read the failing stage's run.log; fix and restart the driver (it skips a finished build/fill/jit).

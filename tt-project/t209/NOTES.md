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

## Run 2 (2026-10-07 ~19:40 UTC)
Driver 1 ended rc=1. The time job 825 and jit attempt 2 (814) both failed on a code error, not a drop:
`TT_FATAL vsa_ring_sdpa: at most 32 exempt block ids`. 10 s 768p fl2va has 78 exempt tiles (1272 tiles, 1188 candidates,
6 pad; 2058 prefix tokens incl. 2016 keyframe vision). The kernels only OR the ids into a per-row bitmap read from runtime
args (no fixed array; leader args stay well under 341), so the cap was raised to 128 in vsa_sdpa and vsa_ring_sdpa:
ttp/t209-h3-e2e @30a52415629 (pushed). Drops so far: job 795 (chip 13, 18:35 UTC, timeout during AdaLN host build) and
chip 25 at 19:19 UTC after job 814 had already failed on the TT_FATAL. Both were our jobs (t209 jit).
Caches are filled: dit-h3opt weights + AdaLN table; JIT 1.7 GB (encoder side only; DiT/VAE kernels still cold).
Restart: blx01 /var/tmp/fasth3/t209/restart2.sh = incremental rebuild (ok, rebuild2.log) -> driver.sh (jit -> time).
Old results moved to out_time_fatal1, driver.marker.1.
Next on wake: same as above (marker, driver.log, out_jit/out_time). If jit times out at 570 s from cold compile, split further.

## Result (2026-10-07 20:00 UTC, blx01 jobs 829 jit / 831 time, ttp/t209-h3-e2e @30a52415629)
fl2va 768p (1344x768) 243f 10.1 s, 50 steps, vsa0.9_ring_fused_kv, 4x8 tp4 sp8, upscaled to 1080p on host.
| stage | seed0 | seed1 |
|---|---|---|
| Encoder | 1.90 | 1.89 |
| Keyframe encode | 0.35 | 1.98 |
| Denoise (49 fwd, ~1.95 s/fwd) | 95.20 | 95.90 |
| VAE decode | 8.11 | 8.87 |
| Audio decode | 12.31 | 12.28 |
| Upscale 1080p (host) | 6.92 | 7.58 |
| total compute | 124.8 | 128.5 |
| call wall | 144.4 | 128.8 |
Warmup (2 steps, warm JIT) 50.0 s. Cold JIT job 829: 2-step run 186.8 s wall, test 209 s.
Video: tt-project/t209/seed0_1920x1080.mp4, still: tt-project/t209/seed0_1920x1080_mid.png.

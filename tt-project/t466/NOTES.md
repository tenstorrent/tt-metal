# t466 notes (ring SDPA LoFi matmuls for FastH3 Turbo)

Branch ttp/t466-turbo-ring-joint-sdpa-lofi-matmul-fideli, base dbd6d66048f (t460 head).
Code: 6dcde6d7dcc/9113b4b2cd4/6a740564a56 = cherry-picks -x of #57979, #58032, #58223 (clean);
f16f89d0f92 = opt-in MINIMAX_H3_SDPA_MM_FIDELITY (lofi|hifi2|hifi3|hifi4; unset = unchanged), ring path only,
CPU tests test_sdpa_mm_fidelity_minimax_h3.py (9 passed); e02052af0ee = device microbench
test_sdpa_fidelity_timing_minimax_h3.py (one layer, 10 s fl2va shape, random weights, HiFi2 vs LoFi interleaved,
logs "T466 summary: ring_joint_sdpa gain X%, layer gain Y%" and PCC between arms).

## blx01 state (2026-10-13 06:22 UTC)
- /var/tmp/fasth3/t466/b: t422/b tree + `git archive e02052af0ee` overlay, fresh build_Release
  (driver build466.sh, log build.log, exit code in build.rc). Python env: /var/tmp/fasth3/t460/python_env.
- No MiniMax-H3 weights on blx01 (deleted after #463). Microbench needs none.

## Next step
1. Build done (build.rc = 0): write microbench job script under /var/tmp/fasth3/t466 (env like
   tt-project/t460/run460.sh: HOME/TMPDIR/TT_METAL_CACHE under /var/tmp/fasth3, TT_METAL_HOME and
   PYTHONPATH = t466/b, `timeout --foreground -k 10 560 python -m pytest -sv -p no:cacheprovider
   models/tt_dit/tests/models/minimax_h3/test_sdpa_fidelity_timing_minimax_h3.py -k 4x8sp1tp0nl2_ring_is_fsdp0`),
   device-lint.sh it, health check, submit run-bg on blx01's broker (-t 600, unmeasured).
2. Gain < 10% on ring_joint_sdpa: stop (record, keep knob opt-in).
3. Gain >= 10%: e2e A/B needs ~360 GB on blx01 (weights 135 + dit cache 115 + metal cache ~59 + build) vs the
   150 GB cap (memory, after #463) -> ask the user to raise the cap before re-fetching weights.

## 2026-10-13 06:29 UTC (12-12 23:29 PDT): microbench submitted
- build.rc = 0 (build done 06:25 UTC). Job script run466mb.sh + env466.yaml copied to /var/tmp/fasth3/t466
  (JIT cache /var/tmp/fasth3/t466/jit, cold). Spec spec_mb.txt submitted via tt-project/t288/blx01-submit.sh:
  health ok, lint ok (cold-cache warning only), blx01 broker job 103, queued behind ltx-host job 102. -t 600 (unmeasured).
- Log: /var/tmp/fasth3/t466/out_mb/run.log (grep 'T466'); broker log /var/log/tt-device-broker/2026-10-13_062903_103.log.
- Next: read "T466 summary: ring_joint_sdpa gain X%". <10% -> stop (knob stays opt-in). >=10% -> e2e A/B needs weights
  (cap 150 GB on blx01, now 48 GB used) -> ask for cap raise.

## 2026-10-13 06:45 UTC (10-12 23:45 PDT): microbench result (blx01 job 103, e02052af0ee, 900 MHz clamp, relative only)
- One ring SDPA layer, 10 s fl2va shape, 4x8, 3 interleaved rounds (tt-project/t466/mb1_summary.log):
  hifi2: layer best 47.18 ms, ring_joint_sdpa best 39.54 ms; lofi: layer best 40.75 ms, ring_joint_sdpa best 33.37 ms.
  Gain: ring_joint_sdpa 15.6%, layer 13.6%. LoFi vs HiFi2 output PCC 99.90%, CCC 99.75%, RMSE/sigma 6.9%.
  Passes the 10% gate -> e2e A/B is next.
- No drops.

## e2e A/B is waiting on blx03
- Needs MiniMax-H3 (135 GB) + dit-h3hf cache (152 GB) + JIT cache (17 GB): ~305 GB. blx01 holds 49 GB of ours and its
  cap is 150 GB (memory, after #463), so it cannot fit there. The only known copy of the weights and cache is on
  blx03 /var/tmp/fasth3, and blx03 is out (no route since 2026-10-11 22:37Z; g14blx03-device paused, trays #364).
- Plan when blx03 returns: run the A/B on blx03 itself through its serial runner (data already there), arms
  base = t460 head dbd6d66048f build, lofi = same build + MINIMAX_H3_SDPA_MM_FIDELITY=lofi (needs this branch's
  cherry-picks built there). Order: 10 s seed 0 pair, 6 s seed 0 pair, then 6 s seeds 1-4 pairs, each one job.
  Alternative if the user raises blx01's cap to ~360 GB: copy from blx03 to blx01 under ionice/nice/--bwlimit=100M.
- blx01 leftovers of this task: /var/tmp/fasth3/t466 (build b + jit). Keep the build until the A/B host is decided.

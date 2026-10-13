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

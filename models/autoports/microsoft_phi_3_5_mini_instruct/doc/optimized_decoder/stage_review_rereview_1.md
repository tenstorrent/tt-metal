# Stage rereview 1

Verdict: more-work-needed.

The fresh reviewer found three blocking issues:

1. Recorded target activations exposed batch-32 decode PCC failure
   (`0.901836` in the then-current zero-prefix diagnostic; later semantic
   prefix controls reproduced a cleaner `0.988046` failure).
2. Cumulative correctness, watcher, performance, and profiler artifacts
   predated the latest RoPE and multi-user-cache changes.
3. Four newly duplicated row-major RoPE decode tables added approximately
   100,663,296 bytes without being represented in the context contract.

It also requested explicit disclosure that the functional timing baseline
uses synthetic weights while optimized signoff uses checkpoint weights.

Resolution:

- `AUTODEBUG_RECORDED_STATE.md` plus the AutoFix H1/H2 controls selected manual
  RoPE and default SDPA; final semantic PCC passes at both serving batches.
- Current-source full correctness, watcher10, recorded performance,
  optimized-prefill cache, and four-window Tracy evidence were recollected.
- The final default shares row-major RoPE tables across prefill/decode, so
  duplicate storage is zero. The 100,663,296-byte total is recorded in
  `context_contract.json` and advertised context remains 131072.
- The timing-source distinction is stated in `README.md`.

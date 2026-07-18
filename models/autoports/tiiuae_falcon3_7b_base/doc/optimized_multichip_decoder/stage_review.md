# Independent stage review

Date: 2026-07-18

Verdict: **clean-pass**

The independent `$stage-review` was run against the optimized-multichip
decoder stage. The review was read-only; implementation and evidence changes
were made by the stage owner and then re-reviewed.

## Findings closed before the final verdict

- Completed the missing per-role geometry sweep and crossed O4/O12 again
  under the selected BFP8 activation policy.
- Measured the lower-movement residual family through distributed norm and
  the next real QKV instead of restoring the old residual contract.
- Added source-identical controlled prefill A/B measurements.
- Proved persistent owner/borrower buffer and semaphore reuse.
- Re-ran the full risk-matched suite with Watcher interval 10 and captured
  post-run device health.
- Added maximum-context physical evidence and context-contract provenance.
- Completed activation/fidelity matrices and the real-weight gathered-input,
  local-output O/down AGMM decomposition after adapting its first TTNN weight
  rank error.
- Corrected one stale README label that called the pre-activation 0.503439-ms
  geometry result the final default. The final exact BFP8 default is
  0.491488 ms at batch 32.

## Final review result

The re-review found no remaining code, correctness, performance, topology,
safety, provenance, or documentation issue. It independently verified:

- TP4 final-source hashes and exact-default PCC/performance artifacts;
- 0.286597-ms batch-1 and 0.491488-ms batch-32 traced warmed decode;
- coherent topology families, profiler advice, persistent resources,
  non-aligned lengths, and the inter-layer residual contract;
- final Watcher result of 10 passed and 2 skipped, followed by four healthy
  p300c devices;
- a clean runtime-fallback audit, 87 parseable JSON artifacts, matching
  profiler/Watcher digests, and a clean `git diff --check`.

Final reviewed source hashes:

- implementation: `1bb774f48c3dd19e9c4ba0550e6eb279809973ff0fc856552cd06f16d2b1a199`
- main test: `c7701566daa7895b2cb1f4940dd8cca4a4243ade11c0e16c2b97900ea1b98b38`
- AGMM decomposition test:
  `1ed8037d28718784fd4263163d7ab79dd1a651ac1d6b6a239446ad72ebbf26ff`

# Optimized-decoder stage review

Verdict: clean-pass

Date: 2026-07-29 UTC

Reviewed source/test hashes:

- `optimized_decoder.py`:
  `c3097938ef5162426d3f8684a9de9fdc3bccdbe0db51aa583aaeb4a9067fc37c`
- `test_optimized_decoder.py`:
  `1b60098494fe482f7e06b6d83992b8b82ca214bfb1a643815c2d99af455b1013`

## Required work

None.

## Review history and resolution

The stable review first returned more-work-needed for mixed dense-MLP
precision, larger legal DRAM geometry, current-source residual chaining, and
precision-locked full-attention QKV/O evidence. Remediation:

- Rejected mixed gate/up BFP4 plus down BFP8 and down-only BFP4 on real-weight
  PCC.
- Exercised packed/down block-22/11 and block-22/3 at B1 and B32; neither beat
  block 11 without a serving regression.
- Exercised API-valid down block 33 through the runtime, where it hit a hard
  L1 static-CB allocation clash.
- Reran coherent residual R11/R22 on the final cumulative source. Sliding
  misses PCC, full is slower at B1, and B32 exceeds L1.
- Rejected precision-locked full QKV on decode PCC and selected
  precision-locked full O after it passed PCC and improved B1/B32.
- Regenerated the full final-v2 correctness, performance, context, capacity,
  watcher, health, and Tracy evidence on the selected source.

The fresh final review found only three stale documentation values. After
they were corrected to the final-v2 JSON values, rereview returned
clean-pass. Source, tests, and runtime evidence were unchanged.

## Controlled anomalies

- Stable candidate artifacts use source hash `56781044...`; the subsequent
  runtime change made the explicitly tested full-O role set the default.
  Exact-source `c3097938...` final-v2 PCC, B1/B32 performance, context,
  capacity, watcher, and Tracy runs reproduce the selected behavior.
- Watcher console teardown emits nanobind reference-leak diagnostics. Seven
  watcher cases pass, the 2,171-line device log has no device fault signature,
  and post-run health reports four healthy P300C devices with zero GDDR
  errors.
- Full-model, multichip, and vLLM behavior remain outside this stage by
  contract.

## Scope inspected

The independent reviewer inspected the optimize, TT-device-usage, and
stage-review contracts; optimized source and tests; final-v2 and relevant
candidate JSONs; JUnit, context/capacity, watcher/health, host-timing, Tracy,
and `tt-perf-report` artifacts. Review commands were read-only and did not use
TT hardware.

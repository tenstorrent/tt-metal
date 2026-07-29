# Stage Review

Verdict: **clean-pass**

## Required Work

- None.

## Verified Evidence

- The documented paged prefill/decode contract matches the implementation and
  covers the sole dense Phi-3.5 decoder-layer kind.
- HF-advertised context 131072 is retained. Prefill runs at exact 131072 and
  non-aligned 131071; real-weight decode passes at logical context 131072.
- Page/tile/chunk boundary tests, batch-2 prefill, permuted page tables, and
  distinct batch-32 current positions are covered.
- HF/manual/TTNN prefill and decode PCC exceed 0.995. The distinct-position
  batch-32 traced replay/reference PCC is `0.9999984913`, and the long-RoPE
  traced replay passes.
- The target-stat-derived nonzero long-prefill oracle passes PCC
  `0.9998646685`. The isolated high-accuracy tail-SDPA exception is supported
  by AutoFix controls showing the framework default failed.
- Real layer-0 weights load and pass. Target per-tensor statistics drive the
  deterministic synthetic fixture.
- Runtime fallback audit is clean; measured forwards contain no Torch
  conversion or host collective fallback.
- Repeated steady trace replays are bitwise deterministic at batch 1 and 32
  and equal fresh-cache eager controls. The distinct capture-execution anomaly
  is classified by AutoDebug/AutoFix and does not affect steady replay.
- Exact-final-runtime watcher evidence under `watcher_final` reports three
  passes and no fatal watcher finding.
- Exact-final-runtime Tracy evidence under `tracy_final` includes signposted
  ops CSV, human-readable reports, and derived CSVs. Warmed host latencies are
  2.109976 ms prefill, 1.098405 ms batch-1 traced decode, and 1.224720 ms
  batch-32 traced decode.
- README, work log, context contract, commands, PCC values, artifact paths, and
  capability/limitation statements now agree with the final evidence.
- No optimized-decoder, multichip-decoder, full-model, or vLLM work was added.

## Other Concerns

- None blocking. The long-context precision exception and trace-capture
  execution effect are both controlled by durable regression evidence.

## Hard-Check Gaps

- None. The local stage commit and SHA logging correctly follow this clean
  review and remain the stage owner's final handoff action.

## Scope Inspected

- Skills/contracts: functional-decoder, tt-device-usage, and stage-review.
- Code: `tt/functional_decoder.py`, functional tests, and profiler tests.
- Evidence: final correctness and focused-oracle logs, HF controls,
  AutoDebug/AutoFix, `watcher_final`, `tracy_final`, context contract, README,
  and work log.
- Commands: read-only source/artifact inspection, metric and watcher searches,
  JSON validation, timestamp/provenance checks, and `git diff --check`. No
  device work was run during review.

## Residual Risk

- The long-context tail relies on a numerically justified Blackhole
  high-accuracy compute override. Its target-derived deterministic regression
  should remain part of future stage validation.

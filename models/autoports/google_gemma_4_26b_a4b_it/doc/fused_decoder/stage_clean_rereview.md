# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The full-attention batch-1 traced-decode median advantage remains only
  0.000132 ms (about 0.004%). The accepted 101-replay artifacts establish the
  literal final-source win required by this stage, but the distributions
  overlap and the separation may not survive unrelated system-noise changes.

## Hard-Check Gaps

- The dedicated-router capability probe is documented by its exact command and
  failure signature in `work_log.md`, rather than a separately retained console
  log. Independent source inspection confirms the decisive facts: the
  generalized-gate test is explicitly skipped on Blackhole, and the required
  generalized-gate LLK header exists only for Wormhole in this checkout. This
  is sufficient to control the candidate for the current P300 stage, although
  retaining a probe log would be stronger provenance.
- The fail-closed host gate validates hashes and the complete accepted artifact
  family, but prose/table agreement and the declared freeze interval remain
  review checks rather than automated checks.

## Anomaly Ledger

- Observed anomaly: The dedicated generalized MoE gate is semantically close
  to Gemma's global top-8/selected-softmax subgraph but cannot build on the
  target Blackhole checkout.
  Evidence: `work_log.md` records the focused P300 batch-32 probe and missing
  `experimental/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h`
  compile error. Independent source inspection found the corresponding
  Blackhole test skip and found that header only under the Wormhole kernel
  tree.
  Affected path: Router top-k and selected-score softmax.
  Control or comparison: The retained final TopK/softmax/scatter route passes
  real-weight PCC, trace, context, boundary, and performance gates.
  Likely subsystem: Architecture support for the experimental generalized MoE
  gate.
  Investigation performed: Compared the operation validator/golden and test
  contract to Gemma's 128-expert dense-routing contract; checked the checkout
  for architecture headers and Blackhole test disposition.
  Resolution: controlled

- Observed anomaly: Other dedicated gate/expert families do not directly match
  the delivered single-device Gemma contract.
  Evidence: `work_log.md` records that `deepseek_moe_gate` implements grouped
  linear-renormalized routing, `moe_gate_mm` is a fixed Wormhole-oriented
  pipeline, and `moe_compute`/`TTMoEDecode` require selected compact indices,
  packed lower-precision weights, token adapters, and decode/CCL orchestration
  rather than accepting the decoder's dense 128-entry post-scaled routing.
  Affected path: Router and expert execution.
  Control or comparison: Static inspection of the named operation/module
  contracts and the final sparse-matmul graph.
  Likely subsystem: Experimental MoE gate/dispatch/compute APIs.
  Investigation performed: Reviewed the source-level shape, routing,
  architecture, precision, and prefill/decode contracts recorded in the
  candidate ledger.
  Resolution: controlled

- Observed anomaly: Earlier README values and repeat counts disagreed with the
  accepted artifacts.
  Evidence: The current README now reports sliding PCC
  0.9986195/0.9996651, full PCC 0.9984834/0.9998653, and the accepted 7
  prefill/101 trace-replay regime. These values agree with the manifest-hashed
  PCC and timing JSONs.
  Affected path: Final handoff documentation.
  Control or comparison: Direct JSON parsing and `sha256sum -c`.
  Likely subsystem: Documentation freeze.
  Investigation performed: Compared every README correctness/performance row
  and repeat count against the accepted JSONs.
  Resolution: fixed

- Observed anomaly: The previous evidence window ended before final manifest
  generation.
  Evidence: The current declared window ends at 01:03 UTC; README and work log
  were finalized at 01:02 UTC and `final_manifest.md` at 01:03:18 UTC.
  `final_manifest.sha256` verifies cleanly, and the fail-closed manifest test
  passes.
  Affected path: Final evidence provenance.
  Control or comparison: Filesystem timestamps, manifest declaration, checksum
  verification, and host gate.
  Likely subsystem: Evidence lifecycle.
  Investigation performed: Rechecked the freeze declaration against final
  documentation timestamps and accepted inventory.
  Resolution: fixed

- Observed anomaly: Retained trace replay cannot produce TTNN per-op rows.
  Evidence: `profiler_summary.md` separates capture topology from replay and
  retains nonzero Blackhole/110-worker device-trace reports for both layer kinds
  at batches 1 and 32; both prefill layer kinds have 523-row
  `tt-perf-report` CSVs and text reports.
  Affected path: Decode profiler evidence.
  Control or comparison: Synchronized 101-replay host timing artifacts and raw
  firmware/kernel durations.
  Likely subsystem: Tracy retained-trace host/device join.
  Investigation performed: Rechecked the profiler summary, report row counts,
  raw report inventory, hashes, and retained failure logs.
  Resolution: controlled

## Scope Inspected

- Goal/skill paths: supplied Stage 02 goal contract,
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/graph-fusing/SKILL.md`, and
  `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: fused README, work log, final manifest/checksums, profiler
  summary, all prior review reports, PCC, batch-2, non-aligned boundary,
  bounded-tail-cache, trace/repeat, advertised-context, timing, watcher,
  prefill op-report, and decode raw device-report artifacts.
- Code paths: `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_fused_decoder.py`, `tests/test_functional_decoder.py`, plus the
  generalized/deepseek/moe-gate and common MoE decode sources relevant to the
  remediated candidate dispositions.
- Commands run: read-only `sed`, `find`, `grep`, `git status`, `git diff`,
  `stat`, `sha256sum`, and small JSON/report analysis; host-only
  `pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py`.
  No device, watcher, profiler, reset, or server command was run.

## Residual Risk

- The accepted evidence supports a distinct fused runtime, real-weight PCC for
  both meaningful layer kinds/cache views, non-aligned logical lengths,
  bounded paged-cache preservation, advertised 262144-token context, batch-1
  and batch-32 traced determinism, watcher cleanliness, complete profiler
  evidence under the documented retained-trace limitation, and six same-regime
  final-source performance wins.
- The final path contains the three accepted GeGLU fusions and setup-time
  router-scale fold, while every applicable dedicated-op, structural rewrite,
  and adjacent-op merge named by the graph-fusing audit has an accepted,
  already-present, structurally blocked, architecture-blocked, or
  stage-out-of-scope disposition.

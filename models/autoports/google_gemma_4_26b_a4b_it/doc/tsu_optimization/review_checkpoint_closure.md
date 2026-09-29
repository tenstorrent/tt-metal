# Stage Review

Reviewer: fresh xhigh `final_checkpoint_review`,2026-09-29. Read-only;
no hardware, server, experiment or implementation mutation.

Verdict: clean-pass

This verdict covers the bounded Gemma4 TSU optimization checkpoint. It does
not qualify the unfinished exact-image benchmark, unrun full sweep, or release.

## Required Work

None for this checkpoint.

## Verified Evidence

Independently recomputed all ten paired local cohorts:

| ISL/concurrency | Control TPOT ms | Selected TPOT ms | TSU gain |
|---|---:|---:|---:|
|4096/C1|19.62031|19.59412|0.134%, noise|
|4096/C8|186.01185|180.53341|3.035%|
|4096/C16|367.60522|351.97489|4.441%|
|128/C8|180.61046|175.40299|2.969%|
|128/C16|357.54196|341.88369|4.580%|

- All104 paired responses match text and input/output lengths; normalized
  commands match and both conditions have zero failures. E2EL and steady ITL
  improve. Per-request ITL sums reproduce TPOT within0.000075ms and E2EL
  within0.01ms.
- Earlier exact-image C1 CI verifies+17.7352%TSU and four exact outputs;
  this predates shared batching.
- All18 C16 qualitative replies match the actual pinned suite. The reviewer
  read all six unique outputs and verified suite hash/chat-format metadata.
- Runtime hashes match both local launch manifests. No production `tt/`
  diff fromc9ec3469. Guards preserve serialized fallback, per-slot attention/
  cache handling, decode precision and trace identity checks.
- Logs attest30layers,262144context,32slots,1GBtrace,async,on-device sampling.

## Hard-Check Gaps

- Full29-row C1/C8/C16 sweep NOT RUN following closure direction.
- Selected remote36584709251 blocked before image execution by checkout
  EACCES in three attempts; successful control has five rows/52 requests.
- Exact-image five-row/two-repeat benchmark provisional at inspection:
  five cohorts/40 responses exact and zero failures. Later results are outside
  this review's completed performance verdict.
- Maximum-context correctness is inherited, not freshly qualified by unchanged
  capacity settings. Instrumented C16 checks cover representative layers and
  six exact output/full-KV cases.
- No full-stack serving device duration or exhaustive sampling/quality pass.

## Anomaly Ledger

- Exact-image first4KC8 TPOT188.44447ms/E2EL51765.29ms; second180.14918ms.
  MedianITL177.57346/177.52831ms matches paired selected range and outputs
  match. Startup/capture is plausible; compilation attribution is unproven.
  Resolution: provisional exact-image evidence, excluded from completed
  checkpoint performance verdict; retain first cohort in final pooled result.
- Four qualitative answers stop at256tokens and thermodynamics wording is
  inaccurate. All replies exactly match pinned prior-stage controls.
  Resolution: controlled inherited behavior, not ideal-quality evidence.
- Stale running-qualification summary/context strings were reloaded after
  correction. Resolution: fixed; blocked remote and unrun sweep explicit.

## Scope Inspected

- Skills: stage-review, model-bringup startup, optimize, tracing,
  vLLM-integration, qualitative-check.
- Artifacts: work log, closure/summary/context, prior batching review,
  paired raw benchmarks/commands, qualitative output, C16 contract, CI raw
  results and image import provenance.
- Code: retained generator, adapter, model/shared-batching diffs and comparator.
- Commands: read-only Git/file inspection and Python JSON/hash/timing analysis.

## Residual Risk

Finite evidence supports retained changes and scoped claims, not full-matrix
readiness, long-running memory stability, every sampling mode, or completed
exact-image reproduction. SWE36530661132 remains outside scope.

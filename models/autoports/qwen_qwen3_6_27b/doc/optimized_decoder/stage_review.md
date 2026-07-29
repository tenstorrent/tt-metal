# Stage Review

Verdict: clean-pass

Independent rereview completed 2026-07-29 after remediation.

## Required Work

- None.

## Anomalies resolved

- The initial official-weight full-attention PCC near 0.69 was traced to an
  inherited contiguous Q/gate split. Per-head splitting fixes the default to
  0.998368503; the corrected oracle rejects BFP4 attention at 0.987799141.
- Tracy-instrumented timing is explicitly separated from uninstrumented
  headline timing, whose saved b1/b32 logs reproduce every result.
- Larger precision-locked QKV/O/gate-up geometries have exact L1 blockers;
  legal gate/up and down alternatives were measured at both batches.

## Scope

The fresh reviewer inspected the goal and optimize, shard-advise,
tt-device-usage, and stage-review skills; implementation and tests; all saved
candidate, cache, static, watcher, advisor, and profiler evidence; the context
contract; and functional baselines. The reviewer used read-only commands and no
hardware.

## Residual risk

Only normal later-stage multichip, full-model, and serving integration risk
remains; those paths are outside this stage.

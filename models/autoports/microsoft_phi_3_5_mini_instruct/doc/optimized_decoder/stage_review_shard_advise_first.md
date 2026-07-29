# Stage review: post-advisor first pass

Verdict: `more-work-needed`

## Required work

- P1: The final stage state was not yet committed, while the work log still
  named the restored pre-advisor branch/checkpoint and clean review. Correct
  the metadata, commit all stage-owned code and evidence, then rereview that
  commit.
- P2: Make every final profiler, watcher, test, and A/B reference point to the
  post-advisor artifacts rather than the older restored evidence. Remove the
  premature clean-review/checkpoint claim.

## Other concerns and anomaly classification

- The watcher console prints nanobind leaked-instance/type/function diagnostics
  after five tests pass and after the watcher reports no kernel error. Device
  shutdown completes. This is controlled Python-binding teardown noise, not a
  watcher/model correctness failure.
- The Tracy profiler reports buffers filling after the measured signposted
  window. The final report contains one complete 62-op decode replay, and the
  independent unprofiled 200-replay A/B reproduces the winner. Later dropped
  markers are not used.
- The DRAM-sharded factory substitutes a round-robin output grid. Directly
  carrying that grid into RMSNorm fails because its bounding box spans 22
  cores; the explicit rectangular restoration passes PCC and is included in
  whole-layer timing. This is controlled and already documented.

## Hard-check gaps

- Only batch 32 was captured by the advisor. Both public batch 1 and 32 decode
  inputs are physically padded to one 32-row tile for matmul, so the compiler
  sees the same dense matmul M geometry. The exact seed was nevertheless
  measured independently at both logical batches.

The reviewer otherwise found that existing artifacts substantiate direct
optimized-path correctness, real-weight PCC above 0.995, deterministic,
non-aligned, and full-context coverage, advisor-seed A/B results, BFP4/LoFi
runtime rows, batch-1 improvement, batch-32 non-regression, and watcher
completion.

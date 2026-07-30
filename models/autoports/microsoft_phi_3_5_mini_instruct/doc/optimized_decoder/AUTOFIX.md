# AutoFix Report

## Starting evidence

The first stage review requested BFP4 geometry, packed-vs-split, adapted BFP8
cache, and prefill-to-decode cache evidence. `AUTODEBUG.md` localized a
full-cache staging ownership defect.

## Hypothesis experiments

- Explicit BFP4/LoFi all-width 2/4, QKV 6/12, gate/up 6/12, and down 16/32
  runs selected QKV12/output12/gate6/down32 at 0.5613/0.7430 ms. Gate/up 12
  has an exact BFP4 L1 blocker.
- A device-resident split gate/up candidate passed 12 tests. Packed won both
  prefill workloads; its cumulative topology was retained.
- Adapted BFP8 cache writes passed cache-consuming PCC, but regressed prefill.
  AutoFix proved full-cache staging erases prior users, so the optional
  rejected branch was removed.
- A durable 33-token prefill then position-33 decode test passes with permuted
  pages at B1/B32, PCC 0.9999883/0.9999882.

## Final status

All four findings have fixes or refutations with artifacts. Final correctness,
Watcher, performance, profiler, advisor consistency, and rereview are the
closing gates.

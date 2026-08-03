# Stage review round 2

Verdict: `more-work-needed`.

The fresh reviewer found four completion blockers:

1. Strongest-candidate coverage omitted the composite expert activation and
   prefill comparison.
2. Dense activation folding had not been isolated from residual folding.
3. Profiler and watcher evidence predated the then-current source.
4. Generated provenance omitted the inherited functional decoder and its test.

Resolution:

- Added a 101-replay dense/composite matrix across sliding/full attention at
  batch 1/32, plus 21 alternating batch-1 prefill runs.
- Added an isolated real-shape dense activation test. It selected folding for
  prefill only and rejected it for decode.
- Regenerated Tracy op reports, device-trace replay, watcher, A/B, context,
  candidate, and final-suite captures after the last source/test edit.
- Bound generated evidence to SHA-256 values for fused decoder/test and
  functional decoder/test.

Completion remained held pending a fresh independent rereview.

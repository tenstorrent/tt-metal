# Independent Stage 02 review

Date: 2026-07-22 UTC

Verdict: `clean-pass`

Required work: none. All prior findings are closed and Stage 02 is ready to
commit.

## Closure evidence

1. Complete cache integrity: repeated eager decode now checks every initialized
   cache prefix and exact untouched suffix through positions 3–18. Ten trace
   replays check the complete initialized prefix, untouched suffix, output, and
   bitwise-stable complete K/V tensors. The retained final suite passes all nine
   tests.
2. Durable candidates: the runner and all candidate logs are retained. Real
   dense split and wide decode both pass at PCC 0.999298; real packed and split
   sparse decode fail at PCC 0.702504 and 0.605508. Default `auto` selects split
   only for multi-token prefill and wide for one-token decode.
3. Capacity accounting: the context contract records 1,646,373,824 bytes of
   functional weights, 1,062,051,840 bytes of persistent split copies, and
   2,708,425,664 total fused static bytes. The configured cache extent remains
   128 and the validated prefill boundary advances from 17 to 33.

## Independent artifact checks

- The 17:56:07 raw prefill capture contains 52 signposted operations, including
  split gate/up/down matmuls and one SDPA.
- The 17:56:21 raw decode capture contains 56 signposted operations, including
  one wide 5760-output projection, no sparse matmul, two paged updates, decode
  SDPA, and exactly one 0.584 us layout conversion.
- The final same-process gate improves warmed prefill from 8.206 to 7.158 ms
  and traced warmed decode from 5.988 to 5.891 ms. It also beats every retained
  candidate that passes real-weight correctness.
- The layer-12 prefill PCC delta (0.999193 functional to 0.993041 fused) is
  explicitly recorded, attributed to BF16 SDPA reduction order, and remains
  above the unchanged 0.99 bar. Layer 13 covers full attention.
- Scoped changes contain only Stage 02 implementation, tests, context, and
  documentation; no later-stage source is present.

## Classified residual risks

- Lengths 34–131071 are unprobed and explicitly recorded as an unvalidated
  interval, not a claimed physical limit.
- Batch 2 emits an internal on-device reshape warning that selects interleaved
  layout. The same test passes with watcher and
  `throw_exception_on_fallback=true`; it is not a Torch/host fallback.
- Firmware and one-chip P300 discovery warnings are retained in logs and do not
  produce watcher, correctness, fallback, or exit-code failures.

The review was read-only and used source inspection, git diff/status, logs,
JSON, Python AST, and raw/processed CSV analysis. It ran no hardware command,
test, reset, edit, commit, or push.

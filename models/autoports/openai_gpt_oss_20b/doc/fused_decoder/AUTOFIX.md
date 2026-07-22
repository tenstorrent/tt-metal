# AutoFix report: sparse real-weight correctness

## Starting evidence

- Source-only diagnosis: `AUTODEBUG.md`.
- Original failure: selecting split sparse MoE for decode improved the synthetic
  traced microbenchmark to 1.852 ms, but real layer 12 and 13 decode fell to
  PCC 0.605508 and 0.668870.
- The synthetic fixture has all-zero gate/up and down expert matrices, so its
  final PCC cannot validate any sparse weight multiplication.

## Hypothesis experiments

### Construction-time split views

- Hypothesis: strided device slices of the interleaved gate/up tensor are not a
  valid projection input.
- Experiment: real layer 12 with the test's exact seed and Float32-to-BF16
  input generation; change only decode MoE policy. A preliminary runner used
  direct BF16 random generation and produced PCC 0.941327 for both split and
  wide, so that input was rejected as a nonrepresentative control.
- Result: with the authoritative input, dense wide and dense split both give
  decode PCC 0.999297600. An isolated real-weight prefill MoE comparison is
  bitwise-equivalent (PCC 1.0) at S=3, 17, and 33.
- Verdict: refuted as a correctness bug. Split remains slower for one token
  (6.795 ms versus 5.906 ms traced in the retained real-layer candidate logs),
  so default `auto` still uses it only for multi-token prefill.

### Packed BF16 sparse path

- Hypothesis: bypassing split views with the original packed weight makes the
  sparse topology correct.
- Experiment: real layer 12, dense split prefill and packed BF16 sparse decode,
  5x6 grid, `in0_block_w=2`, HiFi4/FP32 accumulation.
- Result: decode PCC 0.702504329.
- Verdict: refuted.

### Sparse weight dtype

- Hypothesis: the sparse reader requires the BFLOAT8_B weight dtype covered by
  in-tree sparse-matmul tests.
- Experiment: typecast only packed gate/up and down weights to BFLOAT8_B at
  construction; retain BF16 inputs, outputs, biases, cache, and dense prefill.
- Result: decode PCC 0.702890227.
- Verdict: refuted; still far below the 0.99 stage bar. Candidate copies were
  removed.

### Sparse contraction block

- Hypothesis: `in0_block_w=2` is incompatible with this sparse shape.
- Experiment: packed BF16 sparse decode with only `in0_block_w` changed from 2
  to 1.
- Result: decode PCC 0.697125224.
- Verdict: refuted; the original configuration was restored.

## Final status

Fixed. Sparse policies are rejected candidates, not the default runtime. The
final graph uses split dense gate/up projections for prefill and the
measured-faster wide dense projection for decode. Both real layer
kinds again pass above PCC 0.999 for decode, and real non-aligned multi-token
MoE outputs match the wide control exactly. The remaining broad suite,
watcher, performance gate, and profiler captures are run and retained by the
Stage 02 work log.

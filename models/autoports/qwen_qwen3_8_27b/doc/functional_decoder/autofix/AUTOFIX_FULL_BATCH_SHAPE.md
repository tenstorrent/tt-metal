# AutoFix Report: full-attention batched paged decode

## Starting Evidence

- Source report: `AUTODEBUG_FULL_BATCH_SHAPE.md`.
- Original failing command:
  `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k full_attention_batched_paged_decode -s`
- Observed result before the fix: public input `[2,1,5120]` produced
  `[2,2,5120]`.

## Hypothesis Experiment

- Hypothesis: the shared gated-attention path's `transpose(1, 2)` passes
  `[B,1,H,D]` to paged SDPA decode instead of its native `[1,B,H,D]`
  contract, then transposes the native output `[1,B,H,D]` to
  `[1,H,B,D]`.  The resulting `[1,B,H*D]` attention tensor broadcasts
  against gate `[B,1,H*D]` to `[B,B,H*D]`.
- Prediction: at B=2 the unmodified path returns `[2,2,5120]`; replacing the
  two boundary operations with device-side permutations `(2,0,1,3)` and
  `(1,2,0,3)` returns `[2,1,5120]`.
- Experiment/result: the original command failed exactly with
  `assert [2, 2, 5120] == [2, 1, 5120]`.  With the autoport-local boundary it
  passed and returned the required shape.
- Verdict: verified.

## Fix

`FunctionalDecoder` now installs an instance-local full-attention wrapper.  It
delegates prefill unchanged and implements only paged decode with the existing
loaded weights and TTNN operations.  Query is explicitly permuted
`[B,H,1,D] -> [1,B,H,D]` before paged SDPA and output is permuted
`[1,B,H,D] -> [B,H,1,D]` before head concatenation.  There is no global monkey
patch and no runtime torch, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.

The B=2 test uses disjoint, permuted page-table rows (`[6,2,4,0]` and
`[7,3,5,1]`), so both cache update and SDPA read execute through separate
physical pages.  A separate per-row numerical oracle was not added in this
focused repair; the existing check proves the broadcast/layout failure is gone,
while the real-weight B=1 test proves the numerical path.

## Verification

- B=2 disjoint-page regression:
  `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k full_attention_batched_paged_decode -s`
  -> `1 passed`; output shape `[2,1,5120]`, all finite.
- B=1 real-weight prefill and traced-decode regression:
  `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_paged_prefill_and_decode_pcc -s`
  -> `1 passed`; prefill PCC `0.9972974493968263`, traced decode PCC
  `0.9976662040611718`, repeated trace replay bitwise deterministic.

## Final Status

Fixed with on-device, autoport-scoped operations.  The original failure and
the single-user real-weight traced path both pass.  Residual risk: a future
batch-specific numerical test should compare each batched row with an isolated
B=1 control; the current disjoint-page test checks shape, finiteness, and actual
separate physical routing but not row-wise PCC.

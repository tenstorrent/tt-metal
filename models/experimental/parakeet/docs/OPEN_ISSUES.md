<!-- SPDX-FileCopyrightText: © 2026 Abror Shopulatov -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Open issues / remaining limits (recorded before submit decision, 2026-09-25)

Measurement.md` and the gate-matrix
table in `README.md`; this file records only the remaining limits and deliberate non-goals.

## Oracle note (referenced by `tests/test_parakeet.py`)

The immutable correctness oracle is FP32 on the reference host (A100 preferred; CPU recorded),
 The packaged tests compare the TT path against the local CPU FP32
`reference/` implementation — a development proxy, not the oracle itself. Evaluation verdicts
(encoder max-row NRMSE ≤ 0.04, exact greedy tokens) come from the controller's independent
FP32 oracle, not from these tests.

## bfp8_b: unsupported, by design (measured)

- Receipt: smoke evaluate a measurement (2026-09-25, 8.2 s,
  `candidate_error`, controller suite definition `03492b8d...`): the factory fails fast with
  `ValueError: unsupported precision 'bfp8_b'; supported: ('bf16', 'fp32')` before any
  numerics run. This is the required behavior ("unsupported modes must fail clearly,
  never silently substitute another format"), so no quality or latency numbers exist for the
  mode — that absence is the honest cell, not a gap in measurement.
- Tradeoff of leaving it unimplemented:
  - TT `bfloat8_b` is block floating point, not IEEE FP8; adopting it for weights (and any
    activations) would require a new rounding path plus re-tuning of the fp32 exception list,
    then re-gating smoke and bringup from scratch (source-hash unfreeze).
  - Quality headroom is thin even at bf16: bringup max row NRMSE is 0.0331 (`tone`) against
    the 0.04 gate, and the recorded A100 context study saw its own BF16 run fail two cases
    (long tokens, tone NRMSE 0.089). Coarsening weight precision further puts the exact
    greedy-token gate at material risk for no measured latency benefit anywhere in the
    profiled hot path.
  - The profiled time is dominated by conformer blocks (72–77% of encode) and TDT decode
    round-trips (40–64% of transcribe) — areas where operand format changes were measured
    (H2, H3) and did not clear their acceptance bars.
- Decision: bf16 (with the declared fp32 exceptions) remains the production precision;
  bfp8_b is documented as unsupported rather than shipped ungated.

## Other recorded limits

- **ONNX baseline**: unavailable — `onnxruntime` is not installed on the TT host. Recorded,
  never claimed (the methodology requires baselines be measured, not assumed).
- **Precision exceptions are fp32-heavy**: activations, accumulation, attention softmax,
  decoder LSTM state and several module weights stay fp32 (declared in the precision policy).
  This is quality-driven and bounds achievable speedups; it is a deliberate correctness
  tradeoff, not an oversight.
- **Decode round-trip floor**: measured device-side floor ≈ 87.7 µs/step (H1) with capture-side
  floor 140.3 µs/step (H3); batching syncs across it was rejected (share far below the ≥80%
  acceptance bar — see `PROFILING_PLAN.md`).
- **CUDA context**: A100 numbers (`gpu-precision-parakeet-5929b627da7f`) are context only and
  are never used as speedup targets; TT-vs-TT ratios are reported separately .

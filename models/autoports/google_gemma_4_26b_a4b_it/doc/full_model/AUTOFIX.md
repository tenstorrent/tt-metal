# AutoFix Report

## Starting Evidence

- Source report: `AUTODEBUG.md` (fresh runner sandbox was unavailable, so the
  skill's serial fallback produced the report).
- Original command: fallback-raising opt-in TP4 reduced real-weight full-model
  probe.
- Original failure: trace-time host reads/writes during split decode capture.

## Hypothesis Experiments

- Hypothesis: the new synchronous TP4 embedding gather is untraceable.
  Experiment: warm/capture only `Gemma4FullModel.embed_tokens`.
  Result: passed (`/tmp/gemma4_embed_trace.log`).
  Verdict: refuted.

- Hypothesis: the model graph itself contains the unsupported operation.
  Experiment: eager prefill, then warm/capture only `model.decode_forward`.
  Result: passed (`/tmp/gemma4_model_trace.log`).
  Verdict: refuted.

- Hypothesis: position `plus_one` can be captured on the replicated row-major
  state.
  Experiment: add only both `plus_one` calls to the passing model trace.
  Result: `Writes are not supported during trace capture`.
  Verdict: refuted. Position advance was moved to device-only operations
  between model and sampling trace replays; there is no host position rebuild.

- Hypothesis: an in-place elementwise add can replace `plus_one` in capture.
  Experiment: `ttnn.add(..., output_tensor=...)` on the position state.
  Result: optional output is unsupported for row-major elementwise operations;
  tile optional-output capture also produced trace-time writes.
  Verdict: refuted.

- Hypothesis: semantically greedy k=1 top-k can write directly into the token
  feedback tensor in the sampling trace.
  Experiment: `Sampling1D` top-k with k=1/p=0/temp=1 and
  `tt_out_tok=token_input` after a passing model trace.
  Result: sampling trace ID 1 rejects writes (`/tmp/gemma4_split_trace_device_positions.log`).
  Verdict: refuted.

- Hypothesis: native force-argmax (the other common greedy sampler) supports
  the same direct feedback contract.
  Experiment: `Sampling1D` force-argmax with `tt_out_tok=token_input`.
  Result: sampling trace ID 1 rejects writes (`/tmp/gemma4_split_argmax_trace.log`).
  Verdict: refuted.

## Follow-up repair

The apparent optional-output limitation was caused by warming a different
program variant. The allocator-output warmup did not compile the
`tt_out_tok=token_input` program; capture then attempted binary-upload writes.
After changing warmup to execute the exact optional-output graph and matching
the standard fixed 32-slot decode shape, the original fallback-raising split
trace passed twice with zero host token refreshes. Evidence:
`/tmp/gemma4_split_exact_warm.log`.

## Final Status

Fixed. Canonical force-argmax split sampling, direct device token feedback,
and exact device position progression pass the reduced real-weight TP4 repro.
Full all-layer accuracy, qualitative, performance, review, and commit gates
remain separate stage work.

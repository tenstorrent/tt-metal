# AutoDebug: Gemma4 full-model trace capture host I/O

## Symptom

The fallback-raising reduced real-weight TP4 probe reaches eager prefill and
eager decode, then fails while capturing the model decode trace with four
simultaneous `Writes are not supported during trace capture` fatals followed
by trace-time read fatals. Evidence: `/tmp/gemma4_trace_probe.log`.

## Highest-confidence finding

`Gemma4FullModel.embed_tokens` performs a synchronous `ttnn.all_gather`
through `models/demos/gemma4/tt/ccl.py::ccl_allgather` inside every decode.
Sibling traced generators use `ttnn.experimental.all_gather_async` with
persistent global semaphores. The optimized decoder itself has a passing
standalone trace test, so the newly added full-model entry collective is the
first boundary not already covered by that evidence.

Prediction: capturing embedding plus its synchronous gather alone reproduces
the host-I/O fatal; replacing only that collective with the established
trace-safe async/persistent form removes it without changing sharding,
residual layout, dtype, or decoder policy.

### Experiment result

Refuted. `GEMMA4_EMBED_TRACE_ONLY=1` warmed and captured the real sharded
embedding plus synchronous TP4 gather successfully under fallback-raising
runtime. Artifact: `/tmp/gemma4_embed_trace.log` (`1 passed`). Do not replace
this collective based on the original hypothesis.

## Secondary hypotheses

1. If embedding-only capture passes, isolate final norm/LM head, then each
   layer in sequence; do not change several boundaries together.
2. Trace capture executes and mutates token/position buffers. The generator
   now restores them after capture; verify exact +1 progression after the
   collective issue is removed.
3. Sampling is a separate trace. Its `Sampling1D` gather already uses the
   async trace-safe implementation, but its output shape and `tt_out_tok`
   alias still require the original minimal repro.

## Required focused experiment

Add an opt-in real-weight probe mode that warms and captures only
`model.embed_tokens(token_input)`. Run it on the same 1x4 ring with
`throw_exception_on_fallback=true`. This experiment must verify or refute the
headline finding before implementation changes.

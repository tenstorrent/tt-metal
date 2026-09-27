# AutoFix: fixed-step token output

## Starting evidence

`AUTODEBUG_buffered.md` identifies a host output transfer after every replay in `tt/generator.py::generate`. Device tokens and positions already feed the next step without the transfer.

## Hypothesis experiment

- Hypothesis: fixed-length generation can record outputs with a device index, then read the sequence once, preserving the original token sequence and feedback.
- Source check: `indexed_fill` supports runtime uint32 row indices and row-major rank-4 token storage; `plus_one` supports a one-element row-major uint32 tensor.
- Candidate fix: `_prepare_output_buffer` allocates persistent output/index tensors and warms/captures a recorder trace. `_generate_buffered` queues existing model/sampling traces and the recorder nonblocking, then performs one final read. `generate(stop_on_eos=False)` selects it only without teacher forcing or explicit host sampling. `buffer_tokens=False` remains a same-code comparison control. EOS stopping, teacher forcing, and low-level batched decode semantics are unchanged.
- Resources: for N generated tokens, two `(N-1)*32*4`-byte logical uint32 buffers per device (persistent history plus captured scratch), plus index/alignment and trace commands. All 32 sampler lanes are preserved by the recorder. High-level standalone generation returns lane 0 as before.
- Focused tests: `tests/check_buffered_generation.py --recorder-only` checks exact rows and indices at 1/3/127 steps and reuse. The real-weight reduced path compares buffered and per-token-readback output, request reuse, resized generation windows, final feedback token, and steady-state counters.

## Validation

Passed locally: `python -m py_compile` for generator and probe; Black formatting; `git diff --check`. Python-only change, no C++ build needed.

Hardware validation is coordinated by the parent stage owner to serialize device use. Commands (run with the stage's configured Python environment):

```sh
TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_buffered_generation --recorder-only --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/buffered_recorder.json
TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_buffered_generation --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/buffered_generation_reduced.json
```

## Final status

Candidate validated on device; final all-layer accuracy, exact token controls and end-to-end performance pass (see readiness.json and performance_comparison.json). The final output transfer is included in end-to-end decode timing; it is one completion boundary, not a per-token dependency. The initial prefill token transfer remains part of TTFT. Recorder trace preparation is included in trace-setup/TTFT metrics; warmed same-shaped requests reuse the trace and only reset its index at the request boundary.

Trace lifetime correction: output capacity is allocated and warmed before model capture; recorder capture follows model/sampler capture. A request requiring larger capacity forces full trace release/rebind before allocation; shorter windows reuse existing storage and read only their valid prefix. The focused commands enable trace allocation tracking. `indexed_fill` plus copy moves the complete history each replay (quadratic in generation length); at 128 generated tokens the two logical buffers total 32,512 bytes/device. Final terminal profiling measures the recorder at at most 9.821 us for the 127-row buffer; this does not establish optimality for much longer output. Existing `slice_write` is BF16-only with host-static indices and cannot store exact vocabulary IDs; generic scatter also copies its output.

## Coordinated device validation update

The stage owner ran the reduced two-layer real-terminal test with `TT_METAL_TRACE_ALLOC_TRACKING=1`; `buffered_reduced.json` and `buffered_reduced.log` record `status=pass`. Original-loop and buffered tokens match across 16 generated tokens, repeat requests and shorter output windows. Steady-state token readbacks decrease from 15 to 1, with 15 model/sampling replays and zero input/position/page-table refreshes. The initial recorder-only probe also passed (`buffered_recorder.json`). These are correctness/trace-lifetime results. Tracker-distorted timing from these runs is not accepted as performance evidence.

The focused probe now additionally checks seeded top-k=16/top-p=0.9/temperature=0.8 equality and greedy alternation, changed prompt contents, zero/single-token output, increasing output capacity, and optional `--prompt-lengths` boundary cases. `runtime_audit.device_only()` wraps both recorder warm/capture construction and two actual model/sampling/recorder replays, excluding the final caller readback. Explicit extra replay state is reset before checking repeated generation. The extended probe passes syntax/format checks and the coordinated device run recorded below.

Expanded final-policy reduced validation passes with worker Watcher and allocation tracking: `buffered_extended_watcher.json`, `buffered_extended_watcher_retry.log`. Seeded sampled output, greedy alternation, changed prompts, zero/single output, capacity growth, reset, runtime host-boundary audit and nine logical prompt boundaries pass. Ethernet instrumentation alone was disabled after ACTIVE_ETH config overflow before model execution; recovery is in work_log.md.

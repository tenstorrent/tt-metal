# Completion export hypothesis review

The hypothesis is refuted for the pinned stock runtime: no lossless existing export of worker phase completion was found. Adapter-only timestamps cannot prove the requested complete timeline. This is a source inspection result, not a device measurement.

Inspected root `AUTODEBUG.md`, AutoFix instructions, plugin `90f33d2e392eb8bfda6948df9dabe712a724cb1a` at `/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin`, and the installed generic vLLM `python_env/lib/python3.10/site-packages/vllm/v1/` metrics/engine code. No hardware commands or runtime changes were made.

## Concrete evidence

- `async_decode.py:136-158`: immutable submitted context includes request IDs and submit time; completed output has completion time. These are internal objects, not an export.
- `async_decode.py:1022-1067`: completion time follows existing readback finalization and `_get_output_tokens`, including host sampling. It precedes output construction and state application.
- `async_decode.py:290-304`: actual asynchronous resolution builds the runner output and queues the completed record. This path runs for the final output even if no further model forward occurs.
- `async_decode.py:693-695,807-812`: completed records enter a transient deque and are destructively drained. Polling cannot guarantee all entries; the next adapter invocation is neither a guaranteed completion boundary nor guaranteed to occur.
- `async_decode.py:1110-1176`: output construction and state application are additional work after the timestamp. Output suppression can alter already-published output; reporting only sampled tokens is insufficient for accepted-token accounting.
- `model_runner.py:4219-4248`: synchronous forward returns before `_get_output_tokens` samples host-compatible logits. `:3808-3821` is an existing synchronous finalization wrapper after which sampled output construction has completed.
- `model_runner.py:4782-4809`: final-prefill deferred state stores identical submit and completion timestamps generated after output construction. This cannot reconstruct the prefill start or intermediate chunks.
- `async_decode.py:736-798`: stock counters log aggregate submissions and overlap at intervals/powers of two. They contain neither a lossless per-step timeline nor request positions.
- `stage_outputs.py:1-18,37-80`: stage outputs are model-provided decode payloads collected immediately after forward. They exclude prefill and force a synchronous regime. This interface is not a passive completion observer and enabling it changes the measured runtime.

## Generic vLLM alternatives

`v1/metrics/stats.py:500-506,548-557` derives request prefill/decode durations from scheduler and engine-output timestamps. `v1/engine/output_processor.py:800-860` exports an OTel `llm_request` span containing external request ID, input/output counts and those durations. It therefore can cover the final client-visible token tail at request level, including host sampling. It does **not** export every intermediate prefill or async work later suppressed/discarded, executed model positions, worker step identities, or an exclusive prefill/decode partition under overlapped requests. Summing request spans double-counts shared/overlapping work. `v1/engine/__init__.py:181-185` also explicitly warns that engine event timestamps should not be compared with timestamps from other processes. The stock per-request span is valuable corroboration, but not the missing general full-phase collector.

## Permitted fix versus required exception

A direct adapter observer may record its existing submission and readback/host-processing boundaries, effective resident positions and arithmetic work. A guarded caller-frame snapshot could recover submission IDs, but does not create a callback after the adapter returns. Sampling a transient completion queue cannot close the final-tail or losslessness gaps. No permitted adapter-only repair was established.

The smallest sound exception is an optional observational plugin callback at submission, post-sampling/output-construction and deferred state-application boundaries. It must preserve an immutable correlation ID through dataclass replacement and async lifecycle; carry request IDs, slot mapping, chunk endpoints, sampling route and reload policy; snapshot existing CPU metadata only; and buffer records outside measured work for an explicit flush before shutdown. The adapter must supply effective device-resident positions rather than trusting stale scheduler positions. Scheduler entry markers are needed to include preprocessing/drain gaps. The observer must also report failed/discarded steps and reconcile the final tail. No extra device synchronization, device reads or profiler is needed.

This is a plugin runtime/interface exception, not a test-only correction under PC-02/03. The accompanying PC-04 patch is a reviewable hook proposal, **not applied and not a validated complete collector**. Tests and adapter/collector implementation would still be required after authorization. Neither source inspection nor proposed hooks establish a valid roofline measurement.

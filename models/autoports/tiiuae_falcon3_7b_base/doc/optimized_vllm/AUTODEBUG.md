# AutoDebug: Falcon3 optimized vLLM serving gap

## Verdict

The evidence does not identify a second slow Falcon3 model or sampling graph. The most likely missing 5.5 ms in steady single-user decode is the real-vLLM per-token control path around the same two traced device replays: scheduler/input construction, forward/sample hand-off, deferred-output locking/thread hand-off, read-event completion, token extraction, state mutation, and `ModelRunnerOutput` construction. The observed page-table-compare-neutral result and the regressing immutable-payload experiment both argue against one large adapter tensor-copy or payload-serialization bug.

The device-to-host token read remains the highest-value boundary to isolate. It is submitted nonblocking, but every token is then finalized by synchronizing its event before host output processing. The direct canonical measurement also reads one token per step, so raw token read cost alone cannot explain the gap; the plausible issue is its interaction with vLLM queue ordering and the output thread.

This is an inspection-only diagnosis, not proof of a single root cause. The experiments below are designed to make the leading explanations independently falsifiable without any prohibited profiler.

## Direct observations

- The primary before artifact (`results/before/vllm_benchmark.json`) reports mean TPOT 15.8877 ms (62.9417 t/s/u), median ITL 14.5657 ms, and p99 ITL 15.1213 ms for 128 input / 128 output / concurrency 1. Thus roughly 1.32 ms of mean TPOT relative to median ITL is boundary/outlier weighting, but a steady gap of about 5.50 ms remains versus the 9.067 ms canonical caller-visible measurement.
- The CI burst is a different capacity workload: 32 concurrent prompts produce mean TPOT/ITL 16.8601 ms. It is affected by admission, batching, and chunking and is not evidence for single-user decode speed.
- The full-model canonical steady path executes the model and sampling traces with nonblocking replay and returns the persistent sampled-token device tensor (`tt/generator.py:684-688`).
- The vLLM adapter's steady path deliberately ignores stale host token, position, and page-table payloads and invokes the same traced device-sampling path (`tt/generator_vllm.py:191-219`). This makes an unintended eager model or generic sampler unlikely.
- The plugin nevertheless reconstructs a sampling dataclass every decode, including `.tolist()` on each tensor field (`plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:584-597`), and the adapter converts its active values into Python lists again (`tt/generator_vllm.py:91-106`). The measured immutable-payload reuse candidate regressed to about 16.758 ms TPOT, so this concrete overhead is not dominant in isolation.
- Async output read is real: `cpu(blocking=False)` is followed by a recorded event (`tt/generator_vllm.py:221-227`). Finalization synchronizes that event before host conversion (`async_decode.py:655-675`), and the adapter extracts only sampled token ids, never full logits (`tt/generator_vllm.py:229-234`).
- Each token passes through a deferred wrapper with a lock/event (`async_decode.py:74-116`), finalization/token extraction (`async_decode.py:402-446`), state application (`async_decode.py:459-464`), and vLLM output packaging. These are all outside the canonical two-trace timing.
- The plugin's steady-overlap eligibility is conditional, and a pending async output is forcibly drained whenever the next step fails those invariants (`async_decode.py:389-400`). Advertising async support therefore does not by itself prove that all 126 steady gaps avoided a drain.

## Ranked hypotheses

### 1. Distributed scheduler/plugin control-plane latency is the main steady gap (high confidence)

No inspected operation alone proves a 5.5 ms cost. The cumulative path does substantially more host work than the canonical loop: construct/slice model input, serialize sampling state, create deferred objects and partials, cross the executor output thread, wait/read one token, convert it to NumPy/Python lists, update request state, and build engine output. The neutral page-table optimization and regressing payload reuse predict this distributed-overhead shape.

This explains why the model can retain its 9.06 ms graph while HTTP-visible ITL stays near 14.6 ms. It does not by itself identify which control-plane phases dominate.

### 2. The read-event/output-thread boundary serializes useful overlap (medium confidence)

Decode submission starts a nonblocking host copy, but completion always waits on its event. vLLM 0.22 schedules and resolves through a deferred wrapper/output thread, and strict steady eligibility can force engine-thread drains. A queue/order interaction could expose device completion plus thread wake-up on every token.

Raw device read latency alone is insufficient: the canonical caller-visible 9.067 ms path already includes per-token visibility. This hypothesis requires submit cadence to remain near 9 ms while completion or next-submit cadence expands toward 14.6 ms.

### 3. Post-read token/state/output processing is a material part of the floor (medium-low confidence)

The sampled scalar is repeatedly reshaped, indexed, converted, applied to cached request state, and packaged. This is certainly extra work, but batch one makes it unlikely to explain the whole 5.5 ms without scheduler/thread interaction.

### 4. TPOT accounting inflates the headline but is not the root cause (certain, partial)

Mean TPOT is derived from request boundaries and 127 inter-token gaps, while median ITL is 1.32 ms lower. Longer warmed requests and detailed ITLs should reduce this component. Even the median remains about 60% slower than the canonical path, so accounting cannot close the gap.

## Focused benchmark and contract experiments

1. **Reconcile the critical path with benchmark-only timestamps.** On the same warmed 128/128/1 workload, collect `perf_counter_ns` phase counters for schedule-to-input-ready, input-ready-to-decode-return, read-event wait plus host conversion, token extraction/output packaging, state application, and next-submit. Report medians and p99 after trace warmup; their sum must reconcile with median ITL. Keep this temporary evidence out of the production adapter and do not use any prohibited profiler.
2. **Measure the real-vLLM control-plane floor.** In a temporary contract harness using the same scheduler/request, replace decode with a preallocated host sampled token wrapped as an async model-runner output. Successively enable wrapper/output/state handling, an async read of a preexisting device token, and finally the real two traces. The deltas isolate scheduler/plugin, read boundary, and device work.
3. **Separate submission cadence from completion cadence.** Feed identical steady `TTModelInput` objects through the direct plugin controller and compare current deferred read with immediate completion. If contract-safe, enqueue several steady replays and resolve oldest-first. A roughly 9 ms submit cadence with roughly 14.6 ms completion cadence confirms the read/thread boundary; a 14.6 ms submit cadence points upstream to scheduler/input preparation.
4. **A/B output processing independently.** Preserve real traces/readback but return a fixed prebuilt `ModelRunnerOutput`; separately preserve normal output/state packaging but feed it a pre-read host token. This distinguishes scalar read completion from `_get_output_tokens`, state update, and packaging costs.
5. **Resolve accounting with detailed warmed requests.** Save detailed per-token results for at least ten independent 128/128 requests at max-concurrency 1 on one server, discard the first request, and compare median/p10 ITL with TPOT. Add a 1x512-output control to amortize request boundaries. Keep max model length, mesh, TT config, generation mode, and sampling unchanged.
6. **Prove steady overlap rather than infer it.** Add benchmark-only counters for `reset_batch=False`, steady-fast-path eligibility, trace captures, forced drains, and pending-depth during the 126 steady gaps. Expected evidence is one capture/reset and no forced drain thereafter. Any miss directly explains a scheduler barrier.

## Refuted or lower-priority directions

- Repeating page-table host comparison removal: already performance-neutral.
- Reintroducing immutable sampling-payload reuse as previously implemented: it regressed the matched primary benchmark and has been removed.
- Optimizing the model trace or replacing split greedy sampling before isolating host cadence: current code reaches the same canonical nonblocking model-plus-sampling traces.
- Host greedy/top-1 or full-logit fallback: the measured adapter rejects full-logit host processing and reads sampled tokens only.
- Explaining single-user decode with the CI burst: its concurrency and batching semantics differ.

## Investigation limits

The prescribed `.agents/scripts/autodebug.sh --agent codex` fresh runner was attempted, but its nested session could not launch filesystem commands because its sandbox launcher could not find `bubblewrap`; the supported Claude backend then failed because its OAuth session was expired. A fresh-context inspection subagent was used as the bounded fallback, and the headline claims above were rechecked against the current model/plugin sources and JSON artifacts. No implementation code was edited and no hardware-dependent experiment or prohibited profiler was used.

# AutoDebug: Stage 11 host phase accounting

At the initial inspected revision, the missing measurement is the complete serving lifecycle, not a missing timer around `decode_forward`. The adapter's `decode_submit` occurs before readback and completion. Its existing `prefill_complete` measures only the adapter call, and an intermediate-only prefill can emit it without waiting for device completion. Neither event stream can, by itself, supply Stage 11's full, non-overlapping prefill/decode wall-time denominators.

**Concurrent-change finding:** revision `d84c5effabae003e8cd54b9347cdd34d7515f342` appeared during review and adds a collector. It correctly observes async output readiness, but its phase partition can misattribute prefill/decode transition time by stretching an old step to scheduler consumption and giving prefill unconditional overlap priority. The separate addendum below gives the source-level counterexamples; it does not claim a measured runtime error magnitude.

**Recommended intervention:** retain lightweight monotonic timestamps and immutable request/step metadata in the plugin's existing execution and deferred-output lifecycle; close decode timing at its existing completion path; construct phase intervals from those records and scheduler transitions. Keep the serving configuration, device calls, and synchronization unchanged. Adapter markers remain useful for internal prefill chunks and physical execution metadata. An adapter-only duration patch is insufficient for the full requirement.

This is an inspection-only diagnosis. No target execution, profiling, device access, synchronization experiment, or source edit was performed. Only this report was written. Proposed instrumentation and validation below are follow-up work, not implemented or hardware-verified results.

## Evidence scope and notation

Inspected on 2026-10-25:

- Host tt-metal HEAD: `78dd15ceacf11a789487bde65dca2d87ff272c6d`.
- `A` = `models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py`. SHA-256 `315bd0fe2ac27fb0c9cbfc6efa655664ddee53bfb8c880d0c10d9b9f2b746307`, identical in the host and container `/workspace` at inspection.
- `G` = `models/autoports/qwen_qwen3_8_27b/tt/generator.py`.
- `P` = container `/workspace/bringup/dependencies/vllm-tt-plugin/src/vllm_tt_plugin/`. Plugin HEAD: `d5ee19c6754905e7a9a69c79d9e2ee4b99b84295`, initially clean. Read through read-only `docker exec qwen38-functional-build`.
- `V` = container `/opt/venv/lib/python3.10/site-packages/vllm/`. Installed distribution directory is `vllm-0.30.0+empty.dist-info`; the plugin's installed `direct_url.json` points to the editable checkout above. Installed engine/executor source was inspected, rather than assuming old version comments describe execution.
- `S` = `.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/`.

The governing metric contract is `S/skills/benchmark-model/references/run-contract.md:224–270`: include dispatch, all operations, communication, sampling/readback, every prefill chunk, and intervening host gaps; sum non-overlapping intervals on a common timeline; match work to actual requests/steps/devices; exclude warmups and the initial client probe; use buffered monotonic host markers and existing completion boundaries without extra synchronization or profiling. Unsupported accounting must be reported as a collection failure.

Unless explicitly marked as the concurrent-change addendum, host line references and findings below refer to the initial `78dd15ce...` snapshot. The later adapter change inserts three initialization lines, shifting subsequent adapter references by three. The plugin remained at the inspected clean revision. The independent concurrent edits were neither made nor reverted by this investigation.

## Headline findings

### 1. Decode submission is not completion; the required completion boundary already exists

Direct observations:

- `A:367–379` submits model work, schedules boundary snapshots, then emits `decode_submit`. There is no decode start timestamp or completed duration in that event.
- `G:824–831` executes the model and sampling traces with `blocking=False`, returning the persistent `self.tokens` tensor.
- In normal plugin async execution, `P/async_decode.py:1135–1141` calls `submit_decode(..., read_from_device=False, async_read=True)`. The model call at `1360–1365` therefore returns before host conversion.
- `P/async_decode.py:1403–1417` then calls `model.read_decode_output(..., async_read=True)`. `A:388–390` issues the sampled-ID CPU read with `blocking=False`, increments its counter, and records an existing mesh CQ0 event.
- `P/async_decode.py:1531–1533` waits for that event during `finalize_decode`; `1555` calls `process_decode_output_host`. The adapter converts the already-read host TT tensor to Torch (`A:392–401`). This is the existing completion/readback boundary; a timestamp here needs no additional wait or event.
- Host output work still follows: `complete_decode_step` extracts tokens/logprobs (`973–1018`), then `AsyncTTModelRunnerOutput._get_output_impl` builds the `ModelRunnerOutput` and enqueues the completed step (`285–297`).

The actual installed engine preserves deferred execution. `V/v1/engine/core.py:658–686` schedules, executes, samples, and queues a future together with its `SchedulerOutput`; `701–718` pops the oldest pair, resolves the future, and updates the scheduler. `V/v1/executor/uniproc_executor.py:109–111` wraps async output rather than resolving it, and `AsyncOutputFuture.result` at line 44 calls `get_output` lazily. Thus neither `execute_model` return nor async `sample_tokens` return establishes completion.

**Consequence:** timing just `decode_forward`, recording a timestamp at `read_decode_output` return, or measuring only the duration of the final wait would omit different portions of the requested phase. Successful completion of the original async output implementation is the output-ready boundary after its normal wait, conversion, token extraction, and output construction. It is not the end of every host obligation: deferred application to runner state and scheduler consumption occur later and need separate ownership in full phase accounting.

### 2. Prefill's existing timer is narrower than the required phase, and its name overstates intermediate completion

Direct observations:

- `A:240` starts timing after cache validation and after all plugin input preparation. `A:325` emits `prefill_complete` before returning to the plugin. It omits plugin preparation, output construction, scheduler processing, and inter-step gaps.
- For a final prefill group, `A:314–323` sets sampling state, samples on device, performs the normal blocking ID read, converts it, and fills result rows. Its local duration therefore does include the normal final sampled-output read; claiming it is always enqueue-only would be false.
- Intermediate groups execute `continue` at `A:308–309`, skipping all sampling/readback. The all-intermediate call returns CPU placeholders. The plugin also produces placeholder tokens at `P/model_runner.py:3986–3998` and suppresses them in `_build_chunked_prefill_output` (`3567–3625`). Neither Python return proves those device writes finished.
- `P/prefill.py:48–78` can split one scheduler step into intermediate and final adapter calls, processing intermediate rows first. One adapter call is consequently not always one scheduler step.

**Stage 11 reachability check:** the ordinary fresh 4096-token workload is more favorable than the general chunked path. `tests/serve_vllm.sh:16–19` requests block size 4096 and scheduling budget 8192. `tt/vllm_cache_specs.py:23–38` enforces aligned Mamba blocks/checkpoints of 4096 and no internal checkpoint blocks. With a per-request scheduling cap of at least 4096, installed `V/v1/core/sched/scheduler.py:472–479` rounds a partial cold 4096 prompt end down to zero; `1083–1092` defers that admission. An admitted ordinary cold row reaches end 4096 and has a false intermediate mask (`P/model_runner.py:2150–2153`). At B32, the model's internal 1024-token pieces (`A:123`, `286–305`) remain inside that final call and are followed by its normal readback. At B1 the configured internal chunk is 4096.

That is a conditional source proof, not a claim that a particular benchmark ran with those effective settings. The collector must retain actual prompt ends, caps/configuration, starts, masks, cache-hit/replay status, and final reads. Changed prompt lengths, sub-block caps, prefix-cache paths, or preemption/replay must not be silently treated as the cold case. A one-token final prefix tail is still **prefill**, not decode (`P/model_runner.py:2089–2126`).

The scheduler can interleave unrelated decodes between partial prefills: `P/scheduler.py:829–867`; defaults are two prefill steps then one decode opportunity (`P/config.py:635–637`). If all-intermediate prefill is followed by decode before a natural prefill completion observation, separate exclusive phase completion times are not established by the available markers. Keep that segment unresolved unless an existing boundary proves its split. Do not insert a synchronization or assign its device tail to decode by assumption.

### 3. Current events lack stable identity and add synchronous logging cost

`A:156–167` writes each event using `time.time()` and an immediate `open`/JSON serialization/write/close. `decode_submit` additionally serializes active scheduler table rows (`A:373–379`). The events expose slots, ends, counters, and trace IDs, but no request ID or unique submitted-step ID. Slots are reused/remapped; one prepared trace and the same persistent output tensor are reused for many steps. Those fields cannot independently join measured requests to their deferred completions.

The local duration uses `perf_counter`, but event positions use wall clock. The plotting script reconstructs a start as `event.time - duration` (`tests/draw_serving_timeline.py:19–22`); this is diagnostic visualization, not a verified common monotonic phase timeline. Its own docstring calls the data host submission intervals. Existing runtime notes explicitly omit `QWEN38_SERVING_EVENTS` in performance mode because of its logging cost (`doc/vllm_integration/runtime_audit.md:41–43`).

**Consequence:** enabling the existing diagnostic stream is not the requested lightweight collector. Buffer compact monotonic records and flush outside measured work. Do not perform table serialization, per-token file writes, extra tensor reads, or device queries to obtain timestamps.

## Minimally sufficient instrumentation design

The following describes boundaries, not a source patch. Prefer small plugin-local lifecycle instrumentation plus adapter chunk metadata. Installing monkeypatches from the model initializer would couple the model to private plugin classes; if used for a temporary experiment, pin exact revisions, make installation idempotent, and validate every wrapped path. It is not enough to wrap only the adapter's public methods.

| Boundary | Existing code | Information to retain |
|---|---|---|
| Scheduler decision begins/ends | `V/v1/engine/core.py:658`; TT scheduler `schedule` | Monotonic schedule times, unique step identity, scheduled requests/tokens, lifecycle events. Needed to include and attribute scheduler/transition overhead. |
| Runner entry, before drain/build | `P/model_runner.py:2902–2924` | Step start; drain begin/end when the existing branch runs. Starting only at the model call loses preparation. |
| Input built | `P/model_runner.py:2924–2928` | Snapshot phase, request order, slots, positions, actual chunk ends/lengths, masks, logical and physical batch, mesh/rank. Empty work is a separate event, not a decode. |
| Prefill adapter calls/internal pieces | `P/model_runner.py:3752–3825`; `P/prefill.py:48–78`; `A:256–305` | Parent step ID, child call/chunk ordinal, actual row membership and `[start,end)`, whether the call has a final read. Do not force a completion for suppressed chunks. |
| Decode submitted/read queued | `P/async_decode.py:1360–1440` | Diagnostic submit/read timestamps attached to the same step. Preserve the original device output and read behavior. |
| Synchronous output ready | Original callback popped by `P/model_runner.py:3648–3649`, after `_finish_front_packed_sync` returns | Prefill/sync-decode output-ready timestamp, emitted versus suppressed rows. For intermediate prefill this is host-ready only, not a device fence. |
| Async output ready, exactly once | Successful end of `P/async_decode.py:285–297` | Timestamp after normal completion, host conversion, output build, and completed-queue publication. |
| Deferred state application | `P/async_decode.py:848–868,1072–1115`; called from `P/model_runner.py:2622` | Begin/end, originating step ID **and phase**, applied/skipped/discarded status. Final prefill uses this queue too; its class name does not establish decode work. |
| Scheduler consumes matched output | `V/v1/engine/core.py:716–718`, or existing TT scheduler output-update boundary | Accepted/discarded/finished rows and timestamp. Preserve the original pairing to `SchedulerOutput`. This separates output readiness from later scheduler bookkeeping. |
| Collection outside measured work | Stage collector invocation, not a per-token callback | Flush buffered records, counts, schema/revisions/configuration and dropped-event/exception status. |

Observe the lifecycle already in use; do not call `get_output`, `ensure_finalized`, `sample_tokens`, or `event_synchronize` early. `DeferredDecodeOutput.ensure_finalized` protects exactly-once completion under a lock (`P/async_decode.py:218–234`). Both executor consumption and transition drains can enter it; drains call it directly (`892–902`). A wrapper around `get_output` alone misses drain-driven completion. Timestamping inside the original exactly-once implementation avoids duplicate events when the same cached result is returned again. Record failures distinctly; a finally block must not mark failed readback as successful completion.

The existing `SubmittedStepContext.submit_time_ns` (`P/async_decode.py:539`, captured at `1136`) starts after input construction. `CompletedDecodeStep.completion_time_ns` (`1016`) precedes output build/enqueue. These are useful subordinate timestamps, not full phase denominators. Final-prefill deferred-state records set both fields to the same `now` (`P/model_runner.py:4471–4481`); deriving prefill time from that structure would give zero. Preserve the original step ID and phase on those records as well as on ordinary decode records.

For runner-scoped timing, explicitly say which initial admission and final HTTP handling are outside scope. To claim scheduler-inclusive full host phases, retain scheduler decision/consumption markers as well. Preserve inter-step gaps across those boundaries rather than summing method runtimes. No need to change vLLM core source: the plugin owns TT scheduler/runner hooks; an integration using runtime wrappers must respect the plugin's existing metadata/serialization mechanisms.

## Request and step correspondence

Use a monotonic per-runner step sequence combined with a run/server UUID and rank/PID. A single global `current_step` or `decode_started_at` field is unsafe: the next forward can be submitted before the previous output is resolved. Carry the record with `_SyncForward`, `TTDecodeSubmission`/the async wrapper, or an equivalent per-forward closure in the existing `_pending_samples` FIFO (`P/model_runner.py:2943–2956`, `3648–3649`). Request IDs alone are also insufficient because one request has many steps.

At input construction, retain copied scalar metadata:

- Ordered request IDs and row index, request incarnation, prefill state slot or post-remap decode slot; the step's slot remap.
- Phase from `model_input.prompt_lens is None`, not token-count heuristics. Prompt `prompt_lens` is the scheduled **end position**, so logical chunk length is `end - input_positions` (`P/model_runner.py:2133–2153`).
- Input position, prompt length, scheduled work count, intermediate/final mask, prefix/replay status, physical batch/bucket, participating devices and selected precision/configuration identity.
- Submission/read/output-ready/consumption times; sampled, published, discarded, and accepted row counts separately.

`SubmittedStepContext` already copies request IDs/index mapping (`P/async_decode.py:519–540`). It and `TTModelInput` are shallow frozen containers: referenced lists/dicts/tensors can still change. Snapshot small host scalar metadata at submission, rather than reading current runner rows in a completion hook. Persistent batches condense and recycle rows; state slots are separately allocated/remapped (`P/model_runner.py:2478–2490`). `ModelRunnerOutput` can later have late rows suppressed (`P/async_decode.py:1088–1096`), so an event must copy the observed values and, if necessary, record their later accepted/discarded status.

Do not key a pending map by `id(device_output)`: this model returns persistent `G:self.tokens` on successive submissions (`G:830–831`). A FIFO assumption alone also needs the exact plugin lifecycle; prefill calls and internal reads share adapter methods with decode. Explicit step ownership is safer than counting calls or pairing the nearest timestamps.

For actual decode positions, distinguish scheduler metadata from consumed resident inputs. With `reload_inputs=False`, `G:805–808` deliberately uses resident state; the host `tokens`/`start_pos` arguments are not authoritative. The host contract test even poisons these arguments (`tests/test_serving_generator_contract.py:87–106`). Anchor a per-request position ledger at the last authoritative reload, advance it once per successfully submitted active replay, and apply the observed remap/prefill/release transitions. The adapter already reconstructs `slot_ends` this way (`A:204–217`, `342–346`); for the hybrid serving path its post-submit end minus one identifies the consumed position. Copy that metadata with the step and compare it to scheduler positions instead of silently assuming equality. No device position read is required.

### Existing client ID path can exclude warmups and probes

The actual installed client already offers `--request-id-prefix` (`V/benchmarks/serve.py:1844–1849`). For the random dataset, measured IDs are `prefix + index` (`V/benchmarks/datasets/datasets.py:659–666`); `serve.py:1064–1077` passes them into the request function, and `benchmarks/lib/endpoint_request_func.py:147–148` sends `x-request-id`.

For the completions endpoint, `entrypoints/serve/engine/serving.py:117–126` honors that header. `entrypoints/openai/completion/serving.py:143,179` constructs `cmpl-<client-id>-<prompt-index>`, then `v1/engine/input_processor.py:261–279` normally appends an eight-character random suffix while retaining `external_req_id`; `v1/engine/async_llm.py:475` invokes this assignment. For one prompt with `n=1`, a typical internal form is `cmpl-<prefix><index>-0-<nonce>`. Keep the observed mapping and validate uniqueness; do not disable ID randomization or match only a loosely shared substring.

The initial single-prompt test constructs a separate `RequestFuncInput` **without** the dataset request ID (`V/benchmarks/serve.py:837–868`). Therefore an explicitly recorded, unique measured prefix separates it from the measured dataset, assuming no custom global header/body overrides. Use a different retained prefix for warmup. Check the actual backend and request options before applying this mapping; other endpoints and `n>1` have different fanout.

The initial Stage runner command (`S/runtime/benchmark_stage/run.py:146–155`) does not itself set or persist this prefix, and `--save-detailed` does not retain a request-ID list. The initial-revision integration therefore still needs a wrapper or frontend mapping capture. The concurrent revision's new client wrapper addresses this part, as described separately below.

`S/runtime/benchmark_stage/run.py:141–155` runs a warmup client and then a measurement client, with measured request count `max(8, 3 * concurrency)`. Each invocation can have its own initial probe. Do not select all server traffic between client process start/exit, assume request zero is the probe, or use slot/time proximity as identity. Detailed client `start_times` help only after proving a common monotonic clock and correspondence; client TTFT/TPOT cannot substitute for these phase timings.

## Accounting full wall phases without adding synchronization

**Use scheduler service episodes on one host timeline, not overlapping request lifetimes.** A request awaiting more decode tokens can coexist with a prefill-only step for another request. Its entire first-to-last-token latency includes that prefill time and cannot be charged entirely to decode. The TT scheduler chooses homogeneous prefill or decode steps (`P/scheduler.py:289–295`) but switches between them; HTTP concurrency does not identify those switches.

An implementable accounting procedure for the observed final-prefill/ordinary-decode path is:

1. Select the measured request set using the explicit mapping, and reconstruct all submitted steps, their members, and their existing completion observations. Retain transitions, empty scheduler passes, and late results. Include actual work for those requests even if a late row is discarded; report useful and executed work separately.
2. Build provisional ordered service episodes using the actual scheduler/model-input phase. Adjacent or overlapping steps of the same phase form one episode from its phase opening through the last existing completion/host handling. Carry the episode across intervening scheduler/host gaps while that measured phase has pending/runnable work. Do not remove a gap simply because no model method is executing. Deferred host work is allocated separately in step 5, so final phase intervals can be noncontiguous.
3. At decode-to-prefill transitions, pending decode is naturally finalized before the prefill input is built (`P/model_runner.py:2918–2924`). Record that drain and retain it in the outgoing decode episode; do not start charging all time at the new prefill `execute_model` entry to prefill. Open the prefill service interval after the established outgoing completion/transition boundary, retaining input preparation and output handling.
4. At final-prefill-to-decode transitions, use the existing final read/output lifecycle to close prefill. Include the gap from that boundary to the first decode submission in the decode episode when decode work remains runnable. Apply the same declared rule to subsequent gaps and phase switches. Initial admission before any measured phase, idle periods with no measured work, final HTTP delivery, and collection are separate residual categories, not unexplained subtraction from a phase.
5. On a common monotonic time axis, split at observed phase/transition boundaries and explicitly account for deferred host tails. `build_model_input` updates lifecycle state at `P/model_runner.py:2614`, applies older completed results at `2622`, then prepares inputs at `2627`. An old decode's state-apply span can therefore occur inside preparation for a new prefill, and a final prefill's state-apply span can occur inside a later decode. Tag each apply span with its original step/phase; carve it out of the surrounding service interval and credit it to that original phase exactly once. Do the same for the matched scheduler-consumption span if claiming scheduler-inclusive timing: the engine can consume an old decode only after a newer prefill has executed. Retain a single explicit ownership rule for common scan/prune/lock/transition overhead and any overlapping host subspans. Coalesce the resulting same-phase pieces and sum disjoint intervals once. Do not assign the entire delay between output readiness and late consumption to the old phase: that delay may contain real work of the other phase. Treat final cleanup beyond the declared measured window separately and expose unresolved tails rather than extending the window silently.
6. If an intermediate-only prefill remains in flight across a phase switch with no existing completion observation, retain an unresolved mixed interval. A later decode completion bounds both, but cannot reveal their separate wall split. Similarly, report foreign-request interference or unjoinable events. Do not invent an exclusive split, add a wait, or declare the row valid because the JSON schema accepts it.

For example, two decode steps with `[submit,ready]` intervals `[1,5]` and `[3,7]` have raw duration sum 8, but overlap on `[3,5]`; their union is 6. A later decode step `[10,12]` leaves a host gap `[7,10]`. If the same measured decode episode remains runnable throughout, its full span `[1,12]` is 11, while the union of the three step intervals is only 8. Thus **neither raw sums nor a union of isolated active-call intervals is sufficient**. A prefill service interval inside that span must instead be split out using the actual phase transitions; it must not be charged again to decode.

Natural completion timestamps are host observations, not device hardware timestamps. A deferred future can be resolved after hardware has already finished; that delay belongs to the requested host wall measurement. Conversely, a host placeholder is not a completion observation. The lower runtime confirms the distinction: `ttnn/core/tensor/tensor_ops.cpp:226–244` forwards the CPU read's `blocking` flag to queue read; `ttnn/core/events.cpp:16–29` records a host-visible mesh event and waits through `EventSynchronize`. Preserve those calls and their order exactly.

### Match modeled work to the same intervals

- Prefill useful work follows actual logical prompt/query/context positions and all chunks, with cache hits/replayed history identified. Physical padding and bridge execution remain in the elapsed phase and must be documented as overhead; do not count cached tokens as freshly executed useful work without an explicit definition.
- Terminal prefill emits the first generated token (`A:320–323`). A 128-token response therefore does not imply 128 ordinary decode advances; the usual accepted count is 127 after prefill. Async lookahead and discarded outputs can add executed steps, so use observed records instead of hard-coding either number.
- Estimate decode weight traffic per actual physical model execution and participating chip, accounting for tensor partitioning, replication, physical batch sharing, KV/state operations, sampler traffic and stored dtypes. Multiplying whole-model weight bytes by each request in a shared batch overcounts weight traffic. Conversely, ignoring B32 physical execution because only a few rows are live understates costs that still occur.
- Report useful emitted work separately from physical/executed work, and ensure the numerator and denominator include the same executed step population. If the accounting window closes at accepted output while extra work is still unresolved, do not silently count or exclude that tail on only one side of the ratio.
- Stage formulas remain `100 * useful_prefill_flops / (prefill_seconds * aggregate_peak_flops_per_s)` and `100 * modeled_decode_bytes / (decode_seconds * aggregate_peak_bytes_per_s)`. Select architecture, dtype/fidelity, and peak sources for the actual mesh. This report does not supply a measured denominator or validate a numeric peak.

The old `tests/roofline.py` is a single-layer profiler-derived estimate, not this serving collector. `S/runtime/benchmark_stage/roofline.py:28–66` checks workload/configuration identities, positive values, scope labels, and evidence hashes; passing those checks alone cannot prove timestamp identity or a correct partition. Preserve raw compact events and the interval/work ledger as evidence.

## Integration risks and necessary follow-up validation

- **Flush lifecycle:** `S/runtime/benchmark_stage/run.py:100–115,171–174` invokes collector `check` before accuracy and `collect` after each performance run. Flushing only in `release_persistent_capture`/process exit is too late for a still-live server's collection. Provide a supported host-only collection path that snapshots completed records outside measured work; preserve incomplete records and fail on missing data. Never call a device drain merely to make the telemetry buffer look complete.
- **Hot-path overhead:** buffer bounded compact host records; record drops explicitly. Serialization/file I/O and large table copies belong outside the measurement. Do not retain tensor references merely for telemetry, since that can change lifetime/memory behavior. Short host locks or per-thread buffers must not serialize the existing decode pipeline.
- **Interface/version coupling:** preserve model signatures, tensor return types, existing events, and exact-once output behavior. Cover sync and async paths, empty steps, exceptions, drain-before-prefill, row remaps, and internal prefill split calls. The actual plugin is an editable independent repository, not a vendored legacy vLLM plugin. Retain both source hashes and the installed vLLM identity.
- **Existing configuration:** this adapter requires TP across four devices (`A:35–36`); do not assume lane-DP or arbitrary model capabilities. Do not turn off async scheduling, prefix caching, chunked prefill, or on-device sampling to simplify timing. Keep output IDs/counters/serving behavior invariant.
- **Plugin contribution constraints:** its `AGENTS.md` requires a specific human request before adding a new plugin environment-variable read, capability, or `--additional-config` key. No such interface is necessary to state these hooks; reusing an adapter variable in plugin code would still be a new plugin read. This report introduces none and requests no approval.

Focused follow-up tests should exercise the mechanism with a fake clock/model and existing host stubs: two overlapped decode submissions; normal and drain-driven resolution of the same output; prefill plus sync decode; an all-intermediate/final split; a one-token final prefill; slot reuse/remap; exception and empty-step paths; probe/warmup filtering; discarded late outputs; and a timeline with same-phase gaps and phase switches. Verify exactly one successful completion per submitted forward, preserved original call order/counts, no added waits/reads/events, and interval conservation without overlap. The plugin repository already has one-token-final-prefill and suppression tests (`tests/test_model_runner.py:37–68,373–381`), useful starting points.

Then a separately authorized ordinary serving run can validate recorded identities, both phase outputs, zero lost events, unchanged async overlap, and actual overhead under the same configuration. Hardware runtime correctness, event timing distributions, and measured utilization remain unverified here.

## Concurrent-change addendum: collector added during the investigation

During final checks, HEAD changed independently to `d84c5effabae003e8cd54b9347cdd34d7515f342` (`Prepare Qwen3.8 final benchmark protocol and full-phase accounting`). The adapter SHA-256 became `f65c8e4b8d1c86cdfe4bfcf7c03af80345e40fe7ddfaaca07c681ebc2191c371`. Its only adapter delta is installation of `tt/benchmark_phases.py` at new lines 61–63. The new files were inspected read-only after they appeared. No new collector or target was executed by this investigation.

Positive findings in this revision:

- `tt/benchmark_phases.py:153–170` wraps `AsyncTTModelRunnerOutput._get_output_impl`, so the existing once-only finalization lock protects the completion event. Records follow retained context identity (`134–160`), not the reused device tensor. Completion is observed after the existing readback/output construction with no added device call.
- Records are buffered, and `34–45` provides a host-side signal flush instead of per-step writes. This is an implementation choice to validate, not evidence that a live server is quiescent merely because the client finished.
- `tests/benchmark_client.py:10–15` passes and persists a prefix differentiated by result filename, including warmup versus measurement. `tests/benchmark_roofline.py:97–110` joins internal IDs and rejects steps mixing selected and unselected requests. This supplies the explicit mapping missing from the original generic Stage command; the new wrapper chooses a separate client environment, whose complete runtime behavior was not tested here.
- `tests/benchmark_roofline.py:116` rejects intermediate prefill rows. Therefore intermediate-only enqueue completion is a general hook caveat, not proof that this collector accepts an invalid intermediate-prefill denominator.

**Remaining concrete accounting discrepancy:** `tests/benchmark_roofline.py:107–109` replaces a step's output-ready end with its later scheduler-consumed end. `partition`, at `42`, starts prefill only at `work_start`; at `48–51` it resolves every overlap by assigning the entire overlap to prefill. The hooks provide no drain endpoint, no deferred-state-application spans, and only a consumption **end**, not its begin (`tt/benchmark_phases.py:78–84,88–91,163–170`). Those data cannot reconstruct the complete transition ownership described above.

Two source-reachable sequences demonstrate why the convention is insufficient (times below are symbolic, not measurements):

1. **Decode to prefill:** D1 is pending. P2 is scheduled at `t1`; P2's runner entry naturally drains D1 to output-ready at `t2` (`P/model_runner.py:2918–2921`). P2 input construction then runs until `t3` (`2924`); the new collector takes its `work_start` only now, before `_forward_with_model_input` (`tt/benchmark_phases.py:90,116–119`). P2 is ready at `t4`. The engine then consumes older D1 at `t5` (`V/v1/engine/core.py:701–718`). The collector extends D1 through `t5`, but P priority only begins at `t3`. As a result, **all P2 input construction `[t2,t3]` is charged to decode**. Some old-D state application within that build belongs to D; P's own preparation does not. Without the sub-boundaries these cannot be distinguished.
2. **Prefill to decode:** P1 is output-ready at `t0`; D2 is scheduled and submitted before the engine consumes P1 at `t2`. Extending P1 to `t2` and applying unconditional P priority charges **D2 scheduling/preparation/submission, and the intervening host gap**, to prefill, except for genuinely tagged old-P bookkeeping. This sequence is permitted even though P1's future is ready: the engine returns early to fill its batch queue at `V/v1/engine/core.py:684–692` without testing `future.done()`. Async scheduling selects two concurrent batches for the ordinary non-pipeline configuration (`V/config/vllm.py:589–599`).

These are accounting errors under the declared full-phase semantics; no device timing claim is needed to identify them. Depending on the transition, the split moves in either direction. The total partition can still equal the overall span and contain no overlaps. Thus the retained synthetic test's total-time conservation (`doc/benchmark/subset-source/phase-partition-host-check.json:49–54`) does not establish correct phase ownership.

There is also an unresolved gap classification: `tests/benchmark_roofline.py:55` assigns every uncovered gap to the next phase, but the emitted schema has no pending/runnable/admission state. In sequential B1 traffic, a gap can include final HTTP delivery and client turnaround before the next request reaches the server. The collector cannot distinguish that request-free time from a required service gap using the present records. Retain the missing state and residual/idle intervals instead of treating all time between the first and last selected steps as phase service.

The new host hook check is narrower than these sequences: `tests/benchmark_protocol/check_phases.py:90–102` consumes prefill before submitting decode and directly invokes `_get_output_impl`. It does not exercise the installed engine's batch-queue transition, deferred state application, or repeated/drain-driven `ensure_finalized` resolution. Its passing result would not refute these source-derived counterexamples. Add those cases before relying on its interval totals.

**Smallest implied repair:** retain the existing output-ready time; add markers for the already-existing transition drain and tagged state-apply/consumption begin/end spans. Construct exclusive intervals from those boundaries, including same-phase gaps, and assign only the actual older-phase bookkeeping span to that older phase. Do not stretch a step across intervening other-phase work and resolve it with fixed P priority. Reject unresolved cross-phase spans. No extra synchronization, profiling, serving-mode change, or new plugin configuration key is required.

The new per-request position ledger is not promoted as a normal-path bug: prefills reset it to the scheduled end and request-ID ownership survives slot remaps. It does advance before successful submission, so exception/retry behavior needs validation, but no continuing-success divergence was proved. Likewise the mutable `current` helper is only used while snapshotting forwards on the inspected inline single-process executor path; per-output records are subsequently distinct. A generic warning about global state is not proof of a race in this execution mode.

The concurrent commit also contains an earlier draft copied to `doc/benchmark/AUTODEBUG.md`. This root report includes the subsequent independent review and concurrent-change findings; the earlier copy is not the final reviewed report.

## Other Potential Issues and claims rejected on review

- **Rejected:** “No prefill timing exists.” A local prefill duration exists; its scope/completion semantics are the discrepancy.
- **Rejected:** “All prefill timers are enqueue-only.” Final sampled prefills perform a blocking read; only suppressed chunks lack that boundary. Normal cold 4K scheduling is expected to use the final path under the documented caps/alignment.
- **Rejected:** “The plugin has no completion timestamp.” It does, but existing timestamps are too narrow and do not constitute phase accounting; prefill deferred-state records even use identical start/end times.
- **Rejected:** “A slot, trace ID, output tensor, or request ID alone is a step key.” Reuse and repeated steps disprove this. Request/step and incarnation correspondence must be explicit.
- **Rejected:** “FIFO means the current runner batch is safe to read at completion.” FIFO output resolution does not prevent newer submission, row compaction, or deferred state application.
- **Rejected:** “Sum client latencies, or subtract adapter prefill time from benchmark duration.” Neither establishes which requests, host gaps, overlapping steps, or phase switches those intervals contain.
- **Unproven:** existing diagnostic logging materially changes this model's throughput by a particular amount. The synchronous extra work is visible in source, but its runtime cost was not measured.
- **Unproven:** generalized intermediate-prefill ambiguity occurred in a Stage 11 performance run. No such runtime trace was supplied or produced; the ordinary 4K path has the conditional final-read proof above.

## Review result

The first-draft claims were checked back against the adapter, generator, actual plugin, installed executor/engine, scheduler alignment, client ID path, and benchmark contract. The headline diagnosis is an instrumentation scope/identity gap, not a model numerical bug or a demonstrated device synchronization bug. The smallest adequate repair spans existing plugin lifecycle hooks plus buffered metadata; adding an adapter-local timer alone cannot meet the stated full-phase contract.

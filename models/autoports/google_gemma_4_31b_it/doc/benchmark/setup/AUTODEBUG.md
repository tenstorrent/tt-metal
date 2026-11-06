# AutoDebug: stage-11 full-phase roofline instrumentation

## Findings

**The reported instrumentation gap is present.** The generated adapter emits sampling-mode and memory logs, but no monotonic phase boundaries, immutable request/step association, or decode position history. Existing logs cannot supply the benchmark contract's full-phase denominators. Adding timers only around `decode_forward` would produce enqueue latency on the served asynchronous path, not completion latency.

**A second, important constraint is the observation boundary:** direct adapter hooks can measure completed model/readback work and intervening gaps, but cannot alone observe the stock plugin's subsequent host sampling and final token handling. Request IDs also are not passed through the public adapter API. A complete general collector cannot honestly be claimed from entry/exit timers alone. Below is the minimal safe adapter instrumentation and the precise remaining observation gap; do not silently label its narrower timings `full_phase_wall_time`.

Confidence: high for these source findings. No hardware execution or performance measurement was performed. This is an inspection report, not an implementation.

## Scope and source identity

Inspected working tree at tt-metal HEAD `6d7f2481460e4f48ec21a4d92ee84fa588936157`; plugin checkout `/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09`, HEAD `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. Existing unrelated modifications were left intact. References below use:

- `A`: `models/autoports/google_gemma_4_31b_it/tt/generator_vllm.py`.
- `G`: the adjacent `tt/generator.py`.
- `P`: the plugin checkout's `src/vllm_tt_plugin/`.
- `C`: `/home/mvasiljevic/.codex-gemma4-v3/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/benchmark-model/references/run-contract.md`.

Read `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, the AutoDebug skill and C. No removed Gemma implementations, credentials, hardware, profiler, remote publication or AutoDebug launcher were used. Only this report was written.

## Direct observations versus interpretation

| Direct source observation | Consequence |
|---|---|
| A:232–244 prints sampling mode; A:255–274 prints allocator/counter information. Neither records time or request identity. | These are diagnostic logs, not wall-time evidence. This is the immediate stage-11 deficiency. |
| A:370–388 delegates submission/readback. G:569 and 573 execute traces with `blocking=False`. | Adapter submission return cannot end an async phase. |
| G:580–581 queues `cpu(blocking=False)` and returns an existing recorded event for async reads. P/async_decode.py:1656–1658 waits those events before :1680 invokes the adapter's host-processing hook. | Host-processing entry is an observable post-read-completion upper bound, without any new synchronization. |
| P/async_decode.py:1035–1065 performs output extraction/sampling after `finalize_decode`, then timestamps `CompletedDecodeStep`. | Adapter host-processing exit precedes plugin-level token completion. |
| P/model_runner.py:4094–4139 and P/async_decode.py:1344–1407 construct adapter kwargs without request IDs. | Slots, page IDs, output tensor identity, and batch size cannot substitute for request identity. |
| P/model_runner.py:3208–3222 allows one-step host-stale steady decode. G:543–545 ignores supplied positions when inputs are not reloaded; G:152 advances resident positions. | Logging every supplied `start_pos` as the executed position would miscount KV traffic. |

Interpretation: there is no evidence of a broken device computation here. The defect is absent accounting and insufficient observation if one instruments only the obvious method returns.

## Actual completion boundaries

| Path | Earliest useful existing boundary | What it includes / excludes |
|---|---|---|
| Device-sampled prefill | After A:319 has constructed and validated the returned host tokens | G:408–419 covers all internal prompt chunks; G:434–438 runs the sampler and blocking token read. Includes device sampling/readback and adapter work. Plugin output extraction and state handling still follow. |
| Host-compatible prefill | A:320 return | G:428–429 materializes host logits for submitted rows. Host sampling remains outside the adapter; this return is **logits ready**, not sampled-token completion. |
| Async device decode | A:390 entry, then exit after A:392 | Entry follows the stock plugin's existing event wait. Exit includes conversion and validation. It is a host observation, not an exact accelerator finish timestamp. Subsequent plugin token handling remains outside. |
| Async host-logit decode | Same host-processing hook | Logits are ready; plugin host sampling still follows. |
| Direct synchronous `read_from_device=True` decode | A:381–384 return | G:571–575 performs existing blocking logits/token read. Device-sampled tokens are ready; host-logit mode still needs host sampling. |
| Stock synchronous submission with `read_from_device=False, async_read=False` | Exit of A:390–392, not decode return | P/model_runner.py:4190–4199 immediately finalizes the device tensor. With no read events, the conversion's existing read materializes it. The hook's **entry** is not necessarily post-device-completion in this branch. |

Do not add `synchronize_device`, event waits/queries, `.cpu()`, `to_torch`, or profiler reads for measurement. Reuse the existing calls and timestamp their boundaries. Do not turn off async scheduling, chunking, or host compatibility sampling.

## Minimal adapter-only event implementation

This is a proposed instrumentation design, not a completed fix.

1. Gate collection behind an explicit measurement setting. Record process/session ID, monotonic clock identity, source identity, actual participating mesh device IDs/topology, capacity, layer selection and precision once. Buffer small immutable host records; flush only outside measured work through an explicit collector mechanism. A setting alone or an empty file is not readiness evidence.
2. Allocate a monotonically increasing local step ID at entry to each prefill/decode call, before page refresh, sampling setup and existing logging. Record start and return times using the same clock (`perf_counter_ns` aligns with the plugin's existing timestamps). Record exceptions as failed/incomplete steps, never successes.
3. For prefill, snapshot per-row start, end, and `end-start`, sampling mode, row mapping and intermediate/final mask when available. `prompt_lens` is the scheduled chunk **end**, not necessarily total prompt length (P/model_runner.py:2441–2461). Capture every scheduler call, including intermediate chunks that emit no output. Distinguish host-policy selection from actual sampling: an all-intermediate batch bypasses the host sampler (P/model_runner.py:4311–4323); mixed batches sample final rows and suppress intermediate outputs. Record sampled-row counts separately. One interval around the complete adapter call covers all its internal 1024-token/partial-page chunks (G:296–340); do not add individual chunk durations to it. Retain internal shape/group descriptions separately for work accounting.
4. For decode, snapshot reload flags, supplied positions, active rows and effective resident positions. On successful full reload, initialize a host position shadow from authoritative `start_pos`; on every accepted replay advance nonnegative shadow positions by one, matching G:152. On no-reload steps use that shadow, **not** potentially stale supplied positions. Page-only refresh does not reset positions. Record request/layout transitions, and invalidate the ledger on an ambiguous failure. Verify slot/remap behavior against actual position updates; do not assume a sampler-seed remap itself moves position state.
5. Associate each decode submission with the immediately following `read_decode_output` on its producer path. Store the local step metadata against the **returned host output object**, retaining it until processing to avoid `id()` reuse. The device output object is persistent `self.tokens` or `self.logits` (G:572–575), so a map keyed only by its identity will overwrite earlier in-flight steps. A single mutable `current_step` is also unsafe because submission and finalization can overlap.
6. In `process_decode_output_host`, recover that immutable record, record entry and exit, perform the existing conversion once, and retire the association. Protect shared bookkeeping with a short host lock; never hold it across device work. Keep a separate association for the synchronous no-read-hook branch. Unknown, duplicate or missing completions fail collection rather than falling back to the newest step.

### Request identity without plugin mutation

The public adapter API cannot provide stock request IDs. A narrowly scoped **read-only caller-frame snapshot** is technically possible at this pinned revision: the immediate caller of prefill is `submit_prefill` with local `model_input`; the immediate decode caller is `submit_decode` with the same local. `TTModelInput.row_req_ids` and `live_rows` describe request-to-row correspondence (P/model_input.py:210–214,255–278).

If this private observational bridge is chosen, guard the exact caller code/module and plugin revision, confirm the caller's runner owns this adapter, and check tensor/argument correspondence. Copy only primitive metadata and CPU arrays, then release the frame immediately. Fail closed when fields or mappings disagree. Do not retain frames, read a later mutable `runner.input_batch` during completion, patch plugin methods, or guess identity from page reuse. This is revision-specific instrumentation, not a supported plugin interface. It recovers submission identity but does **not** solve the post-return completion gap.

Client-to-server identity still needs proof: retain an explicit measured request mapping, or a demonstrated same-clock mapping with detailed client start times and unique correspondence. A process start/end window includes probe/warmup traffic and is insufficient. Stock chunk logs are not a substitute: P/input_batch.py:62–108 logs only intermediate/continuation chunks, omitting full new prefills and new-request prefix hits.

## Nonoverlapping accounting and remaining full-phase gap

Keep all raw submission, read-enqueue, host-processing and available scheduler-completion markers. Do not sum `completion-submit` over steps or requests: steady decode can submit the next step before the previous output is finalized (P/async_decode.py:1178–1215). Such sums double-count overlapped wall time.

For a proven contiguous single-phase epoch, use one elapsed span from its first start to its last valid completion. Equivalently use the first span followed by completion-to-completion intervals; this includes inter-step host gaps once. Treat these as accounting intervals, not individual kernel latencies. Do not bridge unrelated idle periods or warmup/accuracy traffic. Preserve every work-bearing step once, including in-flight work later discarded, and distinguish executed work from client-visible output counts.

For prefill/decode transitions, construct disjoint intervals on the common timeline from actual scheduler submissions and completions. The plugin drains pending async steps before a non-steady transition (P/model_runner.py:3208–3222; P/async_decode.py:944–972), but the adapter starts **after** that drain and input construction. Do not claim the adapter sees when the scheduler decided to transition. Charge gaps under an explicit evidenced phase policy, and preserve unattributed gaps if the boundary cannot be established. Never assign concurrent phases from HTTP TTFT/TPOT or concurrency alone.

**The unresolved constraint is real:** host-compatible prefill sampling runs later in `_get_output_tokens` (P/model_runner.py:4219–4248,4438–4444). Async decode's `CompletedDecodeStep.completion_time_ns` is captured only after `_get_output_tokens` (P/async_decode.py:1035–1065); runner-output construction and state application can follow even that marker. The adapter has already returned by these points. The next adapter entry can envelope some of that work, but cannot identify its end or cover the final step's tail, and can occur before a prior completion during overlap.

Those plugin timestamps are useful evidence **if exported with complete identity**, but merely reading its transient completion deque at the next adapter call is not a reliable export: P/async_decode.py:807–812 drains it, and a final step need not have a next model call. Polling it cannot guarantee lossless capture either. Retaining caller frames or monkeypatching plugin completion methods would not be a minimal direct-hook solution and is not recommended under the no-plugin-runtime-change constraint.

Therefore:

- Direct adapter instrumentation can supply correct completed-device/readback events and a useful gap-inclusive adapter envelope.
- A read-only pinned caller-frame bridge can additionally supply request identity.
- Neither alone proves the required general full-phase sampling/token-handling timeline. A lossless existing external completion export or an explicitly allowed observational integration at the plugin's completion boundary is still needed. No such export was established in the inspected path. Do not declare stage 11 complete with adapter-only partial denominators; report this exact limitation if the observation constraint remains strict.

## Collector contract and work accounting

C:225–308 requires `roofline_command` as an argv array with `--run-dir`, `--concurrency N`, and `--action check|collect`. `check` runs on the selected live server before accuracy/warmup; `collect` runs after measured traffic and before the server switch. Readiness must prove both phase events, identity linkage, work accounting, and a usable flush/export path. Missing instrumentation is setup work, not a reason to omit roofline numbers.

`roofline.json` is a mapping keyed by batch strings (`"1"`, `"8"`, `"16"`, `"32"`), preserving earlier collected profiles. Each entry requires:

- `performance_sha256`: SHA-256 of exact `perf-bN.json` bytes.
- `server_identity_sha256`: `benchmark_stage.subsets.digest` of parsed `perf-bN-server.json` (not an assumed equivalent raw-file hash).
- `requests`, `input_tokens`, `output_tokens`, matching `completed`, `total_input_tokens`, `total_output_tokens` respectively.
- `prefill`: `flops`, `seconds`, `peak_flops_per_second`.
- `decode`: `dram_bytes`, `seconds`, `peak_dram_bytes_per_second`.
- In **each phase**: `timing_scope: "full_phase_wall_time"`, `timing_method`, `work_method`, `peak_source`, relative `evidence` path and SHA-256 of that artifact's bytes.

Use FLOP/s and bytes/s. Compute percentages as `100 * work / (seconds * peak)`. Preserve timing events, request mapping, interval partition and per-step calculations in the hashed evidence.

Derive useful prefill FLOPs from logical uncached intervals and the actual model layers/attention windows, separating padded execution shapes from useful work. Derive decode bytes from effective positions, actual active rows, stored weight/cache dtypes, physical packing, TP sharding/replication, sharing within each batch and material intermediate traffic. Do not multiply shared weight traffic by request count or use scheduler KV spec BF16 as proof of physical cache dtype: A:138–139 uses and checks `layer.cache_dtype`. Account for the first generated token being produced by prefill and intermediate chunks producing none; output-token count is not decode-step count. Cite and verify peak sources separately; this inspection did not establish numeric work or hardware peak values.

## Review of the diagnosis and follow-up validation

Cross-checked the first-draft claims against both sides of submission and completion. Important qualifications retained above:

- Prefill return really does include its device work/readback; it is not merely another async enqueue boundary. Its missing component in host mode is the later host sampler.
- Host-processing **entry** is post-wait only for the async event path; synchronous conversion can itself wait.
- Supplied decode positions are not always authoritative; resident execution advances independently.
- Nonoverlapping adapter intervals are possible, but are not automatically full scheduler phases.
- Private caller-frame access can recover IDs, so a blanket claim that adapter-local code can never observe identity would be false. It does not add a completion hook after the caller returns.

Cheap follow-up tests should use mocked clocks/plugin calls: two in-flight submissions sharing the same device output but different host readbacks; stale supplied positions on no-reload steps; full reload/page-only refresh; every intermediate/final chunk; synchronous conversion; failed submissions; missing completion; final-tail accounting; and exclusion of probe/warmup/accuracy requests. Assert disjoint intervals, request mapping and token-count reconciliation. No device tests were run here. Actual event ordering, instrumentation overhead, full-profile completeness and hardware byte/peak estimates remain to be validated by the implementing session without adding synchronization or profiling.

## Other potential issues (not headline findings)

Existing per-step `print(..., flush=True)` and memory logging can affect host wall time. Their cost belongs in the currently served denominator; this report does not claim they are a measured bottleneck or authorize changing serving behavior. No unrelated model numerical bug is alleged.

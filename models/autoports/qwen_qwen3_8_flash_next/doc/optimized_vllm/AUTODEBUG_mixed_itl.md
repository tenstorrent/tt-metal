# AutoDebug: Qwen3.8-Flash-Next mixed-traffic client ITL

Inspection: 2026-11-17. Scope: source, completed artifacts, and local JSON/hash/log analysis only. No target imports or execution, hardware access, profiling, live-server queries, process inspection, resets, or source edits were performed. The active budget9 sweep and its execution manifest were not inspected or modified. Only this report was written. No stage-pass verdict is made.

## Headline findings

**The leading diagnosis is a final-prefill publication barrier in the actual async serving path.** With interleave1/1, the engine finalizes a background decode, then executes the next prefill before publishing that decode. Intermediate prefills enqueue device work and return quickly; the final prefill samples and synchronously reads its token before returning. That final call therefore holds an already-materialized background token while another full prefill completes. The gap between consecutive candidate background-publication opportunities spans the penultimate prefill/decode cycle **plus the final prefill**.

This is more specific than “prefill is slow,” “async jitter,” or “the client skips text.” In **all four completed cases**, the largest consecutive foreground `prefill_result` interval is at the final chunk and matches the reported client maximum within **0.8 ms**:

| Budget | Foreground prompt | Physical chunks | Final boundary interval, server log | Client maximum |
|---:|---:|---:|---:|---:|
| 33 | 4,096 | 128 × 32 | 2,557.148 ms | 2,557.9 ms |
| 33 | 32,768 | 1,024 × 32 | 2,598.984 ms | 2,599.3 ms |
| 17 | 4,096 | 256 × 16 | 1,587.290 ms | 1,587.8 ms |
| 17 | 32,768 | 2,048 × 16 | 1,619.917 ms | 1,620.5 ms |

The source ordering is proven; attribution of the exact client maximizing pair remains a **strongly supported inference**, because the client artifacts discarded their timestamps and request/token correspondence. No hardware rerun was performed. The synchronous final-prefill return is consistent with the existing model interface; this investigation identifies a performance interaction, not an established token-correctness defect or permission to remove a required wait.

**Budget17 can fail the client bound even though individual adapter token-validation gaps are below 1000 ms.** Across each completed server log, the largest adjacent `sampled_tokens` event gaps are 1,365.414 ms at budget33 and 881.682 ms at budget17. Those events include foreground samples and are not per-request publication timestamps. Substituting them for client ITL would miss the final-prefill barrier.

**The smallest targeted existing-setting experiment is budget17/alignment8 with interleave1/2 and async still enabled.** An extra complete decode step can give the engine a publication opportunity between the two prefill costs that currently accumulate. It changes cadence, not decode arithmetic or sampling. Budget9 remains an independent, unobserved smaller-quantum arm. If a smaller quantum is still needed, existing budget5/alignment4 and budget3/alignment2 are valid follow-up candidates. None is proven to meet the bound. Do not infer the budget9 result, relax the client metric, or reduce context/model/decode work to declare success.

## Evidence and revision consistency

Path abbreviations:

- **M** = `models/autoports/qwen_qwen3_8_flash_next` in this checkout.
- **P** = `/home/ttuser/qwen38-flash/vllm-tt-plugin`.
- **R** = `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check`.
- **V** = `/home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm`.
- **E33/E17** = M's `readiness_vllm/stage10_chunk_v3_33` / `stage10_chunk_v3_17`.

Repository HEAD is `f29418e64a4e69f71a20f8f51b0a1b0feb4e492d`. The model `tt/` and common sampling/CCL source have no delta from model checkpoint `755bf420` through this HEAD. All **43 model-source hashes** and all **79 performance before-setup hashes** in [full_model_comparison_v3.json](models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/full_model_comparison_v3.json) match the inspected files. Plugin HEAD is `629f07cb385494b14d0ee3d138de039d71d09546`; it has no tracked modifications. Existing unrelated worktree documentation changes and untracked evidence were left alone.

The full-model artifact has `passed=true`, B1, 128 input/128 generated tokens, 127 decode steps, C262144, resident model+sample **102.270285 ms/step**, and serving-boundary resident **102.240057 ms/step**. This establishes its stated generator control, not mixed HTTP latency, B2 serving performance, or a minimum prefill time. Native/CCL and sampling controls were reported as passing; no new device claim is made here.

The two client artifacts directly establish:

| Budget | 4K TTFT | 32K TTFT | Overall client maximum | Request errors |
|---:|---:|---:|---:|---|
| 33 | 170,452.6 ms | 1,380,028.9 ms | 2,599.3 ms | `[]` |
| 17 | 217,058.8 ms | 1,764,016.3 ms | 1,620.5 ms | `[]` |

Sources: [chunk_33.json](models/autoports/qwen_qwen3_8_flash_next/doc/eval_readiness/long_context_timing/chunk_33.json), [chunk_17.json](models/autoports/qwen_qwen3_8_flash_next/doc/eval_readiness/long_context_timing/chunk_17.json). Both exceed the required **1000 ms**.

The [launcher](models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/run_server.sh), completed E33/E17 `vllm_run_summary.json`, and each server's startup configuration agree on full model, one mesh1×4, resident2, C262144, page32, prefix caching off, `sample_on_device_mode=all`, `trace_mode=all`, alignment8, interleave1/1, and async scheduling. Startup logs identify vLLM 0.31.0. `run_long_context.completion` uses `/v1/completions`, token-ID prompts, temperature0, `ignore_eos=True`, four foreground output tokens, and a 128-token background prompt with a 65536-token cap in this sweep. The required stochastic k40 profile is a separate acceptance control; this greedy interference probe does not validate it.

Cleanup success for completed arms is requester-provided evidence. Saved logs show request drain/teardown and trace releases, but also API-server messages force-killing EngineCore during teardown; those messages alone do not prove process/device cleanup. No live cleanup or ownership checks were run, and the active manifest was deliberately left unread.

## What the client actually measures

[R/served_probe.py:154](</home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/served_probe.py:154>) sets request `t0` with `perf_counter`. It reads HTTP lines, decodes JSON, then records a timestamp **only if completion text, chat content, or reasoning text is nonempty** (190–193). `keep_text=False` suppresses retained text, not this filter. Thus the timestamp includes delivery, client scheduling and JSON-processing time. It is not device token completion or every SSE arrival.

[R/served_probe.py:246](</home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/served_probe.py:246>) computes:

```text
max_gap_ms = max(1000 × (b - a))
for adjacent retained background content timestamps a,b
whose interval overlaps [foreground request t0, foreground first content].
```

The entire crossing gap counts; it is not clipped to the foreground window. Initial background silence without a preceding timestamp and terminal silence without a following timestamp do not count. The foreground window ends at first **content**, which can follow the first sampled token. `token_times()`'s synthetic per-token expansion is **not** used here.

[R/run_long_context.py:355](</home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_long_context.py:355>) starts one background completion per requested decoder, sleeps for the configured warmup (default ten seconds), submits 4096 then32768 foreground prompts serially, sets stop, joins the workers, and evaluates each background record over those foreground windows (379–393). It removes the windows before writing the aggregate JSON. Consequently the saved JSON cannot identify the exact maximizing SSE pair, tokens, or client clock interval. `errors=[]` is request-level evidence, not a proof of token timing or continuous background coverage.

This definition is the required client metric. A device interval, mean TPOT, empty-SSE interval, or an interpolated token series cannot replace it when deciding whether the bound passed.

## Scheduler and physical work

[P/config.py:721](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/config.py:721>) defines interleave counts as positive **whole-step counts**. [P/scheduler.py:866](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py:866>) selects prefill-only and decode-only steps. With both kinds of work pending and settings1/1, submission order is `P,D,P,D,…`; this is not a 1000-ms deadline or device preemption guarantee.

[P/scheduler.py:957](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py:957>) hides running decode requests during prefill scheduling, reducing admission caps so the hidden decoder still occupies its sequence slot. Decode scheduling hides partial prefills and waiting requests (983–1012). Therefore **there is no “budget minus one reserved decode token” calculation** for these prefill steps. The whole budget is offered to prefill and a partial endpoint is rounded down by alignment8:

- 33 offered →32 partial prefill tokens.
- 17 offered →16 partial prefill tokens.

Final prompt chunks are explicitly exempt from endpoint alignment ([819–856](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py:819>)); arbitrary nonaligned prompts remain supported. The model partitions each logical submission into its fixed `1,2,4,8,16,32` shapes on the absolute32 grid ([generator.py:810](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py:810)). All spans are submitted before returning; there is no scheduler yield inside the loop (953–979). Intermediate device completion remains asynchronous.

Static parsing of the completed logs confirms the actual dimensions, not just configured labels: **1156 physical32 replays** at33 and **2312 physical16 replays** at17, including background128 and the two foreground prompts. The foreground counts are shown in the headline table. Each foreground logical interval is covered once in these aligned arms; there is no evidence of restored-prefix duplication in them.

[scheduler_chunk_structure.json](models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/scheduler_chunk_structure.json) describes an older alignment0/control arm:125 logical submissions became571 physical replays over4158 positions for a4K prompt, versus128 physical32 replays/4096 positions under alignment8. That comparison also changes prefill interleave2→1 and is explicitly structural, not latency evidence. The cancelled alignment0 32K arm cannot be spliced into the completed alignment8 results.

## Async execution, completion, and publication ledger

The key distinction is **device completion →host materialization →engine publication →client content arrival**. These are different boundaries.

1. **Decode submission:** [P/async_decode.py:1190](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:1190>) submits with `read_from_device=False, async_read=True`. Qwen returns device output metadata; [generator.py:1316](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py:1316) queues a nonblocking CPU copy and records a CQ0 event. `finalize_decode` waits on the event before host conversion ([P/async_decode.py:1653](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:1653>)). Sampling stays on device.
2. **Two-entry engine queue:** [V/config/vllm.py:592](</home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/config/vllm.py:592>) sets two concurrent batches for V1 async scheduling at PP1. [V/v1/engine/core.py:697](</home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core.py:697>) schedules and executes the next step before popping/resolving the older future and calling `scheduler.update_from_output` (739–757).
3. **Worker calls are inline:** [V/v1/executor/uniproc_executor.py:108](</home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py:108>) invokes the worker immediately even with `non_block=True`. `AsyncOutputFuture.result()` later calls `get_output()` on the resolving caller (32–48). A future does not make a synchronous prefill worker call run in another Python thread.
4. **Prefill transition:** [P/model_runner.py:3245](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:3245>) drains pending decode before building/executing prefill. [P/async_decode.py:935](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:935>) explicitly drives `ensure_finalized`, rather than waiting for an engine pop that has not happened. This materializes the background token but does not itself publish it to the HTTP client.
5. **Intermediate prefill:** [generator.py:551](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py:551) enqueues a trace on CQ0 with `blocking=False`; no sampling slots means `prefill_forward` returns `None` (987–988). The adapter logs `prefill_result` at Python return ([generator_vllm.py:136](models/autoports/qwen_qwen3_8_flash_next/tt/generator_vllm.py:136)). This timestamp proves host return, not device completion. The next decode read waits behind queued prefill work in CQ order.
6. **Final prefill:** `prefill_forward` executes the canonical `_sample`, then `_read_tokens` before returning (989–995). [_read_tokens:640](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py:640) materializes tokens through `ttnn.to_torch`; counters explicitly record a readback/synchronization. The runner executes this prefill synchronously in `execute_model` ([P/model_runner.py:3287](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:3287>)). Until it returns, core cannot publish the preceding decode output.

A concrete final-boundary ledger at budget17/4K:

| Host monotonic time | Event | Consequence |
|---:|---|---|
| 2488690.376163 | Background D previous token validated | Token ready on host; previous future still awaits engine publication. |
| 2488690.389705 | Penultimate P returns, without a sample | Core can now pop/publish that older D. Penultimate P device work is queued. |
| 2488690.390511 | Next background D submitted | Its read follows penultimate P work. |
| 2488691.229269 | That D token validated during drain before final P | Token ready, but core has entered next-step execution. |
| 2488691.230320 | Final P starts | Prior D publication remains pending. |
| 2488691.976995 | Final P sample/read finishes and generator call returns | After adapter validation/bookkeeping, core can pop/publish the already-ready D. |

These prefill-return log markers are separated by **1587.290 ms**; source ordering places the corresponding publication opportunities just after them. That D token was ready **747.726 ms** before the final return marker. Observed client maximum:1587.8 ms. The same relationship holds for both budgets and both prompt lengths:

| Case | Server log lines, previous/result final | Host-ready D held until final return |
|---|---|---:|
|33/4K|[3795](models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_chunk_v3_33/server.log:3795),3802|1236.590 ms|
|33/32K|[10124](models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_chunk_v3_33/server.log:10124),10130|1257.229 ms|
|17/4K|[4567](models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_chunk_v3_17/server.log:4567),4573|747.726 ms|
|17/32K|[17094](models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_chunk_v3_17/server.log:17094),17100|769.900 ms|

This predicts the four observed dimensions: shrinking chunks helps; budget17's ordinary token readiness can be below1000ms while client ITL fails; the inferred client maximum occurs at final prefill; and successful requests with vocabulary-valid samples coexist with large gaps. It does not depend on a kernel hang, invalid sample, or unexpected delay in a free-running background finalizer.

Actual async is present, rather than merely requested. For example E33 lines3807–3810 show two consecutive resident device `decode_submission`s at2486863.945053 and2486863.946496 before the next token validation at2486864.151118. TTScheduler inherits AsyncScheduler (P/scheduler.py:327), and Qwen declares async support (generator_vllm.py:29); the startup custom-scheduler warning is conditional generic prose, not evidence that async was disabled. During `P,D` alternation, transitions invalidate the steady chain and force `reload_inputs=true`; this reduces overlap without disabling the async path. [P/async_decode.py:1213](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:1213>) marks the first nonsteady decode unsafe for overlap. Steady no-prefill portions of the same run exercise queued decode. Acceptance must still verify async transition/history coverage under the ultimately selected configuration.

## Text/token effects: real mechanisms, secondary attribution here

The client can observe a longer gap than per-token production even without the final barrier:

- Completion requests default to skipping special tokens. A generated token may produce empty text. Slow detokenization explicitly withholds incomplete Unicode ([V/tokenizers/detokenizer_utils.py:261](</home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/tokenizers/detokenizer_utils.py:261>)); the fast detokenizer can also return an empty delta ([V/v1/engine/detokenizer.py:209](</home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/v1/engine/detokenizer.py:209>)). The active implementation and exact generated tokens are not retained in these client summaries.
- [V/entrypoints/openai/completion/serving.py:371](</home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/entrypoints/openai/completion/serving.py:371>) skips an initial empty chunk only when both text and token IDs are empty. A real token with empty text can reach SSE serialization, then lose its timing in this client.
- [V/v1/engine/output_processor.py:51](</home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/v1/engine/output_processor.py:51>) merges streaming deltas when producer output gets ahead of the consumer. An explicitly larger `stream_interval` can also batch tokens; the installed default is1. SSE `created` is a request-level integer timestamp, not a per-event emission timestamp.
- Client scheduling, transport buffering and JSON processing can shift recorded times.

None of these mechanisms is proved to produce the observed maxima. The initially attractive “roughly2× means an empty-text token” story is unnecessary after matching **all four final-boundary intervals**. Keep it as a competing or additive possibility until per-request timestamps exist. Neither sampled-token count nor anonymous event proximity establishes which client chunk contained a specific token.

## Discriminating experiments and minimal intervention boundaries

These are recommendations, **not executed work**. Any later runs must wait for the current sweep's owner to finish and release its resources. Preserve full48, resident2/C262144, mesh1×4, page32, repaired source, actual async, `trace_mode=all`, canonical device sampling, prompt IDs, request semantics, and the complete decode computation per generated token. Keep the official client-content metric unchanged.

### 1. Existing cadence control: budget17, alignment8, interleave1/2

Change only existing `decode_interleave_decode_steps:1→2`. Source schedule becomes `P,D1,D2,P,…`. D1 after prefill is nonsteady; D2 drains D1 before submission. Core can then pop/publish D1 **before entering final P**, instead of holding the only intervening D until final P returns. The next publication can follow D2 plus final P. This can split the current approximately `P_penultimate + D + P_final` gap into gaps with one P each. Budget17's saved ordinary mixed token-validation intervals below0.9s make it a useful discriminating arm, **not a guarantee** that client gaps will remain below1s.

Expected discriminator: if maxima drop and the final2× boundary disappears while full prefill/decode math, samples and output coverage stay valid, that supports the predicted cadence/publication mechanism. If the maximum persists, record which output was waiting and why; do not assume the source-order prediction establishes runtime passage.

This changes decode service share and usually increases background tokens generated during the foreground TTFT. It preserves the complete decode computation per token. Smaller prefill quanta also create more decode opportunities at interleave1/1: the33→17 change doubles foreground chunk counts. If “unchanged decode work” requires identical *total token count within the timed windows*, both cadence and quantum changes need a separately matched, count-checked workload; neither meets that stricter interpretation automatically. Treat unmatched arms as mechanism diagnostics. Keep the original background cap and verify that it does not expire mid-window.

Interleave1/3 is a separate control if actual steady overlap **inside the prefill window** is needed: D1 is nonsteady; D2 drains D1; D3 can be submitted over steady D2. It adds work/service share and worsens prompt TTFT. More decodes do not make an individual prefill preemptible, and generic empty-text gaps can still exceed a single cycle.

### 2. Existing smaller quantum: owner-completed budget9, then 5/alignment4 or 3/alignment2

Do not infer budget9 from the two completed runs. Once its owner provides the completed result, compare the same two client windows and final-boundary ledger. If it misses or has insufficient margin, use existing settings `max_num_batched_tokens=5, prefill_chunk_alignment=4`, then optionally3/alignment2, keeping interleave1/1. These select existing4/2 model buckets for ordinary partial chunks, retain full decode computation per token and the1:1 step cadence, and preserve final unaligned tails. Additional chunks also create additional decode opportunities.

The current launch script hardcodes alignment8: lowering only the budget below8 triggers the existing permanent-cap validation ([P/scheduler.py:267](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py:267>)). Supply the matching existing tt-config value in a separate future candidate launch. No new interface, default, model capability, or plugin source change is needed. Budget8 and9 both round ordinary alignment8 partial chunks to8; changing the label alone does not yield a smaller regular quantum.

Do not extrapolate linearly. Each span runs the complete layer stack, final mixer/head and padded output copy ([model.py:275](models/autoports/qwen_qwen3_8_flash_next/tt/model.py:275), [generator.py:535](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py:535)). Multirow MoE reduction has physical32 work even at smaller logical lengths ([multichip_decoder.py:893](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py:893)); singleton takes a different numeric path. State-transfer costs also remain. B1 resident102ms is not a per-prefill-token predictor.

### 3. Retain the correspondence needed to prove the cause

On the selected unchanged execution path, record all background SSE events with local receipt time, response ID, choice index, token IDs, finish reason, and cumulative ordinal. Installed completion protocol supports `return_token_ids=True` without logprobs ([V/entrypoints/openai/completion/protocol.py:169](</home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/entrypoints/openai/completion/protocol.py:169>)). Apply it only to the short background prompt to avoid adding32K prompt-ID serialization. Save the official content timestamps, maximizing pair, and foreground window alongside this diagnostic stream. This is a proposed harness extension, not something currently present in the raw chunk JSON.

An empty-text SSE with token IDs inside the content gap supports text filtering. One later SSE containing multiple token IDs supports batching, without locating the batching layer. For the leading diagnosis, retain request/ordinal mapping at decode finalization, scheduler output commit, collector put/get and SSE yield, using existing host time only. Extra plugin or vLLM source instrumentation is outside this inspection and needs separate authorization; do not add a device synchronization, request logprobs, switch sampling to host, or enable a costly tracer to obtain timestamps.

A short otherwise-matched `ASYNC_SCHEDULING=0` control is available in the existing launcher and would test removal of next-step-before-publication ordering, alongside other execution changes from disabling async. It is **diagnostic only** and cannot satisfy the requirement for actual async exercise. It is not a proposed final serving configuration.

### 4. Model-only performance candidates if settings remain insufficient

The smallest clear redundant work is terminal projection for **nonemitting intermediate prefills**: `prefill_sample_mask=False` avoids sampling but `forward_chunk` still computes final mixer/head/logits, and `_prefill_body` copies them to the persistent output. A future model-only experiment can omit that terminal work for intermediate spans while preserving every recurrent/KV/PLE update, terminal sampling, `return_all_logits` semantics, and all decode computation. Output-emission mode must be represented in trace identity and warmup; the existing key of only `count` cannot safely reuse two different graphs. There is no measured saving or proof of crossing1000ms here. Final P still needs its logits/sample/read, so this only shortens part of the boundary.

A larger candidate is reducing full-capacity scratch transfers or using per-slot persistent prefill captures. [slot_state.py:48](models/autoports/qwen_qwen3_8_flash_next/tt/slot_state.py:48) copies recurrent/PLE/history/index state, page tables and full compressed keys, while shared K/V is deliberately excluded. Transfers are already prepared persistent traces (117–135); this is not evidence of unsafe eager allocation. Removing copies requires proving live extents, stable per-slot addresses, compressed-key/history ownership, migration/rollback correctness and restored survivor state. Do not lower C262144, alias scratch with a live cache, or replay a graph against a newly rebound state to save time. Per-copy timings are absent; copies cannot be named the dominant bottleneck from this evidence.

Moving the final token read into a new deferred-prefill protocol, changing engine publication order, or adding a latency-aware scheduler option is a different intervention boundary. Existing runner prefill handling expects materialized synchronous output ([P/model_runner.py:4247](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:4247>), [4320](</home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:4320>)). Returning decode submission metadata or stale placeholders from `prefill_forward` is not an existing supported model-only replacement. Such plugin/interface/source/default changes were not authorized or implemented.

### 5. Nonaligned tails and trace/sampling gates remain necessary

Alignment settings affect partial endpoints, not accepted prompt lengths. Final tails can require multiple physical full-model replays within one scheduler submission: e.g. a five-token remainder can become4+1. Odd continuation starts may restore a saved prefix and replay its last token to preserve the current singleton/multirow arithmetic policy ([prefill_checkpoint.py:79](models/autoports/qwen_qwen3_8_flash_next/tt/prefill_checkpoint.py:79), generator953–972). A successful aligned4K/32K arm cannot prove a worst-case arbitrary-tail bound.

For any candidate, retain mixed nonaligned prompts and continuation starts around1/3/7/15/31/33, including existing129/513 cases, page32 transitions, survivor decode, slot reuse/migration, and final emitted singleton paths. Check raw tokens/state against the established canonical controls, seed progression, inactive slots, narrow/wide sampler-family reuse, trace allocation safety, and actual queued async exercise. Preserve full context capacity and full decode work. Separate cold capture from warmed interference: preparation compiles all six prefill shapes, but `_prefill_chunk` captures a trace lazily on first use. **There are no capture events in either completed foreground interval here**, so cold capture is not the cause of these four maxima; new tail shapes can still require it in later tests.

## Other potential issues and measurement limitations

These are concrete inspection findings or remaining hypotheses; none explains the four measured maxima better than the final-prefill barrier.

- **Background coverage can be under-checked.** A worker stores its record only when its one request returns; warmup is a sleep, not a first-token/live barrier. A naturally finished background request is not restarted. Stop is checked after a new HTTP line arrives; after a120-second join timeout, still-`None` records are excluded from both gaps and errors (R/run_long_context381–393). Some overlap is enough to obtain a number; continuous two-window coverage is not validated. The completed logs show continuing device decode submissions/validations between foreground chunks; `active_decode_slots` alone only indicates resident cache positions. No evidence says missing background traffic produced these failures. Future candidates with more decode share need explicit cap/coverage checks.
- **The raw gate is not a provenance certificate.** `run_chunk` takes the supplied budget label before log extraction, does not enforce async or device-route identity, and returns exit0 even after writing request errors. Timing-stage selection reads every `chunk_*.json` without configuration/source provenance. The model sweep wrapper does reject request errors/missing per-prompt measurements and only chooses a bounded candidate after its loop ([run_optimized_vllm_chunk_sweep.py](models/autoports/qwen_qwen3_8_flash_next/tests/run_optimized_vllm_chunk_sweep.py)). Completed launch summaries/logs corroborate the two investigated arms; aggregate JSON alone would not. No selection or pass is inferred from the still-running sweep.
- **A documentation claim about background threads does not match this installed executor.** P/docs/SCHEDULING.md:548 describes uniproc async outputs resolved by a background `ThreadPoolExecutor`. Installed V/uniproc_executor resolves the lazy future on its caller. The executable path, not that prose, controls this investigation.
- **Trace-mode toggling is not a clean prefill ablation here.** The model captures/replays prefill unconditionally and accepts `enable_trace` through unused extra kwargs. Changing a plugin flag to decode-only cannot be assumed to make this generator's prefill eager or to measure trace savings. Keep the required trace path for acceptance.
- **No final-source numerical or persistent-buffer contradiction was established.** Repaired sampling/CCL controls and matching hashes are relevant controls, not proof of all async histories or request quality. Large ITL with clean sampled-token validation does not itself implicate those repaired kernels. Missing latency timestamps are not evidence against buffer safety, either.

## Claim review and remaining uncertainty

The first-draft hypotheses were rechecked against executable source and all four completed cases. The final-prefill barrier survived three attempts to falsify it: intermediate returns are nonblocking rather than full-device waits; the installed uniproc future does not have an independent publishing thread; and the final boundary, rather than an arbitrary anonymous token gap, matches all four client maxima. Independent inspection reproduced the widths, boundary intervals and queue ordering. The broad “text filtering doubles gaps” interpretation and lazy-capture explanation were demoted as causes of the reported maxima.

The report proves the source ordering and preserves exact server evidence; it does not invent the discarded client-to-token mapping or claim a measured remedy. The next decisive result is the official client-content maximum with retained correspondence under a candidate that preserves the serving contract. Budget9 remains unobserved here. Budget33 and17 fail; no stage passes on this report.

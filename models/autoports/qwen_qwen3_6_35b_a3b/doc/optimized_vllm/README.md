# Qwen3.6-35B-A3B optimized vLLM — complete with accepted limitations

Primary warmed **128 input / 128 output / 1 user, concurrency 1**: TTFT **2969.75 ms**, decode **31.8201 tokens/s/user**. The same baseline workload measured 3035.46 ms and 18.5884 tokens/s/user. TPOT fell **41.58%**, from 53.79697 to 31.42670 ms.

Runner remediation reproduced **exit 2** with the same sole critical P4 speed-floor row. The [fresh audit](runner_failure_refutation.json) verifies the original owner instruction, unchanged raw results, and the checker's lack of an explicit-owner waiver. See the [work log](work_log.md#runner-failure-reproduction-and-scoped-refutation--2026-11-18) for the evidence-backed refutation of a new acceptance failure. The numerical speed shortfall remains valid; no exit-0 result is claimed.

**The owner accepts both speed-floor misses as known limitations.** The confirmed 128/128/1 decode result remains31.8201 versus46.2293tokens/s/user; the32-user×6000-output run remains2.36 versus4.7649tokens/s/user in its first1000-token bucket (maximum74prompt tokens). [Owner decision](owner_decision.md) and [limitations](limitations.json) preserve the exact scope and measured evidence. The best validated configuration is retained; no further speed work is authorized in stage10. Capacity dispatch is fixed. Independent [review round3](stage_review_round3.md) returns **pass-with-carried**, with no remaining Required Work under the revised owner contract. Stage10 is complete with the recorded limitations.

| Primary metric — 128 input / 128 output / 1 user | Before | Confirmed after |
|---|---:|---:|
| TTFT mean / P99 (ms) |3035.45914 /3035.45914|2969.74762 /2969.74762|
| TPOT mean / P99 (ms) |53.79697 /53.79697|31.42670 /31.42670|
| ITL P50 / P99 (ms) |53.77326 /54.65166|31.43190 /32.28367|
| End-to-end output throughput (tokens/s) |12.97117|18.38711|
| TPOT-derived decode (tokens/s/user) |18.58841|31.82007|

[Confirmation comparison](confirmation_comparison.json) preserves full precision and actual config equality. The [earlier selected measurement](final_comparison.json), same 128/128/1 workload, measured 31.42557 ms TPOT; confirmation differs by 0.00361%. These are warmed single-request measurements, not confidence intervals. Three supplemental matched HTTP requests also preserve exact token arrays.

## Capacity and prefill

Secondary CI capacity: **100 input / 100 output / 32 requests**, canonical burst concurrency. All 32 requests completed in both runs. Aggregate output throughput **23.76859 →24.78697 tokens/s**; mean TPOT 657.50292 →622.09900 ms; TTFT median/P99 58495.69/89763.59 →56657.96/87442.37 ms; ITL P50/P99 452.29/904.35 →419.92/877.60 ms. These are capacity metrics, never headline per-user decode speed.

Serial prefill burst: **128 input / 32 output / 32 requests, concurrency 32**. Median TTFT 52544.48 →51639.49 ms; after/before ratio **0.982777**. Batched prefill was not adopted. This compares two serial-prefill implementations and does not establish batched-equals-solo behavior.

## Measured path and native comparison

Both runs use real TT-plugin vLLM serving through `tt/generator_vllm.py` and the same shared `run_vllm_server` workloads. Actual API-server argv matches: all 40 layers, selected `baseline_bfp8` policy, capacity 32, **context 262144**, budget 2048, mesh 1×4 Blackhole/P300x2, FABRIC_1D, decode-only traces, trace reservation 805306368 bytes, async scheduling, page 32, 8224 served KV pages, all-device sampling and greedy ignore-EOS benchmarks. No profiler was used. Audit and allocation tracking are disabled for both performance runs; the corrected confirmation collector verifies the nonempty inherited API-parent environment instead of treating an empty renamed-worker environment as proof.

The confirmed collector completes 65 canonical benchmark requests plus 3 matched HTTP requests, with 259 device/0 host canonical decode steps and 381 device/0 host matched steps. Raw [before](../../readiness_vllm/optimized_before/performance_execution.json) and [confirmation](../../readiness_vllm/optimized_acceptance_perf/performance_execution.json) records preserve commands, outputs and configuration. The negative identity control rejects the tracked server; the confirmation is the positive control. Cleanup releases all 6 traces and leaves no device holders.

Like work, **128 input / 128 output / 1 active user, capacity 32, context 262144, 127 decode intervals**: native traced serial **31.50643 ms/token**, adapter serial 31.62472, queued async 31.44499, confirmed real HTTP **31.42670**. All 9 native/adapter sequences match. [Native comparison](native_qk_zero_fill_metrics.json) uses the same grouped B2 numerical family and served cache geometry. Historical Stage 7 indexed-expert B1 decode uses a different numerical family and cannot replace this comparison or excuse the failed floor.

## Correctness, sampling and quality

Current selected-source full 40 validation passes 11 full-vocabulary comparisons and three exact 96-token trajectories after nonaligned prompts 129/130/131. Component checks cover route coherence, inactive state, all-chip BF16 transport, poisoned padding, signed-zero/nonfinite transport, cache signatures and native guards. Native build/install and 19 host adapter/sampler tests pass. [Cumulative contracts](cumulative_contracts.md) and the [checklist](optimization_checklist.md) link the evidence.

The final **full configured sampling profile passes 71 tests, with 71 expected skips and zero failures**, across the same 142 test IDs as the original run. Skips: 70 require speculative decoding; one requires speculative artifacts. Six documented short-answer request-mode overrides preserve assertions, caps, seeds and concurrency. This is not unchanged-default execution of those six tests. [Sampling acceptance](../../readiness_vllm/optimized_acceptance_final_sampling/sampling_acceptance_summary.json) records 1474 device/988 host-logit steps in this diagnostic API suite; host routes cover unsupported features/constraints/logprob diagnostics and are not zero-host performance evidence. All 6 trace identities stay fixed across 5157 traffic events, with no recapture or release until clean shutdown.

The original raw-completion suite remains preserved: 70 passes,71 skips,one concurrent recall failure. All 10 truncated TT responses exactly match HF raw controls. Explicit nonthinking chat at the original 24-token cap passes all 14 canonical HF, native and fresh served fixtures. Fresh served answers contain exactly their own phrase and stop in 9–10 tokens; solo prompts 3117/9296 and concurrent prompts 3089/3115/3123/3149. The focused partition audit passes 11 checks, including ragged starts 640/896/2688/2944 and 9 continuations with intervening decode submissions. See [recall repair](recall_repair/AUTOFIX.md). No broad model-quality waiver is inferred.

All 12 default-thinking shared served outputs and 12 native controls were read. All 6 greedy current pairs match exactly; all 12 outputs on each path reproduce their respective Stage 9 controls. Matching HF prompt IDs/settings and degeneracy checks pass. This is a bounded 256-token reasoning smoke, not completed-answer accuracy. The four-mode checkpoint-default check passes before/after explicit thinking-off. Five odd constrained bad-word responses also exactly reproduce prior controls by seed; [that comparison](bad_words_prior_control.json) is separate from qualitative acceptance. See [shared review](served_shared_quality_review.json) and [native comparison](shared_native_comparison.json).

## Trace and async behavior

The tracked mixed lifecycle passes 23 checks over 18 requests, including 129..133 and 4101-token prompts, real 2048→4096→4101 continuation, B2/B4/B32/solo0 and native 32/native 64 sampler reuse. HTTP logit comparison passes 12 requests with zero difference from the current source-matched standalone control. Async growth passes 13 checks/318 submissions; sync passes 12 checks/318 submissions. All six request outputs (129/160,130/112,131/80 in two admission orders) exactly match across scheduling modes. Actual server argv differs only by the async flag; cache geometry/precision/context/capacity match. Maximum pending outputs 2 async versus 1 sync, changed and unchanged pages 5..8, page-only refresh and slot remaps are exercised. Both owners clean up all traces/devices. [Scheduler comparison](async_sync_acceptance.json) and [async contract](async_contract_acceptance.md) give the details.

`read_from_device=False` returns a device-backed immutable `DecodeOutput` lease under the pinned plugin contract. Persistent model/sampler inputs replay with `blocking=False`; deferred reads and event synchronization occur after the async boundary. Every async submission in the growth check has readback delta 0 before return. No host argmax, eager/force-argmax path, full-logit performance readback, aligned-only restriction, context reduction or shared plugin/vLLM change was introduced. Explicit diagnostic logprobs are separate.

The untracked mixed-traffic run passes all 18 exact baseline token arrays, but its 4101-token chunked group has a **51.40s maximum inter-token gap**. This remains a latency limitation, not an ITL-bound pass. The completed three-budget eval sweep below measures the same limitation separately.

## Retained changes and unfinished work

Retained model changes suppress inactive expert routes, skip inactive solo attention/norms, fuse integer route planning, condition recurrent-state writes, fuse hidden-row extraction/masking, select packed postreduce weights, reduce the valid logical prefix and clear native Q/K padding with completed NoC zero writes. Defaults for unrelated callers remain unchanged; `post_combine_reduce(valid_token_rows=0)` preserves the original default. No completion barrier was removed.

Rejected options: 4×8 grouped geometry violates a native guard; ND placement gains 0.1409%; empty-expert mailbox compaction gains 0.6969% and was reverted/rebuilt. The last two kept gains were 3.8784% and 2.6365%; the stop reason is the four-hour optional-search cap. Untried items remain in the checklist.

The required chunk sweep completed once per budget, with **31 background decoders (128 input, 16384 output cap)** and foreground **4096/32768 input, 4 output**, capacity 32/context 262144 and 180s admission warmup. All 31 background prefills completed before long-prompt admission. All requests completed, all decode sampling stayed on device, all six traces were retained, and every server cleaned up.

| Budget | Foreground TTFT — 4096 /32768 input (s) | Worst background ITL (s) |
|---|---:|---:|
|512|95.1399 /772.6622|25.6824|
|2048|94.1334 /736.9219|92.4546|
|4096|94.0028 /742.5912|186.7118|

**No budget qualifies against the 1 s ITL bound.** [Sweep review](../../readiness_vllm/optimized_acceptance_eval_v2/sweep_review.json) preserves measured results and admission/trace/route/cleanup checks. P5/P4 retain the existing 2048 configuration for diagnosis, explicitly not a qualified selection. [Limitations](limitations.json) records this outcome and the owner-accepted single-user speed limitation.

P5 completed all timing requests on this same server: **1024/8192/32768/131072 input, 32 output, 1 user**, decode TPOT **31.50/31.86/32.37/34.75 ms**, respectively. TTFT for those shapes was **22.4295/181.3235/722.5270/2932.6762 s**. No decode cliff was detected. For **127 versus 128 input, 4 output, 1 user**, three measured repetitions after warmup gave median TTFT **3505.2 versus 3044.2 ms**, ratio **1.151**, passing the 1.5 bound. [Timing result](../eval_readiness/long_context_timing.json) has one advisory: no chunk budget meets the ITL bound. The interrupted wrapper's exit status is unavailable; the surviving client wrote this fresh canonical result, printed its summary, and left the server idle before P4 ([recovery record](../../readiness_vllm/optimized_acceptance_eval_final/records/p5_timing.recovery.json)).

P4 completed **32 users ×6000 output tokens**, `lm_eval_reasoning`, thinking on, ignore-EOS, actual prompt lengths at most **74 tokens**. All **192000 output tokens** completed in **2638.6 s**, wall aggregate **72.8 tokens/s**. The matched **32-user ×256-output** plain/eval comparison measured median TPOT **421.53/421.20 ms**, ratio **0.999**, passing the 1.15 bound. Health, slots/KV release, trace retention, zero preemptions and answer-length checks pass.

| Answer bucket — 32 users, prompts ≤74 tokens | Median per-user tokens/s | Probe aggregate estimate (32× unrounded median) |
|---|---:|---:|
|0–1000|2.36|75.67|
|1000–2000|2.34|74.80|
|2000–3000|2.32|74.09|
|3000–4000|2.31|73.82|
|4000–5000|2.29|73.17|
|5000–6000|2.28|73.02|

**P4's functional floor fails:** first-bucket **2.36 tokens/s/user** versus the [necessary batch-32 bound](eval_batch32_necessary_floor.json) **4.764924390924285**. The measured maximum prompt length validates that bound's scope; this is not a measurement of actual expert coverage. The raw result remains failed; the owner explicitly accepts it as a stage10 limitation. The recorded `fix_hours: 0.5786` excludes earlier single-user AutoFix work. This is an owner decision, not a time-box or HF-same waiver.

The installed planning defaults forecast, at **32 users**, **GPQA 198 questions ×12000 typical answer tokens: 8.98 h; MMLU-Pro subset 280×3000: 3.12 h; IFEval 541×1500: 2.99 h**. Total **15.08 h** exceeds **14.4 h**, the 80% allowance for an 18-hour job. The total is computed before row rounding. Each individual job fits its full default question count; the combined plan does not fit that allowance. GPQA extrapolates past the measured 6000-token answers and is optimistic if later decode slows. At the observed 74-token maximum prompt, 32 simultaneous 12000-token answers would require 386368 logical tokens, exceeding the 263168-token pool; the forecast also omits potential preemption at that length. These planning defaults are not measured eval accuracy or a model-specific release plan. Stage 11 must use its actual release question counts and answer lengths; prefer subsets over shortening required reasoning answers.

[Final eval summary](../../readiness_vllm/optimized_acceptance_eval_final/acceptance_summary.json) links the raw results. P4 alone uses **6509 device/0 host decode steps**; the continuous P5/P4 server uses **6657 device/0 host steps**, one unchanged six-trace identity set across **13411 traffic events**, maximum pending outputs **2**, and zero pre-async-boundary readbacks. Shutdown releases all six traces, closes the devices and leaves no owned processes or device holders. Nanobind reports 2 instance/19 type/259 function reference leaks at interpreter shutdown; this warning is preserved, not a claim of a live engine or device leak.

The four retained P4 answer tails were read. Three contain repeated turn markers or post-answer dialogue under forced ignore-EOS generation. This performance probe does not establish answer quality, and its retained tails cannot locate the first EOS; the independent shared qualitative checks above remain the quality evidence. No broader quality waiver is inferred.

All original acceptance measurements and cleanup are complete. The installed stage check returned **exit 2** for P4’s numerical floor failure before owner acceptance. Its waiver logic does not recognize explicit owner decisions; raw failure is preserved separately from the revised stage acceptance contract. Degeneracy and the full context contract pass. The serial-prefill comparison lacks a batched `serial_reference`, an advisory because batched prefill was not adopted. Historical nested context-cap messages are advisory; actual serving remains 262144. See [stage check](stage_check.json). Independent [review round 1](stage_review_round1.md) returns `more-work-needed`: accepted capacities 3 and 5–31 reach an unsupported fused row-mask helper, and both performance floors remain unmet. The capacity-dispatch repair passes [focused validation](../../readiness_vllm/optimized_capacity8_repair/repair_acceptance.json): host dispatch across capacities 1–32, all 23 lifecycle checks on full40 capacity8, and 17 exact greedy comparisons against capacity32. Other capacities use generic extraction/masking; the selected capacity32 operations are unchanged. A separate AutoDebug/AutoFix investigation covers the newly measured P4 failure. No prior owner waiver is broadened.

[Commands](final_validation_commands.md), [work log](work_log.md), [manifest](stage_manifest.json), and [live continuation state](acceptance_resume_state.json) preserve exact ownership and artifact paths. Production checkpoint 08384c1007d1a1065955221284f6d2c197d85260 and documentation checkpoint 8fbe38e7ce079a19153b1baad80f270f3d0c3025 are local WIP commits; further acceptance artifacts await local checkpointing. No push was performed. The shared plugin remains unchanged at 5d1785b76d17a8989c9495f6939abea6cef62f09.

## Review remediation and retained source

RW1 is fixed: capacities3 and5–31 keep generic row extraction/masking, while the selected2/4/32 path is unchanged. Host dispatch checks cover every prepared bucket for capacities1–32; full40 capacity8/context262144 serving passes23 mixed-shape checks and17 exact greedy comparisons, with clean shutdown.

RW2 and RW3 were open in round2; the2026-11-18 owner decision now accepts their exact measured speed limitations. Separate P4 AutoFix tested prefix-balanced combine and batch32 ordered input normalization. Component checks and23 tracked mixed-serving checks pass, but the **32-user /1000-output /maximum74-input** diagnostic still measures **2.56tokens/s/user**, below4.764924390924285. That shortened diagnostic does not replace the **32-user /6000-output** canonical result of2.36tokens/s/user in its first1000-token bucket. The experimental source was reverted and the native baseline rebuilt, installed and revalidated across36 exact cases. The retained implementation keeps the completed original acceptance evidence plus the capacity fix.

[Repair disposition](p4_floor_repair/serving_disposition.json) and [AutoFix](p4_floor_repair/AUTOFIX.md) record the failed repair, commands, clean shutdown and unpromoted candidate patch. Full candidate acceptance was not run; the patch is promising handoff work, not a rejected-slow or rejected-inexact result. The later explicit owner acceptance covers only the two retained-path speed misses. No two-hour waiver or experimental-candidate acceptance is claimed.

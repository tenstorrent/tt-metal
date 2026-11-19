# Optimized vLLM serving — pass-with-carried

**Primary single-user, 128 input / 128 output tokens / 1 request / concurrency1: TTFT 176.926 ms; decode 114.944 t/s/u.** Matched baseline: 248.984 ms and 114.923 t/s/u. TTFT improves 28.94%; decode is effectively unchanged (+0.019%). Guarded collection completed and the [final source/config/route audit](final_serving_audit.json) passes with no errors or advisories. Independent [stage review](stage_review_round1.md) returns **pass-with-carried**, with no required-work findings, under the user’s cap instruction. Local checkpoint details are recorded below.

The measured path is real `run_vllm_server` HTTP serving through `tt/generator_vllm.py`. Selected configuration: all32 layers, TP4 / P300x2 / mesh1×4, `bfp8_all_lofi`, `max_model_len=262144` from [context_contract.json](../context_contract.json), decode-only traces, integrated64 device sampling, `sample_on_device_mode=all`, asynchronous scheduling, responsive interleave of one prefill to two decodes, and **`max_num_batched_tokens=4096`**. Primary uses `max_num_seqs=1`; capacity/load use32. Nonaligned prompts remain supported. Plugin revision `2a1504554aa7babb1d426dfbca8c7db8c0a795ef` is unchanged; this stage makes no plugin or vLLM source changes.

## Primary single-user results

Both rows use identical full server commands and sampling configuration, including budget4096 and responsive interleave1:2. Workload:128 input /128 output tokens /1 request /concurrency1, B1/context262144, greedy explicit generation, `ignore_eos`, warmed server, allocation tracking and profiling off. Decode t/s/u is `1000 / mean TPOT`; output throughput includes prompt and serving time. Values below are source/config audited; one primary request per run does not establish a confidence interval.

| Source / workload | TTFT mean ms | TPOT mean ms | ITL p50/p99 ms | Output tok/s | Decode t/s/u |
|---|---:|---:|---:|---:|---:|
| `stage10_final_before`,128/128/1/c1 | 248.984129 | 8.701514 | 8.703125/8.855249 | 94.499707 | 114.922523 |
| `stage10_guarded_after`,128/128/1/c1 | 176.926293 | 8.699881 | 8.697813/8.933093 | 99.829186 | 114.944095 |

The [guarded full-model comparison](guarded_full_model_comparison.json) passes: all 32 layers, TP4, selected precision, integrated64 greedy split sampling, 128 input / 128 output tokens / B1 / context262144, three warmed nonblocking traced repetitions. Streamed decode (127 steps, per-token readback) has median **8.743032 ms/token, 114.376803 t/s/u**; resident decode (126 steps, one final synchronization) has median **8.703898 ms/token, 114.891057 t/s/u**. Resident runs perform no token/position/page uploads, host reads, captures or releases inside the timed window; final tokens match streamed runs. These are standalone controls. Served mean TPOT / streamed full-model median is **0.995065×** (8.699881 / 8.743032 ms/token), consistent with like-work decode parity. The primary single-request and standalone three-repetition protocols differ; this is not evidence that serving is intrinsically faster.

## Secondary capacity results

CI workload:100 input /100 output tokens /32 requests /concurrency unset, B32/context262144/budget4096/responsive1:2, greedy explicit generation and `ignore_eos`. These are capacity results; admission delay makes burst TPOT unsuitable for headline decode t/s/u.

| Source / workload | TTFT mean ms | TPOT mean ms | ITL p50/p99 ms | Output tok/s |
|---|---:|---:|---:|---:|
| `stage10_final_before`,100/100/32/c unset | 2518.887479 | 13.000634 | 11.166322/73.715103 | 839.794117 |
| `stage10_guarded_after`,100/100/32/c unset | 2001.170764 | 13.072623 | 11.148533/85.380783 | 969.500991 |

Serial-prefill burst,128 input /32 output tokens /32 requests /concurrency32, B32/context262144/budget4096/responsive1:2: **mean TTFT 2512.055104→1967.417498 ms (after/before 0.783190×); p99 2587.219642→2025.786681 ms (0.782998×)**, using the matching original-source reference. CI output throughput above improves15.45%; its p99 ITL rises73.715→85.381ms for100/100/32/concurrency unset, so no universal latency improvement is claimed. A new dense batched-prefill optimization was not adopted. Existing multirow generator correctness remains covered by [batch32_full.json](../full_model/batch32_full.json).

## Completed guarded budget selection

All three guarded sweep rows use the same frozen runtime/native policy and server settings apart from token budget: B32/context262144/responsive1prefill:2decode,31 background decoders plus4096- and32768-token arrivals. Each has `errors=[]` and source-bound result/launch/environment sidecars. [guarded_budget_selection.json](guarded_budget_selection.json) selects **4096, the largest tested budget with worst ITL at most1000ms**. The matched original baseline already uses4096/responsive1:2. These are mixed-load latencies, not primary decode throughput or a universal latency SLA.

| Budget / workload | Worst ITL ms | TTFT at4096 input ms | TTFT at32768 input ms | Decision |
|---|---:|---:|---:|---|
| [512](../eval_readiness/long_context_timing/chunk_512.json), B32/31 decoders +4K/32K arrivals | 994.8 | 2864.9 | 51156.2 | Pass |
| [2048](../eval_readiness/long_context_timing/chunk_2048.json), B32/31 decoders +4K/32K arrivals | 972.8 | 870.7 | 12995.8 | Pass |
| [4096](../eval_readiness/long_context_timing/chunk_4096.json), B32/31 decoders +4K/32K arrivals | 947.6 | 541.4 | 6633.0 | Selected |

The [pre-guard sweep](final_budget_selection.json) remains historical. All final guarded collection is complete, including tracked/untracked mixed traffic, full-model controls, HTTP sampling/quality and P4/P5; final artifact audit passes.

## Kept changes and correctness evidence

- **Defer reads until the async boundary.** Decode returns device-backed handles; `read_decode_output` queues one token transfer from the first shard. Explicit host-logit compatibility retains required shard reads. Guards reject replay before unqueued borrowed output can be overwritten, before state writes; cleanup drains pending work. Duplicate sampler-history reset was removed.
- **Refresh page inputs only when values change.** Cache the first unmapped page bound and compare row values on explicit reload. Preserve rejection before writes for sparse/changed pages, remaps and stale host objects; unchanged rows avoid uploads. A failed page transfer or forward-input remap permanently invalidates that decode state, preventing a retry from trusting an obsolete host snapshot; normal cleanup remains available. Token/position progression and changed/unchanged table tests remain required.
- **Reduce discarded prefill work.** For canonical16384/32768 horizons consuming at most8192 rows, select the prepared8192 execution profile. Skip the terminal head only for authoritative discarded intermediate/rebuilt-prefix outputs after state/history bookkeeping; retain preparation and final outputs. [Full-stack control](compact_prefill_full_model.json), [production selector](compact_production_selector.json) and [horizon control](candidate3_device/vllm_prefill_horizon_probe.json) cover exact state/logits, continuation, sampler progress and trace retention, including nonaligned prompts.
- **Reuse unchanged prefill/attention inputs.** Cache masks/history indices/row indices and attention positions/page inputs. [Prefill control](prefill_refresh_full_model.json), [attention control](attention_inputs_full_model.json) and [4096-budget control](attention_inputs_4096_full_model.json) pass exact full-stack comparisons, including4096/4105-token fragments, completed32768-token prompts and next decode at context262144.

The guarded source passed160 CPU tests and affected-file hooks. All five [guarded device probes](guarded_device/provenance.json) completed successfully: adapter deferred-read/stale-input/remap controls, original-prompt sampling boundary, prefill horizon/discarded-head behavior, external cache binding, and sparse resident page admission. These reduced-stack controls use their recorded diagnostic contexts; they do not replace the full-context32-layer comparisons or serving gates. The guarded three-budget sweep is complete as recorded above. The [guarded compact selector](guarded_compact_production_selector.json) passes exact equality for layers0/3, B1/TP4, original horizon32768 and diagnostic context65536; serving/eval context remains262144. All 12 guarded full-model qualitative outputs were read through first EOS (six greedy and six sampled, seed0/cap256). The [review](guarded_fullmodel_qualitative_review.json) finds coherent answers or prefixes with existing token-cap and thermodynamics limitations; [raw output identity](guarded_fullmodel_output_identity.json) matches both prior TT control sets exactly. HF control metadata matches. This small visible-label control is not a comparative degradation-rate study. Final HTTP sampling has **67 passed,75 skipped**; skip scope is retained in the sampling log. The [HTTP qualitative review](guarded_http_qualitative_review.json) passes with limitations. The owner-reviewed matched blind story control finds0/10 HF and0/10 HTTP failures under the predefined rule;18/20 answers are cap-limited and incomplete, so this is neither complete-story evidence nor proof of equivalent population quality. Final mixed and served/full-model parity checks pass. Earlier mixed checks and [tracker positive control](trace_tracker_positive_control.json) support, but do not replace, that repetition.

Persistent trace inputs and greedy split device sampling remain model-owned. Replay uses `ttnn.execute_trace(..., blocking=False)`. Final audit confirms device sampling, `read_from_device=False`, `enable_trace=True`, deferred `async_read=True`, six initial positive captures and no captures/releases inside primary, capacity, P4 or P5 windows. Discard-all intermediate prefill may log host mode without producing logits; deliberate mixed-test host compatibility must remain outside measured/load windows. No serving profiler is used. Audited default windows contain127/127 decode/async-read steps for primary,261/261 for capacity,148/148 for P5 and6511/6511 for P4. P5 has39 host-labelled intermediate prefills that discard outputs. Explicit compatibility/logprobs/mode-switch tests account for574 B32 host decodes outside measured/eval windows; B1 has zero outside-window host decodes. This is a default-path no-host-fallback claim, not a claim that host compatibility is absent.

## Geometry decisions and rejected options

Selected BFP8/LoFi geometry uses actual weights/activations; earlier HiFi2 is not substituted. [Decoder decision](selected_decoder_geometry_decision.json) covers ten roles across real layers0/3: packed input, MLP gate/up/down and attention output, two natural-text stimuli, B1/physical32,128-token input/context262144, one through four readers and legal K divisors through16,40 nonblocking warmed replays ×3 repetitions. Of234 rows,232 pass; two fail recognized pre-capture K3 divisibility checks. Production baseline/repeat wins every role; no geometry change is adopted. [Raw retry](selected_decoder_all_geometry_retry.json) preserves measurements.

[Head search](selected_head_geometry.json), terminal norm plus unsplit LM head, B1/physical32 /128-token natural-text residuals /context262144 /selected BFP8/LoFi /40replays×3: production r3/K2 takes0.570413ms (repeat0.570273ms), versus best legal alternative r4/K2 at0.573664ms. One-reader controls exceed L1. This diagnostic is not complete split-sampler or serving throughput.

[Compact6144](compact_prefill_6144_reduced.json) is rejected: both completed32768-token schedules on real layers0/3,B1/context65536 fail exact state/logit equality. Production keeps8192. Earlier failed sweeps, interrupted geometry attempts and orchestration failures remain in the [work log](work_log.md) and [AutoFix record](AUTOFIX.md), excluded from final acceptance. [Checklist](checklist_draft.md) and [Rule3 disposition](rule1_disposition_draft.md) record retained topology and untried extensions. **Stop Rule3 fired at2026-11-14 05:07:49UTC**, recorded in [optimization_stop.json](optimization_stop.json), under the user’s explicit cap instruction. No new optimization candidates were run afterward; remaining required gates were completed once. Rule1 is not claimed and Rule2 is unsupported. Untried handoffs have unknown gains. Independent review accepts these handoffs; local checkpoint details are recorded below.

## Mixed traffic and eval readiness

Corrected-client mixed workload:15 requests ×70 outputs, prompt lengths129/130/131/8193/32769, concurrency3, B32/context262144/budget4096. Both tracked and untracked runs pass all10 checks: exact greedy outputs across order/device→host→device modes, six existing traces, no recapture or retirement, changed continuation offsets and decode during chunked prefill. Corrected client snapshots match their manifest. Untracked TTFT p50/p99 is436.604/9904.987ms, ITL p50/p99 21.046/756.362ms, maximum pause1225.932ms; tracked TTFT p50/p99 501.700/8764.485ms, ITL p50/p99 21.045/677.212ms, maximum pause818.198ms. The untracked stress pause exceeds1s; the separate31-decoder budget-sweep bound is not a universal latency SLA. Old bytewise-client long-prompt latency is invalid and retained only as historical correctness evidence.

[P5 timing](../eval_readiness/long_context_timing.json) passes at context262144 with32 output tokens/request: inputs1024/8192/32768/131072 yield TTFT175.7/1218.4/5871.0/43624.2ms and TPOT17.14/18.11/21.16/33.41ms. Nonaligned127-token versus128-token TTFT (four output tokens, three measured repetitions after warmup) is208.3/208.8ms; all timing errors are empty. These P5 request lengths and client timings differ from the primary128-output workload.

[P4 long-answer load](../eval_readiness/long_answer_load.json) passes all11 checks:32 users ×6000 output tokens, `lm_eval_reasoning`, `ignore_eos`, context262144;192000 outputs in77.9s, wall aggregate2464.7tok/s. Per-user throughput is88.73tok/s in answer positions0–1000 and75.43tok/s in5000–6000. The256-output eval/plain sampler comparison at32users is11.18/13.34ms TPOT (0.838×), with no errors. No preemptions or new captures; slots and KV return to idle, health200.

The P4 planning forecast is0.43h total at concurrency32 under an18h job cap (80% fit budget): GPQA198questions ×12000 typical answer tokens,0.26h, extrapolated beyond the measured6000; MMLU-Pro subset280×3000,0.09h; IFEval541×1500,0.08h. These are default planning assumptions, not executed evaluations or measured release accuracy; replace them with the release plan before scheduling. [limitations.json](../eval_readiness/limitations.json) records existing quality/cap limitations; forecast assumptions are explicit in the P4 artifact.

| Final requirement | Result |
|---|---|
| Matched primary/CI and serial burst, workloads above | PASS |
| Full-model/serving decode parity,128/128/B1/context262144 | PASS; ratio0.995065× |
| Sampling and explicit standalone/HTTP qualitative review | PASS with documented skips/limitations |
| Tracked/untracked mixed reuse, nonaligned/chunked paths | PASS; stress latency scope above |
| P4/P5 and three-budget sweep | PASS; selected4096 |
|23-file runtime/config/route/native/client audit and promotion | PASS, no audit errors/advisories |
| Packaged stage gates | PASS with advisories; exit1, zero critical failures/errors; explicit degenerate-output gate PASS |
| Independent stage-review and findings disposition | PASS-WITH-CARRIED; no required-work findings |
| Final owned-process cleanup and local commit SHAs; no push | Cleanup PASS; checkpoint record below |

## Commands and provenance

The serialized [guarded validation](guarded_validation_frozen.sh) and [acceptance](guarded_run_acceptance.sh) completed exit0. Original `stage10_final_before` and final `stage10_guarded_after` use the matched4096/responsive1:2 configuration; `stage10_guarded_tracked` repeats trace reuse with allocation tracking. Exact server commands, environments, workload windows and loaded-source/native provenance are in their `*_measurement_execution/` directories; [before metrics](stage10_final_before_measurement_execution/metrics.json) and [after metrics](stage10_guarded_after_measurement_execution/metrics.json) retain full precision. Probe/eval commands are recorded in raw artifacts and driver files; sweep sidecars bind each budget to its runtime and result. No serving profiler was used.

Earlier `stage10_final_after` results are pre-guard intermediate evidence, not the final headline. Earlier bytewise-client mixed latency is withdrawn; token/trace checks remain historical correctness evidence. Both guarded mixed directories snapshot the corrected client as `mixed_probe.py.source`. [Client anomaly](mixed_stream_client_anomaly.json) and its loopback transport tests explain the correction. The final audit requires that snapshot to match the corrected-client manifest and current probe, and checks all23 runtime files plus every copied runtime snapshot against the final source manifest. The historical [pre-guard schema audit](pre_guard_schema_audit.json) is explicitly non-final and fails the missing-client-snapshot gates; its successful schema traversal is not acceptance.

Completed audit command:

```bash
model=models/autoports/ornith_ai_ornith_1_5_9b
python "$model/doc/optimized_vllm/audit_final_serving.py" \
  --before-label stage10_final_before --after-label stage10_guarded_after \
  --tracked-label stage10_guarded_tracked --budget 4096 --sweep-budgets 512 2048 4096
```

The audit passes and canonical promotion is complete. [Gate disposition](final_gate_disposition.json) records exit1 with zero critical failures/errors: historical512-token fixture mentions and optional stronger dense-batching targets are advisory; final context remains262144 and no dense batched-prefill path was adopted. [Cleanup](final_cleanup.json) confirms no vLLM/EngineCore processes or device owners, zero resets, unchanged plugin and no push. [Gate/packaging plan](final_gate_packaging_plan.md) records required packaging/gate commands without deleting originals or bypassing hooks. The independent review and carried items are recorded in the work log. Checkpoint details follow below; no push is authorized.

The canonical mixed report is compacted only for the repository file-size limit; [compaction receipt](canonical_mixed_compaction.json) links the unchanged full report and lossless bundle. Checks, latencies, outputs and trace signatures are preserved; omitted prompt/SSE payloads and event details remain recoverable.

## Local checkpoint

Accepted independent review: [stage_review_round1.md](stage_review_round1.md). Stage-owned runtime, tests and evidence are committed on `tt-metal/main` at **`3d82102e1b493143bbc29f50ec320938a0048bad`**. This documentation-only bookkeeping commit records that checkpoint; all normal commit hooks passed. The serving plugin and vLLM are unchanged. No push.

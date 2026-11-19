# Ornith vLLM integration

**Stage 9 accepted: independent review clean-pass.** Final64 correctness, quality
and measurements are complete; the installed gate reports advisories only.
The owner approved the tested top_k20 default on 2026-11-14. The primary B1 result is **236.249337 ms TTFT** and
**114.941493 decode tokens/s/user** (`1000 / mean TPOT`), for 128 input/128 output
tokens, one request, concurrency1, TP4 and context262144. Full sampling passes 67 tests with 75 capability-fixture
skips; tracked mixed/lifecycle, chunked-prefill, P3 critical checks, all nine P5
cases, qualitative controls and the matched logit oracle pass within the scopes
below. Final full-profile P2 passes applicable checks, with the documented
missing-generation-config skip. The [follow-up review](stage_review_round2.md) closes the sole required finding.

All final measurements use full32 layers, TP4, context262144, block32, device
greedy sampling (`temperature=0`), explicit generation settings, random prompts
and `ignore_eos`. Allocation tracking and profiling are off. Each workload
completed its requested output tokens. Values below are rounded to six decimal
places; [metrics.json](final64_measurement_execution/metrics.json) retains full
precision, exact workload settings, actual launch summaries and raw result paths.

## Primary single-user measurement: final64 B1

The B1 server uses 128 input/128 output tokens, one request and concurrency1.
TTFT and TPOT percentiles describe that one completed request, not a repeated
request distribution; ITL percentiles describe its token intervals.

| Latency (ms) | Mean | Median (P50) | P99 |
| --- | ---: | ---: | ---: |
| TTFT | 236.249337 | 236.249337 | 236.249337 |
| TPOT | 8.700078 | 8.700078 | 8.700078 |
| ITL | 8.700078 | 8.705536 | 8.926978 |

Aggregate output throughput: **95.410489 tokens/s**;
TPOT-derived decode rate: **114.941493 tokens/s/user**.
Request throughput: 0.745394 requests/s; total input+output
throughput: 190.820977 tokens/s.

See the [primary benchmark](../../readiness_vllm/final64_b1_untracked/vllm_benchmark.json).

## Secondary CI serving burst: final64 B32

The separate B32 server uses 100 input/100 output tokens with 32 requests and
concurrency unset. All 32 requests completed. This measures burst admission and
serving, separately from the primary single-user rate.

| Latency (ms) | Mean | Median (P50) | P99 |
| --- | ---: | ---: | ---: |
| TTFT | 2289.410984 | 2733.208891 | 3253.989109 |
| TPOT | 27.122795 | 24.475838 | 35.305579 |
| ITL | 27.122795 | 11.168118 | 619.431904 |

Aggregate output throughput: **519.415016 tokens/s**;
TPOT-derived decode rate: **36.869356 tokens/s/user**.
Request throughput: 5.194150 requests/s; total input+output
throughput: 1038.830031 tokens/s.

See the [CI benchmark](../../readiness_vllm/final64_b32_untracked/vllm_ci_serving_benchmark.json).

## Serial-prefill burst: final64 B32

The B32 server uses 128 input/32 output tokens, 32 requests and concurrency32.
Prefill remains serial; no batched-versus-serial ratio is claimed.

| Latency (ms) | Mean | Median (P50) | P99 |
| --- | ---: | ---: | ---: |
| TTFT | 2232.048485 | 2315.043664 | 3360.730757 |
| TPOT | 69.003276 | 66.111424 | 92.647742 |
| ITL | 69.003276 | 11.146836 | 824.356172 |

Aggregate output throughput: **197.497607 tokens/s**;
TPOT-derived decode rate: **14.492066 tokens/s/user**.
Request throughput: 6.171800 requests/s; total input+output
throughput: 987.488034 tokens/s.

See the [prefill benchmark](../../readiness_vllm/final64_b32_untracked/vllm_prefill_burst_benchmark.json).
The earlier original32 primary baseline (B1, 128 input/128 output, one request,
concurrency1, TP4/context262144) was 115.278753 decode tokens/s/user, with
232.763165 ms TTFT and 8.674625 ms mean TPOT. Its separate B32 CI workload
(100 input/100 output, 32 requests, concurrency unset, TP4/context262144)
produced 528.018212 output tokens/s. Historical measurements and their original
source scopes remain in [before_sampling64 metrics](before_sampling64_measurement_execution/metrics.json);
they are not substituted for the final64 results above.

## Shared serving path and precision

[`OrnithForCausalLM`](../../tt/generator_vllm.py) translates the pinned plugin
interface into the canonical `OrnithGenerator` prefill, traced decode, split
sampling and deferred-readout methods. It introduces no serving-only sampler,
host argmax or Python token-feedback loop. Device sampling reads tokens only.
The corrected prefill path also discards intermediate logits on device; the
earlier tracked run redundantly read those unused logits. The reduced
[horizon device check](../../readiness_vllm/prefill_horizon_device.json) passes
profile-change and missing-tail transitions, zero discarded-row logit readback,
and exact sampler-state nonadvancement. The earlier corrected full-model mixed, chunked,
lifecycle and P5 checks pass with the original32 sampler. Integrated64
mixed/lifecycle, chunked and all nine P5 checks also pass.
vLLM allocates the attention pool through the adapter callback; every execution
checks the exact cache identity. Recurrent state remains model/slot-owned.
The served cache has8224 physical32-token blocks at B32 and context262144.

The datatype-sweep `bfp8_all_lofi` selection is loaded by the default model
constructor: attention/MLP/head weights BFP8 and projection fidelities LoFi;
decode projection inputs/residuals BF16; large prefill norm activation, CCL and
KV BFP8; recurrent state FP32; norm HiFi4, terminal norm and decode SDPA HiFi2;
no layer exceptions. See the [selected policy](../datatype_sweep/configs/bfp8_all_lofi.json)
and server registration event. The context remains the full262144 recorded in
[context_contract.json](../context_contract.json); maximum-context acceptance passed with262143 input+1 output tokens on the
earlier tracked server ([artifact](../../readiness_vllm/context_acceptance.json)).

`supports_async_decode=True` is exercised with `--async-scheduling` and
`trace_mode=decode_only`. Prepared decode buckets are `[max_num_seqs]`; final
B32 capacity and B1 primary measurements are recorded separately. Intermediate
prefill chunks do not sample. Explicit logprobs, unsupported processors and
stochastic top-k above64 use the optional host compatibility path in the
integrated64 implementation. Its tracked reasoning probe exercises stochastic
top-k40 on device. The reported original32 performance baseline used a real
32-candidate boundary and routed that request to host. Final tracking-off
integrated64 performance is recorded above. The operator default
`--override-generation-config '{"top_k":20}'` keeps ordinary sampled requests
on device. The pinned checkpoint has no generation_config.json. Chat defaults
use `enable_thinking=false`; explicit thinking-on requests are covered by P3.

Prefix caching has the prior, documented `recurrent_state` exclusion: attention
KV blocks cannot reconstruct the24 recurrent/conv layers. Its explicit
[artifact](../../readiness_vllm/prefix_cache_check.json) records this exclusion.
Batched prefill is not enabled. The final64 serial-prefill burst is reported
above; the [original32 burst](../../readiness_vllm/before_sampling64_b32_untracked/vllm_prefill_burst_benchmark.json)
remains historical evidence. No batched-versus-serial ratio is claimed.

## Stack and local plugin repairs

[Stack provenance](stack_provenance.json) records standalone
[tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin) pin
`042a94863ad27d88207bcf71abdf67122c6f4c27`, whose `docs/install-vllm-tt.sh`
recommends upstream [vLLM](https://github.com/vllm-project/vllm) v0.31.0.
The installed CPU/empty-target build is0.31.0+empty. Runtime registration uses
`TT_MODEL_CLASS_OVERRIDES=TTQwen3_5ForConditionalGeneration=models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm:OrnithForCausalLM`;
the worker registration event resolves to this class. No upstream vLLM source
was changed.

All plugin changes have [individual change records](plugin_changes.md):

| Repair | Local commit | Host validation |
| --- | --- | --- |
| Preserve ownership across prefill continuation | `a09b0ef5f600305c738da0482bb76fee0e2e776a` |2322passed,1skipped |
| Owner-approved original prompt boundary and shared host history | `215effdee4e473f8c50af608d9812fafc56fd48b` |2348passed,1skipped |
| Replace retained page rows on immediate resume | `2a1504554aa7babb1d426dfbca8c7db8c0a795ef` |2354passed,1skipped |

All three commits are local and passed pre-commit; nothing was pushed. The first
and third repair existing contracts. The second implements the exact Entry1
interface approved by the owner. tt-metal stage changes remain uncommitted until
acceptance and review finish.

## Serving correctness evidence

Integrated64 sampling and qualitative collection completed on the first
EngineCore, PID **1265789**: the full canonical plugin profile passed **67 tests
and skipped 75** under its normal capability fixtures. The original
[sampling log](../../readiness_vllm/sampling64_tracked/first_attempt/sampling_tests.log),
[runner log](../../readiness_vllm/sampling64_tracked/first_attempt/full_sampling_runner.log),
[actual launch summary](../../readiness_vllm/sampling64_tracked/first_attempt/vllm_run_summary.json)
and server log are preserved under `first_attempt`. That attempt's
[mixed check](../../readiness_vllm/sampling64_tracked/first_attempt/mixed_trace.json)
failed only concurrent-decode coverage; output identity, capture counts and
trace lifetime checks passed. It shut down cleanly. The retry changes workload
admission timing so a short request starts streaming before long prompts arrive.

The explicit [completed sampling reuse manifest](../../readiness_vllm/sampling64_tracked/completed_sampling_reuse.json)
retains these sampling/qualitative results with their original EngineCore origin.
Their retained files are byte-identical to the preserved originals; production
model sources, plugin identity and native provenance are retained across the
restart. Sampling/qualitative and the remaining probes are separate attempts;
there is no claim of a single combined runner pass.

The retry full32-layer B32 server, EngineCore PID **1293243**, passes all **10
checks** in both [mixed trace](../../readiness_vllm/sampling64_tracked/mixed_trace.json)
and [lifecycle trace](../../readiness_vllm/sampling64_tracked/lifecycle_trace.json).
Each workload has 15 requests, 70 outputs per request and concurrency 3, with
device, explicit-host-logprobs and device phases. Mixed prompt lengths are
129/130/131/1033/2051; lifecycle lengths add 8. Actual prefill starts include
**0, 896, 1024 and 2048**, interleaved with concurrent decode. All six prepared
trace signatures retain one capture each, with no new capture or retirement.
Greedy token sequences agree across ordering and sampling modes. Allocation
tracking and program-cache allocation checks remain enabled; these are
correctness workloads, not headline performance measurements.

The [page-growth analysis](../../readiness_vllm/sampling64_tracked/page_growth_analysis.json)
records **24 device page-table-only refreshes, 246 steady calls with unchanged
inputs/table, and 69 host-control calls** in each workload. Distinct concurrent
short requests cross logical page entries at positions 160 and 192, and their
outputs match the explicit-logprobs control, whose plugin path excludes decode
overlap. The trace does not expose a complete per-request physical-page ledger.
Targeted [chunked-prefill tests](../../readiness_vllm/sampling64_tracked/chunked_prefill_tests.log)
pass **3 tests in 29.60 seconds** with the 1024-token scheduler budget.

Earlier adapter batteries separately cover stale inputs, positions, page-only
updates, unchanged tables, inactive slots and pending submissions: see
[adapter_decode_probe.json](adapter_decode_probe.json),
[adapter_decode_after_seed.json](adapter_decode_after_seed.json),
[adapter_seed_decode_probe.json](adapter_seed_decode_probe.json) and
[original_boundary_device.json](original_boundary_device.json). Historical
original32 [corrected route evidence](../../readiness_vllm/corrected_tracked/runtime_audit.json)
is retained. The integrated64 [benchmark/prefill route audit](../../readiness_vllm/sampling64_tracked/runtime_audit.json)
passes with 261 device decode calls, zero host calls, 238 steady calls, no new
captures and no OOV errors. Its tracked timings are route evidence only.
The earlier collection import-order failure and block64 summary default remain
historical artifacts; current attached collection explicitly uses block32 and
`PYTEST_ADDOPTS='-p vllm'`.

Reduced diagnostic servers and forced-preemption comparisons remain labeled
as diagnostics, not full-model quality or performance. The page-row defect is
fixed. A canonical recomputation control reproduces the sampled divergence with
identical seeds/history; the reduced greedy follow-up also reproduces both served branches exactly
through50 outputs, with equal sampler state. The [aligned151 state comparison](replay_model_state151_interpretation.md)
passes structural checks: all48 host/device replay tensors and sampled inactive
regions are exact. KV rows0–128 match exactly; differences begin at decode
row129. This demonstrates a canonical prefill/decode computation-route difference
with no concrete remaining adapter state defect identified in this stimulus.
It covers layers0/3, B3 and a fixed larger pool, not the full model or live
scheduler replacement. It does not establish harmless rounding or a tolerance.
**All three original strict forced-preemption token-parity checks remain failed.**

## Evaluation and quality

Integrated64 retry P3 passes every behavioral case, including all 10 critical
checks. [Current chat_conformance.json](../../readiness_vllm/sampling64_tracked/eval_model/doc/eval_readiness/chat_conformance.json) retains one
rendering advisory: deprecated `reasoning_content` tokenizes differently through
upstream `/tokenize`. The actual chat endpoint normalizes that alias, preserves
the88-token history and returns Ada. Current API `reasoning` plus the existing
HF template alias reproduces the expected tokens. Exact requests, token IDs and
schema proof are in [reasoning_field_tokenization.json](../../readiness_vllm/before_final64_promotion/doc/eval_readiness/chat_conformance/reasoning_field_tokenization.json).
No request fixture, tokenizer template or installed probe was weakened.

Integrated64 [P5 long-context accuracy](../../readiness_vllm/sampling64_tracked/eval_model/doc/eval_readiness/long_context_accuracy.json)
passes all nine needle cases at approximately 8K/32K/128K tokens and depths
0.1/0.5/0.9 with device greedy sampling. The
[P5 text review](../../readiness_vllm/sampling64_tracked/p5_text_review.json) reads
every raw answer: each exactly equals its distinct expected passphrase, stops
normally, and shows no repetition, gibberish, language drift or contamination.
This does not replace the separately scoped earlier maximum-context acceptance.

The final64 full32-layer/TP4/B32/context262144
[logit verdict](../../readiness_vllm/sampling64_tracked/logit_verdict.json) passes
with 8224 external blocks. All 12 HTTP repeat/reorder comparison groups are
exact; all nine standalone-versus-HTTP groups match at three positions each
(**27 positions**). Selected tokens are argmax, top20 ordering and candidate
logprobs match with maximum error zero, and repeated raw logits are exact.
The matched [HTTP artifact](../../readiness_vllm/sampling64_tracked/logit_http.json)
and [standalone oracle](../../readiness_vllm/sampling64_tracked/logit_tt.json) are
retained. This optional full-logits/logprobs diagnostic is separate from the
measured token-out performance path.

Original32 [P2 harness results](../../readiness_vllm/before_sampling64_b32_untracked/harness_requests.json)
accept the reasoning profile but fail its TPOT ratio:2.410 at1 user and7.268
at32 users (critical threshold1.15), with top-k40 routed to host at the real
device limit32. Non-reasoning and agentic ratios pass. The failure is retained;
the integrated64 tracked [reasoning-only P2 probe](../../readiness_vllm/sampling64_tracked/eval_model/doc/eval_readiness/harness_requests.json)
now passes with ratio 1.00 at both 1 and 32 users. Its
[route audit](../../readiness_vllm/sampling64_tracked/p2_route_audit.json) observes
644 device decode calls, zero host decode calls, no new captures and no OOV
errors. This validates the top-k40 route under tracking. The final
[tracking-off full-profile P2](../eval_readiness/harness_requests.json) also
passes applicable acceptance and ratio checks: reasoning 1.000/0.998,
non-reasoning 1.000/1.009 and agentic 1.000/0.996 at 1/32 users, respectively
(threshold 1.15). The model-card profile retains its explicit skip because the
checkpoint has no readable generation_config.json. See the
[evaluation plan](eval_plan.md) and [measurement plan](final_measurement_plan.md).

All six greedy and six sampled [integrated64 served outputs](../../readiness_vllm/sampling64_tracked/first_attempt/vllm_qualitative_outputs.json)
and all 12 fresh same-code, same-chip TT controls were read in full. The
[qualitative verdict](../../readiness_vllm/sampling64_tracked/qualitative_observations.json)
is **pass**: coherent, on-topic output without mechanical repetition,
gibberish, unrequested language drift or request contamination. Some longer
responses reach the 256-token cap; matching HF controls also show capped tails.
The 12 previously read HF outputs were reused only after exact shared
prompt/model/revision/tokenizer/template/settings metadata equality was verified
for both arms. [Control collection provenance](../../readiness_vllm/final64_qualitative_controls.json)
records the fresh TT collection and HF reuse. TT controls use TP4 B1/context2048
versus served B32/context262144. The verdict establishes qualitative equivalence;
unseeded served samples and seed0 controls do not establish exact RNG or
cross-geometry token identity.

## Performance and process status

Final64 primary B1, secondary B32 and serial-prefill measurements are recorded
above, with allocation tracking off and no concurrent HF CPU jobs. The
[measurement execution](final64_measurement_execution/commands.sh) records exact
commands; [flow.exit](final64_measurement_execution/flow.exit) is 0. CI burst TPOT
remains separate from the headline single-user decode rate.

[Final evidence promotion](../../readiness_vllm/final_evidence_promotion.json)
completed exact artifact copies after validation, preserving the previous
canonical evidence under `readiness_vllm/before_final64_promotion`. It records
sampling/qualitative from the first tracked EngineCore separately from retry
probes, and keeps distinct tracked, B32 measurement and B1 measurement launch
summaries. P2, the earlier-source maximum-context acceptance and prefix-cache
exclusion remain preserved. Promotion does not itself assess the stage verdict.

The prior selected-policy resident token-out baseline is116.141tokens/s/user and
88.885ms warmed TTFT (B1,128 input/128 output, context2048, serial TP4, device
greedy). Its teacher-forcing measurement is114.540tokens/s/user (B1,161 input/100
output, AIME24). These are generator measurements at different context/workload;
teacher forcing supplies only a latency lower bound, not a serving parity gate.
See [datatype-sweep evidence](../datatype_sweep/README.md).

The corrected tracked and original32 measurement servers completed idle shutdown;
[measurement shutdown evidence](before_sampling64_measurement_execution/shutdown.log)
and [device-owner checks](before_sampling64_measurement_execution/device_owners.log)
are retained. The integrated64 first attempt and retry both closed cleanly.
The retry [execution flow](../../readiness_vllm/sampling64_tracked/execution/flow.exit)
exited 0 with [normal shutdown](../../readiness_vllm/sampling64_tracked/execution/shutdown.log);
its [acceptance index](../../readiness_vllm/sampling64_tracked/acceptance_index.json)
records the separate sampling origin. Final B32 and B1 measurement servers
also completed [normal shutdown](final64_measurement_execution/shutdown.log).
The [installed final gate](final_stage_gate.log) completed with
[exit 1](final_stage_gate.exit), an **advisory** result with no critical failures
or errors: P2 has 13 passing/skipped checks and P5 has four passing summary
checks. Retained advisories are the known P3 `render_c07` alias difference,
historical 512-token host diagnostic logs, and absent `serial_reference` for
the prefill burst. The latter is a serial-prefill measurement with batched
prefill disabled; no comparison ratio is claimed.
[Final cleanup](final_process_cleanup.json), corroborated by the fresh
[closure audit](closure_process_cleanup.json), passes: all four devices have no
process owners, no vLLM/EngineCore process remains, and no reset was used.
The [focused follow-up review](stage_review_round2.md) returns **clean-pass**.
Implementation and evidence are already committed locally. No serving profiler was run and no
reset was needed during the accepted full-model run. Exact commands, failures,
repairs and artifact paths are retained in [work_log.md](work_log.md).

## Independent review and checkpoint status

[Round 1 review](stage_review_round1.md) identified only operator acceptance of
the tested omitted-top_k20 default. It found no concrete current implementation
defect, incorrect metric or missing mandatory runtime gate. The owner explicitly
approved that default on 2026-11-14; [round 2](stage_review_round2.md) returns
**clean-pass**, with no Required Work.
This approval is separate from Entry1. Source and recorded runtime evidence are
unchanged. Local checkpoints are `aa3d320cd6f0f1b64c0ddd9a5034c15b0f8b68bf`
(implementation), `e4746774a229575bd4bcd08c6be7cdba9692cdf6` (evidence), and
`cc9b199e26521aac5729f7f446e9d7b5088da8e5` (checkpoint log). Nothing was pushed.

## Approved serving command

Owner approval received 2026-11-14 via the multigoal runner:

> Owner decision: the tested default top_k=20 for requests that omit top_k is approved (it matches the model card recommendation). Record this approval beside the serving command, finish the follow-up review and complete the stage.

This approves the already-tested `--override-generation-config '{"top_k":20}'`
for omitted top_k; explicit values remain honored. The exact B32 server command
below comes from its [actual launch summary](../../readiness_vllm/final64_b32_untracked/vllm_run_summary.json).
The model-card rationale above is the owner's statement. The
[approval record](serving_default_approval.json) retains the exact decision and argv.
Required environment and full runner commands remain in
[measurement environment](final64_measurement_execution/environment.json) and
[commands.sh](final64_measurement_execution/commands.sh).

```bash
/home/ttuser/ornith-9b/tt-metal/python_env/bin/python -m vllm.entrypoints.openai.api_server --model ornith-ai/Ornith-1.5-9B --block_size 32 --max_num_seqs 32 --port 8000 --max_model_len 262144 --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 200000000, "fabric_config": "FABRIC_1D", "trace_mode": "decode_only"}}' --max-num-batched-tokens 1024 --language-model-only --async-scheduling --revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53 --tokenizer-revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53 --default-chat-template-kwargs '{"enable_thinking":false}' --override-generation-config '{"top_k":20}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

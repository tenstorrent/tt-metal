# Benchmark: google/gemma-4-26B-A4B-it

Status: **incomplete**. Both serving profiles are valid; accuracy is incomplete. Accuracy concurrency: 32.

| Benchmark | Completed / selected / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| MMLU-Pro | 8 / 280 / 12032 | exact_match,custom-extract | — | 82.60 | — | [Google model card](https://huggingface.co/google/gemma-4-26B-A4B-it#benchmark-results) |
| IFEval | 0 / 256 / 541 | Four upstream strict/loose prompt/instruction metrics | — | Unavailable | — | Unavailable |

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 2534.35 | 19.61 | 51.00 | 25.47 | 0.48 | 8.73 |
| 32 users | 32 | 32 | 4096 / 128 | 79663.91 | 699.92 | 1.43 | 24.30 | 0.49 | 6.84 |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for both serving profiles; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

## Run details

Subset: `ef36dff6d479319adc5b0438e79acc4e139bd108300e209479ab6ee864c4c994`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| MMLU-Pro (partial) | 8 | 0 | 0 | Unavailable per task |
| IFEval (incomplete) | 0 | 0 | 0 | Unavailable per task |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 40.20 | 0.20 |
| [32](perf-b32.json) | 96 / 96 | 505.67 | 0.19 |

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 1 | TTFT | 2534.35 | 2519.37 | 2583.56 | 2584.93 |
| 1 | TPOT | 19.61 | 19.64 | 19.73 | 19.73 |
| 1 | ITL | 19.65 | 19.62 | 20.26 | 26.02 |
| 1 | E2EL | 5024.60 | 5017.92 | 5064.99 | 5075.49 |
| 32 | TTFT | 79663.91 | 80044.04 | 80097.26 | 80097.49 |
| 32 | TPOT | 699.92 | 697.26 | 697.41 | 782.03 |
| 32 | ITL | 699.92 | 697.15 | 701.25 | 717.10 |
| 32 | E2EL | 168554.07 | 168613.32 | 168632.93 | 168633.39 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 32, prefill: 207.57 s. Source-derived full 30-layer TP4/EP4 actual invocation shapes and selected stored precision; active MoE weight reads per serial row, shared head per invocation; material activation estimates detailed in WORK_ACCOUNTING.md. Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; Four Blackhole ASICs, 120 physical cores/ASIC, 4096 LoFi FLOP/core/cycle; mixed-fidelity useful-work ratio against LoFi upper envelope, not FPU utilization; 512GB/s DRAM/ASIC (P300 1024GB/s per dual-ASIC card). Timing: Full normal dispatch through actual existing async completion/output construction; same-phase overlap counted once, gaps retained; transition gaps assigned to following phase. No added device waits. [Evidence](phase-accounting-b32.json).
- Concurrency 32, decode: 298.067 s. Source-derived full 30-layer TP4/EP4 actual invocation shapes and selected stored precision; active MoE weight reads per serial row, shared head per invocation; material activation estimates detailed in WORK_ACCOUNTING.md. Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; Four Blackhole ASICs, 120 physical cores/ASIC, 4096 LoFi FLOP/core/cycle; mixed-fidelity useful-work ratio against LoFi upper envelope, not FPU utilization; 512GB/s DRAM/ASIC (P300 1024GB/s per dual-ASIC card). Timing: Full normal dispatch through actual existing async completion/output construction; same-phase overlap counted once, gaps retained; transition gaps assigned to following phase. No added device waits. [Evidence](phase-accounting-b32.json).
- Concurrency 1, prefill: 17.6262 s. Source-derived full 30-layer TP4/EP4 actual invocation shapes and selected stored precision; active MoE weight reads per serial row, shared head per invocation; material activation estimates detailed in WORK_ACCOUNTING.md. Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; Four Blackhole ASICs, 120 physical cores/ASIC, 4096 LoFi FLOP/core/cycle; mixed-fidelity useful-work ratio against LoFi upper envelope, not FPU utilization; 512GB/s DRAM/ASIC (P300 1024GB/s per dual-ASIC card). Timing: Full normal dispatch through actual existing async completion/output construction; same-phase overlap counted once, gaps retained; transition gaps assigned to following phase. No added device waits. [Evidence](phase-accounting-b1.json).
- Concurrency 1, decode: 22.5267 s. Source-derived full 30-layer TP4/EP4 actual invocation shapes and selected stored precision; active MoE weight reads per serial row, shared head per invocation; material activation estimates detailed in WORK_ACCOUNTING.md. Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; Four Blackhole ASICs, 120 physical cores/ASIC, 4096 LoFi FLOP/core/cycle; mixed-fidelity useful-work ratio against LoFi upper envelope, not FPU utilization; 512GB/s DRAM/ASIC (P300 1024GB/s per dual-ASIC card). Timing: Full normal dispatch through actual existing async completion/output construction; same-phase overlap counted once, gaps retained; transition gaps assigned to following phase. No added device waits. [Evidence](phase-accounting-b1.json).

Error: Accuracy incomplete: interrupted after 8/536 responses at 942.54 s to reserve the original remaining budget for required serving measurements. No complete benchmark aggregate is available.

## Accuracy scope and retained inputs

Eight MMLU-Pro answers completed with normal `stop`; zero completed answers were
length-limited and zero had empty final content. The other 272 MMLU-Pro and 256 IFEval
questions have no completed response and are explicitly unscored. No subset
aggregate or published-score delta can be reported. This is an incomplete
measurement, not an accuracy verdict. The decision and observed progress are in
[budget-decision.json](budget-decision.json); the pre-recovery summary/report
remain separate artifacts. Accuracy was stopped deliberately before the deadline
to preserve time for the required performance measurements; the deadline did not
expire during accuracy.

All 536questions, exact upstream request arguments and document hashes are retained
in [benchmark-inputs.jsonl](benchmark-inputs.jsonl). All eight responses and links
are in [mmlu_pro/responses.jsonl](mmlu_pro/responses.jsonl) and
[mmlu_pro/request_links.jsonl](mmlu_pro/request_links.jsonl). Actual upstream
`task.apply_filters()` and `task.process_results()` produced the retained
[per-question scores](partial-accuracy-questions.jsonl); missing answers remain
null. [Partial accuracy metadata](partial-accuracy.json) records the denominator
and limitations. Completed answers are a completion-biased subset, so their
average is not presented as the benchmark score.

The frozen common manifest is `ef36dff6d479319adc5b0438e79acc4e139bd108300e209479ab6ee864c4c994`.
MMLU-Pro uses14upstream subject tasks, five-shot, `custom-extract` then exact-match;
IFEval is zero-shot with its four upstream metrics. Population, document and
few-shot hashes are in [manifest.json](manifest.json); full task/scorer settings
are in [task_settings.json](../setup-previous/preparation/task_settings.json).
The common request pool used lm-eval0.4.13, concurrency 32, seed0/1234 and identical
generation overrides:4096output-token cap, temperature1, top-p0.95, top-k64,
`do_sample=true`, native `enable_thinking=false`, empty textual stops, native EOS
retained. The published MMLU-Pro82.6% covers the full benchmark; the subset,
nonthinking mode, cap and upstream recipe here may differ.

Upstream chat messages reached the server as structured data. The server applied
the pinned native HF template once. The readiness probe's actual input-token hash
matches native rendering. For all eight completed benchmark questions, native
OpenAI text-part rendering produces the exact API-reported prompt lengths and
one BOS. A one-token difference from plain-string rendering is explained by the
native template's system text-part trailing space, not duplicated templating.
[Template audit](template-audit.json) and [diagnosis](../AUTODEBUG_template_audit.md)
state the evidence limits: individual benchmark input-token arrays were not
retained. Accuracy-only `logprobs=true,top_logprobs=0` selected the existing host
logits sampler for exact top64; the live route probe verifies this. Performance
uses greedy device sampling, verified by every measured phase event.

## Performance method and identity

These are full 30-layer vLLM token-out autoregressive measurements. Each profile
uses4096actual input and128actual generated tokens per request, no prefix cache,
greedy generation, ignored EOS and distinct measured prompt-token hashes.
Warmups completed1/1 and32/32; measured requests completed8/8 and96/96. The upstream
vLLM CLI also sends an initial sanity probe per invocation; probes and collector
readiness requests count toward elapsed stage time but are excluded from the
measured request cohorts. Client JSON retains request-level TTFT/TPOT/ITL/E2E
samples, percentiles, throughput and actual token counts. [Validation](performance-validation.json)
checks both warmups, both measured rows, identities and roofline evidence.

The single-user profile uses server slots1 and concurrency1, with the inherited
Stage10 decode-only trace configuration and selected precision. The other profile
uses slots32/concurrency 32. Both retain262144context capacity, TP4/DP1 on four
Blackhole ASICs (P300x2), one-billion-byte trace regions, FABRIC_1D, disabled
chunked prefill and disabled prefix caching. Startup logs show the actual
Blackhole discovery and four-device mesh. Model and tokenizer revision is
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; precision is
`head4_inner_all4_shared_down4`, with construction checks for every layer.
The imported module is
`models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm` in this checkout.
[Identity](../identity.json), both server records and both `configuration-bN.json`
files retain imported paths/hashes, all layer indices, actual `/server_info`,
launch argv and selected environment values.

Measured model checkout: `3432e2048c20b80aa2f72dc6f801f4d4d722db7d`;
TT plugin checkout: `7f72b1c6e905f5137fe3377f2e7b42738d3f271d`.
The actually imported vLLM core is an installed0.26.0+empty wheel, distinct from
that plugin checkout. It embeds no core commit ID; identity records this as null
and retains its source-file hashes. No source commit is invented for the wheel.

## Full-phase accounting limits

Both profiles exported their raw dispatch/completion events before their owned
server stopped. No live device profiler or additional device synchronization was
used. Existing model trace warm/capture synchronization was left unchanged.
Host intervals include forward work, normal asynchronous completion, sampling,
communication, host work and gaps. Initial HTTP queueing and final transport are
outside these model phase windows and remain represented by client latency.
Same-phase overlap is counted once; transition gaps belong to the following phase.

The collector includes extra executed warm forwards on trace rebinding:
C32 has 381 decode submissions plus 3 warm executions; C1 has 1016 decode submissions
plus 8 warm executions. Trace capture records commands into a host bypass buffer;
it is not counted as another executed model pass. The C32 work estimate was
corrected offline from its unchanged pre-stop export; its measured time and raw
client results were unchanged. Source audit and assumptions are in
[WORK_ACCOUNTING.md](../WORK_ACCOUNTING.md).

Prefill uses useful active-MoE FLOPs. Decode uses estimated stored-precision DRAM
traffic including padding, exponents, actual KV windows and per-row expert reads;
activation/metadata traffic has explicit approximations. The prefill peak is a
four-ASIC nominal 1.35 GHz LoFi upper envelope of 2.654208 PFLOP/s, not an attained or
fidelity-weighted mixed-precision peak. The graph also uses HiFi2/HiFi4 and SFPU
work. DRAM peak is 4 × 512 GB/s. These percentages are **host-wall estimates**, not
device utilization; no percentages were clamped. Device-time/roofline telemetry
fields remain null because matching device measurements are unavailable.

## Timing, checks and remaining failure

[Setup](../setup.json) is separate from the client clock: the operator provisioned
the evaluation environment; datasets, tokenizer and transport checks were
prepared before invocation. No supplied Stage 10 server remained alive; its prior
startup was approximately 190 seconds, with its command/log retained. The first
attempt failed at configuration inspection after 178.0 seconds and is preserved
under [attempts/01-configuration-endpoint](../attempts/01-configuration-endpoint/REPORT.md).
AutoFix enabled the installed wheel's configuration endpoint on localhost;
no inference configuration changed. The resumed run's 32-slot and 1-slot launches
took 176.2 seconds and 186.2 seconds, respectively, inside its clock.

Performance recovery retained the original conservative process-start deadline
in [lifecycle.json](lifecycle.json), including all prior accuracy time. Its initial
report finished at 1917.7 seconds; final verification/reporting time is recorded
below and in summary.json. No one-hour completion claim is made for the stage:
accuracy is incomplete regardless of the elapsed time. The evidence gate fails
on incomplete status; the full 262144 context check passes. Forty-two host tests
pass after the collector fixes. Python/docs-only changes required no C++ build.
All owned serving processes were stopped after phase collection; no reservation
was acquired/released and nothing was pushed.

Remaining requirement: complete the frozen 280 MMLU-Pro and 256 IFEval questions at
concurrency 32 and produce their upstream aggregate scores within the original
stage contract. This attempt does not satisfy that requirement and must not be
marked complete. Faster execution or an owner-approved change to the time/workload
contract is needed before another completion attempt.

Final client-stage wall time through verification and reporting: 3014.2 seconds (original clock, not reset).

A subsequent [source-only runtime audit](../AUTODEBUG_runtime_limit.md) found no
proven transport/scheduling defect. Trace turnover and full-logit host sampling
are optimization hypotheses, with no measured repair. Runtime projections do
not prove mathematical impossibility; this attempt remains incomplete.

The third continuation audit leaves the goal **blocked**, with 528 accuracy
responses missing and no authorized time-limit exception or faster validated
implementation. This is an incomplete measurement, not an accuracy verdict.
See [continuation-audit-03.json](continuation-audit-03.json).

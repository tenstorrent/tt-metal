# Qwen/Qwen3.6-35B-A3B benchmark handoff

| Benchmark | Frozen samples / full population | Metric | Score % | Published full % | Delta pp | Evidence |
|---|---:|---|---:|---:|---:|---|
| MMLU-Pro | 14 / 12,032 | subject_macro:exact_match,custom-extract | 78.57 (11/14) | 85.20 | -6.63 | [Upstream scores](mmlu_pro/results.json); [published model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| IFEval | 18 / 541; 17 responses retained | Prompt strict | Unavailable: incomplete | Unavailable | — | [Partial upstream evidence](ifeval/partial-results.json) |
| IFEval | 18 / 541; 17 responses retained | Instruction strict | Unavailable: incomplete | Unavailable | — | [Partial upstream evidence](ifeval/partial-results.json) |
| IFEval | 18 / 541; 17 responses retained | Prompt loose | Unavailable: incomplete | Unavailable | — | [Partial upstream evidence](ifeval/partial-results.json) |
| IFEval | 18 / 541; 17 responses retained | Instruction loose | Unavailable: incomplete | Unavailable | — | [Partial upstream evidence](ifeval/partial-results.json) |

| Required performance profile | Concurrent requests planned | Server slots observed (planned) | Input / output tokens | Throughput / TTFT / TPOT / ITL / end-to-end | Prefill / decode rooflines | Status |
|---|---:|---|---:|---|---|---|
| Batch 1 | 1 | — (1) | 4096 / 128 | — | — / — | Not measured |
| Batch 8 | 8 | — (8) | 4096 / 128 | — | — / — | Not measured |
| Batch 16 | 16 | — (16) | 4096 / 128 | — | — / — | Not measured |
| Batch 32 | 32 | 32 (32) | 4096 / 128 | — | — / — | Not measured |
| FUNCTIONAL reference | 1 | 32 (32) | 128 / 128 | — | Not required | Not measured |
| Long-context lone user | 1 | 32 (32) | 32768 / 128 | — | Not required | Not measured |

**Stage incomplete.** A fixed 3,600-second HTTP timeout in the accuracy client aborted the shared evaluation before performance began. The parent client clock had not expired. All 14 MMLU-Pro responses and 17 of 18 IFEval responses were saved. Offline replay recovered upstream scores without new inference. IFEval document 59 has no response; it remains in the frozen denominator and is neither dropped from the requested task nor assigned a fabricated answer. No complete IFEval headline is reported.

No batches were skipped as unsupported: `skipped_batch_sizes` is empty. The four batch profiles and both lone-user rows remain outstanding. Warmup and setup smoke measurements below do not satisfy those rows. Required per-profile roofline accounting is consequently missing. This is a measurement-completeness failure, with no accuracy acceptance verdict.

The MMLU-Pro delta is descriptive: this small frozen subset differs from the published full population, and the published scorer/generation recipe was not established as identical. GPQA preparation encountered gated dataset access; MMLU-Pro and IFEval were selected before observing scores. No exact-model IFEval reference was available on the cited model card.

## Answer statistics and retained scores

| Task / scope | Extracted | Normal stops | Token-limit stops | Blank at cap | Output tokens median / p90 / max | Configured / recommended cap |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro, complete frozen subset | 14/14 (100%) | 14 | 0 | 0 | 1521 / 2295 / 2560 | 32768 / 32768 |
| IFEval, collected responses only; 1 missing | 17/17 (100%) | 17 | 0 | 0 | 2438 / 5809 / 6025 | 32768 / 32768 |

These statistics were read and checked. MMLU-Pro uses upstream `custom-extract`; IFEval uses `none`. No retained answer is blank or cap-limited. Full IFEval statistics remain unavailable because one transport request failed; its missing response is distinct from an empty model answer. [MMLU-Pro statistics](mmlu_pro/answer-statistics.json), [partial IFEval statistics](ifeval/partial-answer-statistics.json).

For audit only, the 17 collected IFEval answers score 15/17 prompt strict and loose (88.24%), and 27/29 instruction strict and loose (93.10%). These are partial-response diagnostics, not the frozen 18-question benchmark score. All upstream per-document scores and raw responses are retained.

## Protocol and identity

- Upstream lm-eval 0.4.13: MMLU-Pro recipe metadata 3.1, five validation shots using `first_n`, subject CoT descriptions, regex `answer is \(?([ABCDEFGHIJ])\)?`, `take_first`, exact match ignoring case/punctuation. IFEval metadata 4.0, zero-shot, train evaluation population, upstream prompt/instruction strict and loose scorers. [Recipe sources](setup/task-recipes.json).
- Common frozen seed `tt-bringup-benchmark-v1`; MMLU-Pro reduced 280→14 (one per subject), IFEval 256→18, using leading common rankings before scores. Pilot speed and reductions are recorded in [configuration](run_config.json). Population, document, few-shot and manifest hashes are in [manifest](manifest.json); all 32 questions, structured contexts and request identities are in [benchmark-inputs.jsonl](benchmark-inputs.jsonl).
- Native pinned HF chat template applied exactly once by the server to structured messages; few-shots as multiple turns. `enable_thinking=true`, temperature 1, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1, do_sample=true, max_gen_toks 32768, until=[]; native EOS preserved. Only final content is scored. Native presence penalty uses the supported host sampler; greedy performance warmup uses device sampling.
- Generated `models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm`, all 40 layers; checkpoint and tokenizer revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`; selected `baseline_bfp8` precision policy, BF16 activations/KV/CCL with policy-specific FP32 state/arithmetic. Exact policy and imported module hash are in [identity](../identity.json) and [observed configuration](configuration-b32.json).
- P300x2: four Blackhole p300c chips, TP4/EP4, mesh 1×4, FABRIC_1D; serial exclusive regime, 110 workers/chip and 8 DRAM banks/chip. Full context 262144, block 32, max batched tokens 2048, decode-only trace 805306368 bytes, 8224 KV pages. Prefix caching remained false, matching Stage 9/10, for accuracy and warmup. Server slots 32 and HTTP concurrency 32 are recorded separately.
- Executed source tt-metal `b08fe50aea6f34b9790d8eec5005ed3d7d37df6d`; vLLM TT plugin `5d1785b76d17a8989c9495f6939abea6cef62f09`; server vLLM 0.31.0+empty from PyPI sdist (release association, not embedded build SHA); performance client upstream vLLM 0.26.0. Server/client tokenizer compatibility was repaired by matching transformers 5.12.1/tokenizers 0.22.2. [Package provenance](setup/server-package-identity.json), [launch argv](perf-b32-server.json), [source identities](setup/source_commits.json).

## Setup measurements, not required warmed profiles

| Setup workload | Server slots | Concurrent requests | Completed | Exact total input / output | Wall seconds | Requests/s | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Explicit 4096/128 warmup | 32 | 32 | 32/32 | 131072 / 4096 | 2977.948 | 0.010746 | 1.375444 |

| Warmup latency | Mean ms | Median ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| TTFT | 1506254.694 | 1506229.159 | 2781283.620 | 2895278.113 |
| TPOT | 11544.736 | 11546.393 | 21533.648 | 22422.329 |
| ITL | 11544.736 | 432.239 | 91551.622 | 91906.250 |
| E2EL | 2972436.173 | 2972621.124 | 2977883.416 | 2977933.816 |

Warmup reciprocal mean TPOT is 0.086620 tokens/s/user. Interleaved prefill delays are included in these client latencies. This setup workload is not a warmed repeated measurement. [Raw result](perf-b32-warmup.json).

The completion ledger reconciles all 32 warmup requests, 64 prefill chunks and 158 decode calls: full prefill 2898.122657s and full decode 79.790499s. Modeled useful prefill work 729810879906560 FLOPs and decode traffic 14258318950168 bytes give setup-only roofline estimates 0.217549% and 8.725437%. [Accounting](warmup-accounting.json).

Timing ends at existing async completion/readback boundaries; host gaps are included, same-phase overlaps are unioned, and cross-phase overlap is rejected. The ledger was flushed at idle before server stop. No device synchronization or live profiler was added. Work is source-derived: decode traffic includes executed packed rows, KV/state and model weights, with explicit expected-unique-expert estimates and bounds rather than measured routes or DRAM counters. Peaks use nominal hardware limits and a work-weighted fidelity ceiling, not measured sustainable rates. [Accounting method](setup/accounting_design.md), [implementation](../../../tools/benchmark_work.py), [hardware source](https://tenstorrent.com/hardware/cards).

A separate two-request setup smoke at concurrency 2 on 32 slots completed exact 4096/128 requests and validated the collector before accuracy. Its full phase evidence remains in [setup/collector-smoke-tokenizer-fixed](setup/collector-smoke-tokenizer-fixed/). The initial tokenizer-mismatch smoke is preserved separately and is not a valid 4096-token result.

## Clocks, failure and checks

Attempt 2 runner execution: client 3616.330/7200s, setup 2983.741/10800s. Continuous post-failure recovery/cleanup/check/report work: 829.457s. Client through this report: **4445.787/7200s**; setup remains **2983.741/10800s**; total 7429.528s. Clock anchoring uses the original failed-summary timestamp with subsecond uncertainty. [Summary](summary.json), [unchanged failed execution](failed-execution/summary.json).

Pre-invocation setup was separately recorded: 2382.085s from the documented 02:10UTC start (minute precision), including the fresh 275.533s server launch and the negligible first failed attempt. Attempt 1 used 0.002096s client and 0.063737s setup before a canonical launch-flag validation fault; its evidence is retained under [attempts/attempt1](attempts/attempt1/). Both fresh attempts are preserved.

There is also a measured clock conflict independent of the HTTP timeout: the runner requires 176 repeated 4K prompts across the four profiles. The validated setup smoke measured 91.229992s prefill per prompt, projecting 16056.479s of prefill alone, before accuracy and the long-context rows, against 7200s client allowance. No extended budget, omitted batch or reduced context was assumed.

AutoFix repaired tokenizer compatibility and canonical capacity recording. The remaining hidden HTTP-timeout fault is repaired and CPU-tested in the editable workflow source; its local commit/test evidence is linked from [RUN_NOTES](../RUN_NOTES.md). It was not installed into the cached package or validated by a third hardware attempt. Offline score recovery prohibits socket/model requests, matches all 31 saved request-instance hashes and retains all 32 inputs. [Recovery provenance](recovery-provenance.json).

The context check passes with target=supported 262144 and preserves `doc/context_contract.json`. The benchmark evidence gate exited 2 (CRITICAL: stage did not complete); its result is retained in [checks](checks.json). [Context log](context-check-final.log), [evidence log](evidence-check-final.log). Owned idle-server shutdown succeeded; all four devices are free, with no reset or foreign-process termination. [Cleanup verification](cleanup-verification.json).

Raw logs and full phase snapshots remain intact locally under run/. Three performance JSON files received only a final newline for repository checks; their original bytes and both hashes are preserved in [formatting provenance](raw-formatting-provenance.json). Repository log exclusions and its 500 KiB file limit keep those artifacts out of Git; [artifact-retention.json](artifact-retention.json) lists them. Frozen inputs, raw accuracy responses, upstream scores, compact accounting, identity and the report are versioned. All commits are local; nothing was pushed. See [RUN_NOTES](../RUN_NOTES.md) for commit SHAs and remaining work.

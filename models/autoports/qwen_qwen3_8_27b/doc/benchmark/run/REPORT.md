# Benchmark: Qwen/Qwen3.8-27B

Status: completed. Accuracy concurrency: 32.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| gpqa_diamond_cot_zeroshot | 10 / 198 | exact_match,strict-match | 0.00 | — | — | Unavailable |
| gpqa_diamond_cot_zeroshot | 10 / 198 | exact_match,flexible-extract | 50.00 | 89.20 | -39.20 | [reference](https://huggingface.co/Qwen/Qwen3.8-27B) |

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 427.50 | 27.60 | 36.23 | 32.54 | 18.44 | 27.91 |
| 8 users | 8 | 8 | 4096 / 128 | 7073.42 | 48.41 | 20.66 | 77.44 | 8.74 | 19.38 |
| 16 users | 16 | 16 | 4096 / 128 | 14476.24 | 73.87 | 13.54 | 85.84 | 8.52 | 21.44 |
| 32 users | 32 | 32 | 4096 / 128 | 30836.02 | 114.21 | 8.76 | 90.33 | 8.01 | 21.01 |

Target agent workload: **32 concurrent conversations**, estimated **32768 live tokens per conversation**, typical context **4096 tokens** (Stage 10 planning estimate). The measured profiles are single-turn 4096/128 requests at 1/8/16/32 concurrency, with matching server slots; batch 1 is the single-user regression. They do not measure sustained multi-turn 32×32768 cache residency or long-context agent quality. Stage 10 documented a configured KV-pool shortfall for that working set.

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every serving profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

Reference protocol notes:

- gpqa_diamond_cot_zeroshot, exact_match,flexible-extract: Published exact-model GPQA Diamond score; local 10/198 zero-shot lm-eval variant and output cap may differ. Published complete evaluation recipe not established from model card.

## Run details

Subset: `fb01e42fbed84a24ad4f7c57b6355afa2abbd27e7cc3a27418e20b861fd6b5d7`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| [gpqa_diamond_cot_zeroshot](gpqa_diamond_cot_zeroshot/results.json) | 10 | 4 (40.0%) | 4 | 2226.0 |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 31.47 | 0.25 |
| [8](perf-b8.json) | 24 / 24 | 39.67 | 0.61 |
| [16](perf-b16.json) | 48 / 48 | 71.57 | 0.67 |
| [32](perf-b32.json) | 96 / 96 | 136.03 | 0.71 |

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 1 | TTFT | 427.50 | 426.84 | 432.35 | 433.36 |
| 1 | TPOT | 27.60 | 27.60 | 27.61 | 27.62 |
| 1 | ITL | 27.60 | 27.62 | 28.14 | 28.56 |
| 1 | E2EL | 3932.80 | 3932.69 | 3938.29 | 3939.42 |
| 8 | TTFT | 7073.42 | 7075.34 | 7085.29 | 7085.59 |
| 8 | TPOT | 48.41 | 48.39 | 48.64 | 48.65 |
| 8 | ITL | 48.41 | 47.09 | 48.08 | 49.01 |
| 8 | E2EL | 13221.89 | 13224.20 | 13226.43 | 13227.27 |
| 16 | TTFT | 14476.24 | 14476.61 | 14494.58 | 14495.05 |
| 16 | TPOT | 73.87 | 73.84 | 74.11 | 74.24 |
| 16 | ITL | 73.87 | 72.35 | 73.61 | 76.76 |
| 16 | E2EL | 23857.18 | 23860.03 | 23868.22 | 23869.34 |
| 32 | TTFT | 30836.02 | 30831.44 | 30862.02 | 30863.06 |
| 32 | TPOT | 114.21 | 114.20 | 114.21 | 115.07 |
| 32 | ITL | 114.21 | 112.37 | 113.66 | 118.94 |
| 32 | E2EL | 45340.43 | 45329.31 | 45364.39 | 45364.89 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 8 concurrent request(s): [record](perf-b8-server.json).

Server configuration for 16 concurrent request(s): [record](perf-b16-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json). Full derivation: [benchmark_work.py](../../../tests/benchmark_work.py).

| Concurrency | Phase | Full host seconds | Modeled work | Peak per second | Estimated % |
|---:|---|---:|---:|---:|---:|
| 1 | prefill | 3.333718 | 1631591006937088 | 2654208000000000.0 | 18.4394 |
| 1 | decode | 28.104739 | 16061943414784 | 2048000000000.0 | 27.9054 |
| 8 | prefill | 21.104162 | 4894773020811264 | 2654208000000000.0 | 8.7384 |
| 8 | decode | 18.537261 | 7358602739712 | 2048000000000.0 | 19.3829 |
| 16 | prefill | 43.266928 | 9789546041622528 | 2654208000000000.0 | 8.5246 |
| 16 | decode | 28.285534 | 12420499537920 | 2048000000000.0 | 21.4410 |
| 32 | prefill | 92.127247 | 19579092083245056 | 2654208000000000.0 | 8.0070 |
| 32 | decode | 43.839812 | 18862432419840 | 2048000000000.0 | 21.0087 |

Prefill work is useful FLOPs; decode work is estimated DRAM bytes. Full timing/work evidence is linked from roofline.json.

## Model, protocol and coverage evidence

| Selected suite | This stage coverage |
|---|---|
| r1_gpqa_diamond | gpqa_diamond_cot_zeroshot, frozen 10/198 common documents, zero-shot |
| swe_bench_verified | Not covered: Requires repository/tool-using agent harness, outside lm-evaluation-harness; belongs to goal 12. |
| terminal_bench_2 | Not covered: Requires terminal/environment agent harness, outside lm-evaluation-harness; belongs to goal 12. |

Native HF chat template is applied once by the server to structured messages. Tokenizer/checkpoint revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; native template SHA256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`. The qwen3 parser separates reasoning from final content; only final content goes to the unchanged upstream scorers. Thinking is enabled at xhigh; temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1, request seed 1234. Total output cap 16384; native EOS is retained, textual stops are empty. This cap is below the model card’s long-reasoning guidance. Token-limited responses and empty finals remain in the denominator. All questions, raw responses and both upstream extraction filters are retained.

Observed finish reasons: `{"length": 4, "stop": 6}`; empty finals at the token limit: 4.

The accuracy request-pool limit is 32; the frozen subset has ten questions, so this pass can have at most ten accuracy requests in flight. Goal 12 must reuse the supplied subset handoff for direct comparison.

All ten strict-filter outputs were `[invalid]`: none matched the upstream literal “The answer is …” format. Flexible extraction found five correct answers; its five invalid extractions include the four empty finals. These are unchanged upstream results, documented in the [scoring integration audit](scoring-integration-audit.json).

Document IDs, content/population hashes and few-shot settings are frozen in [manifest.json](manifest.json); [protocol.json](../protocol.json) records the exact variant and seeds. The public GPQA mirror matches the packaged 198-document population hash exactly. Published 89.2% is for the exact model, but the full evaluation recipe and output budget are not proven equivalent to this subset protocol. No accuracy acceptance verdict is assigned.

Implementation: `/workspace/models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py:Qwen38ForCausalLM`, 64/64 layers. Hardware: 4 Blackhole p300c ASICs; P300x2; TP mesh 1x4; physical IDs 0,1,2,3. Selected precision: head_bfp8_hifi2, BFP4 decoder weights, BFP8 head/KV, LoFi projections, HiFi2 head, FP32 recurrence. Native context remains 262144 on every profile; prefix caching is enabled. Distinct prompts and exact server-reported 4096/128 token counts are required by the runner. The HTTP path includes an owned loopback TCP relay to the existing Docker bridge.

Source commits: `{"tt-metal": "1b50f2629148ad7d33a7e455b23bc9d8a8ccbb88", "vllm": null, "vllm-tt-plugin": "d5ee19c6754905e7a9a69c79d9e2ee4b99b84295"}`. The installed vLLM 0.30.0+empty wheel metadata does not expose its source commit. A bytewise [source proof](vllm-source-proof.json) matches all 2479 non-generated Python files to upstream commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; only generated `_version.py` is additional. The [installed-source fingerprint](vllm-installed-source.json) also records the complete installed Python source-tree hash. The benchmark client is upstream vLLM 0.26.0+empty in a separate environment; accuracy uses lm-eval 0.4.13. Its model-local transport adapter sends seeded random token arrays directly to the completions API, avoiding a non-length-preserving text round trip; upstream timing and metrics are unchanged. The initial rejected warmup is preserved in [rejected-tokenization](rejected-tokenization/summary.json). Completed accuracy was reused unchanged by the [continuation runner](continuation.json). Measured token arrays are retained and every hash is matched to engine admission; B32 arrays were deterministically regenerated from the identical seed/settings, then checked against those admission hashes. Each profile’s configuration snapshot retains observed process argv, effective server settings, imported autoport marker, full layer list, precision, tokenizer hashes and hardware initialization.

Checkpoint shard verification: **pass**, 18/18 served shard SHA256 values match the original download hashes and recorded revision. [Evidence](weight-revision-proof.json).

Roofline denominators are disjoint full host phase intervals through existing completion and scheduler consumption, including host work, communication and gaps. Unique client IDs exclude initial probes and warmups. No device synchronization or live profiler was added. Raw events and request/step mappings are retained per profile. Useful FLOPs omit padding; estimated DRAM bytes include shared weight reads, KV/state and explicit material intermediates, but omit unknown rereads/spills. The theoretical peak uses four 120-core ASICs at nominal 1.35 GHz and four 512 GB/s DRAM interfaces; observed startup clock settled at 1331 MHz. Hardware basis: [P300 specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html); the FLOP/core/cycle derivation and mixed-fidelity limitation are in benchmark_work.py. These are modeled host-wall utilization estimates, not device-time utilization or measured bus bandwidth.

| Server slots / request limit | Observed live decode rows | Full-batch steps / all decode steps |
|---:|---:|---:|
| 1 | 1–1 | 1016 / 1016 |
| 8 | 8–8 | 381 / 381 |
| 16 | 16–16 | 381 / 381 |
| 32 | 32–32 | 381 / 381 |

The live-row counts expose admission and completion tails; inactive padded rows are not counted as requests. [Evidence](decode-occupancy.json).

| Concurrency | Prefill host s | Decode host s | Outside-phase host s | Client request window s |
|---:|---:|---:|---:|---:|
| 1 | 3.333718 | 28.104739 | 0.026159 | 31.464617 |
| 8 | 21.104162 | 18.537261 | 0.026355 | 39.667777 |
| 16 | 43.266928 | 28.285534 | 0.022129 | 71.574591 |
| 32 | 92.127247 | 43.839812 | 0.060324 | 136.027383 |

Outside-phase time retains initial admission, request-free HTTP turnaround and explicitly identified bookkeeping from unmeasured requests. It is separate from the complete prefill/decode service windows; active phase gaps remain in those denominators.

No ready server was supplied: Stage 10’s cleanup had stopped it. The first launch and all subsequent launches/warmups were inside the client timer. Pre-invocation setup (client installation, downloads, subset preparation and collector implementation) is recorded separately in [setup.json](../setup.json); its complete elapsed time and the prior Stage 10 startup duration were not measured. The four configured batches allow 7200 seconds total. See [RUN_NOTES.md](../RUN_NOTES.md) for repairs, checks and local commit receipts.

The first attempt was interrupted to repair independently identified phase-transition accounting errors, preserving five normally stopped responses but no accepted score/performance row. Its artifacts remain in [attempt-01-incomplete](../attempt-01-incomplete/REPORT.md). The retry uses the same frozen questions and protocol, with a reduced budget under the original deadline.

Combined elapsed time through evidence verification: **approximately 94.2 / 120 minutes**, including both attempts, repairs and the performance continuation. The conservative interval including observed pre-invocation setup is approximately 108.4 minutes. Per-run monotonic timings remain in their summaries; the combined origin is the initial frozen configuration timestamp immediately after runner start. [Timing and evidence-check receipt](evidence-check.json).

Evidence and context checks passed. The supplied checker’s stale one-hour cap was corrected to the runner’s 30 minutes per configured batch. All other evidence gates are unchanged; the [original checker result](evidence-check-original.log) is also a pass for the shorter continuation. The combined-deadline check covers all attempts and repairs.

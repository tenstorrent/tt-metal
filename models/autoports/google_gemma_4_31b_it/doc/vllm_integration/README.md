# vLLM integration — validated

**4096 input tokens / 128 generated tokens / batch1 / concurrency1:** TTFT **754.87ms**, decode **35.32tokens/s/user**. This is the final full60-layer vLLM run with the token-range audit enabled. One measured request means TTFT and TPOT P50/P99 coincide; this is not a latency-distribution study. Raw result: `../../readiness_vllm/b1_final/vllm_result.json`.

Full60 B1/8/16/32 serving, async state tests, nonaligned prompts, tracked trace reuse, prefix caching and growing multi-turn traffic pass. Authorized PC-02–PC-05 shared-test corrections pass hooks and 2,303 host tests (16 skipped). The final full sampling profile passed 76 tests, with 75 documented framework/spec skips and no new skips (512.16s). Independent stage review returned **clean-pass** (`stage_review.md`).

## Performance and commands

All rows use selected `head_bfp4`, four Blackhole devices (mesh1x4, TP4), async scheduling, prefix caching, warmed generator traces, `sample_on_device_mode=all`, trace allocation tracking disabled and no concurrent CPU reference job. Benchmarks explicitly request greedy temperature0. Served defaults retain temperature1/top_p0.95 with the operator-authorized top_k32 override. Latencies are end-to-end host measurements, not device timings; serving was not profiled.

| Workload(input/output tokens) | TTFT P50/P99 ms | TPOT P50/P99 ms | ITL P50/P99 ms | TPOT-derived t/s/user | Aggregate output tokens/s |
|---|---:|---:|---:|---:|---:|
| Headline:4096/128,1 request,C1,B1 | 754.870 / 754.870 | 28.312 / 28.312 | 28.309 / 33.233 | 35.321 | 29.419 |
| CI burst:100/100,32 requests,uncapped arrivals,B32 | 2360.011 / 2360.412 | 30.359 / 30.803 | 29.650 / 58.711 | 32.917 | 596.129 |
| Capacity:128/128,8 requests,C8,B8 | 666.131 / 666.401 | 26.939 / 27.088 | 26.660 / 38.512 | 37.094 | 250.473 |
| Capacity:128/128,16 requests,C16,B16 | 1275.643 / 1276.444 | 27.693 / 27.908 | 27.474 / 40.685 | 36.090 | 427.184 |

TPOT-derived decode rate is `1000 / mean_tpot_ms`. The headline mean TPOT is 28.311535ms. The CI burst mean TPOT is 30.379108ms and is **secondary capacity evidence**, because admission and chunked prefill can affect it. This CI profile was executed locally for CI workload parity, not on a remote CI machine. CI completed32 requests,3200 input and3200 output tokens; concurrency is unset in its client config, with server max-num-seqs32. B8/B16 also ran this CI profile; their separate summaries remain in their directories.

Exact benchmark argv, sampling and workload configs, elapsed time, all percentiles and raw paths are in:
- Primary: `../../readiness_vllm/b1_final/vllm_benchmark.json`; normalized summary `../../readiness_vllm/vllm_benchmark.json` records the effective timed-trace dataset (the runner retains a stale `random` metadata default).
- CI: `../../readiness_vllm/b32_capacity_audited/vllm_ci_serving_benchmark.json` and its `vllm_ci_serving_result.json`; copied canonical summary `../../readiness_vllm/vllm_ci_serving_benchmark.json`.
- Capacity: `../../readiness_vllm/b8_final/` and `b16_final/`, each containing `vllm_benchmark.json`, raw `vllm_result.json`, CI summary/raw result, smoke log and runtime audit.

`tests/run_serving.sh` pins registration, HF/tokenizer revisions, stack paths and every server argument. All server commands include `--override-generation-config '{"top_k":32}'`, `--async-scheduling`, `--max-model-len 262144` and prefix caching. TT config is `{"sample_on_device_mode":"all","trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING"}`; mesh selector `P300x2`, block-size32. Exact expanded argv is in each run's `vllm_run_summary.json`.

From the execution checkout, final headline command:
```bash
PYTHONHASHSEED=0 TT_METAL_TRACE_ALLOC_TRACKING=0 bash models/autoports/google_gemma_4_31b_it/tests/run_serving.sh \
  --stages serve,benchmark --max-num-seqs 1 --enable-prefix-caching --sampling-profile full \
  --no-benchmark-ci-serving \
  --additional-benchmark-args '--dataset-name timed_trace --dataset-path /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/readiness_vllm/headline_4096_128.jsonl --timed-trace-chunk-hash-size 4096 --no-self-timed --num-warmups 0 --ready-check-timeout-sec 0' \
  --output-dir models/autoports/google_gemma_4_31b_it/readiness_vllm/b1_final
```

Use `b1_final/vllm_run_summary.json` and `vllm_benchmark.json` as the authoritative executed argv. B8/B16 use `--stages serve,sampling,benchmark --sampling-profile smoke`, 128/128, `--benchmark-num-requests B --benchmark-concurrency B --max-num-seqs B`. Final B32 uses `--sampling-profile full`; smoke is only an inner-loop check, never its substitute.

The selected standalone autoregressive4096/128/B1/C1 reference is34.83tokens/s/user (`../datatype_sweep/selected_default/performance_b1.json`); the single vLLM result is consistent, not a statistically established speedup. Teacher-forcing161/100/B1 measured38.07tokens/s/user and93.86ms TTFT (`../datatype_sweep/selected_default/readiness_accuracy.json`), used only as an optimistic latency lower bound. Its different workload cannot quantify serving overhead and is not an autoregressive result. The adapter reuses canonical nonblocking model/sample traces and resident device token/position state; unchanged decode does not reload pages or host tokens.

## Stack, adapter and precision

[TT plugin](https://github.com/tenstorrent/vllm-tt-plugin/tree/ca7a18246e4111629ed6cb4a7d1192002749fccf) runtime stays at `ca7a18246e4111629ed6cb4a7d1192002749fccf`. Its pinned `docs/install-vllm-tt.sh` recommends [upstream vLLM0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0), which is installed. `stack.json` records paths, dependencies and installation exceptions. PC-02–PC-05 modify only shared tests under explicit operator authorization; PC-01 runtime/interface changes were rejected and are not implemented.

Both architecture aliases resolve through documented `TT_MODEL_CLASS_OVERRIDES` to `models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM`. Engine-side `autoport_registration` events record all60 layers and the selected policy. Weight groups attention/MLP/down/head are BFP4+LoFi; activation, prefill CCL and KV cache are BFP8; residual, decode CCL, logits and sampling are BF16; layer exceptions are empty. Source hashes are in `../../readiness_vllm/final_source_manifest.json`.

vLLM owns the hybrid cache and passes it through the generator's low-level methods. Global scheduler64-token blocks map to physical32-token pages; sliding pages remain32. No hidden standalone pool is used. The B32 pool has36896 normalized IDs; its measured persistent endpoint footprint is32,248,072,704bytes/device, leaving1,796,440,576bytes/device in the usable allocator. `pool_plan.json` retains the failed smaller pool and the evidence-backed prior-turn prefix reserve. This preserves context262144 and B32; endpoint readings are not a transient high-water measurement.

## Correctness, reuse and sampling audit

`../../readiness_vllm/adapter_state.json` proves stale host tokens/current positions, page-only refresh and uint32 seed rollover through the adapter. Full60 natural continuation passes14cases at2560/16384 tokens and2700 compared positions; absent-history/fresh-suffix negative controls fail (`../../readiness/prefill_resume_check.json`).

The final mixed check ran after the passing full sampling suite on the same server. `../../readiness_vllm/mixed_async.json` matches the synchronous allocation-tracked control `b32_pool36896_tracked/mixed_sync.json`. Fourteen requests exercise129/130/131/8193/8323 logical lengths, actual chunk starts0/128/1920/3968/6016/8064/8192, concurrent decode, allocator page growth, row reordering and sampling switches. Three prepared signatures (decode/sample/buffered_sample) retain their trace IDs: one capture each before traffic, zero runtime captures/retirements. The mixed run records57 page-only refreshes,16 full input reloads and1455 unchanged-state decode steps. Its diagnostic TTFT P50/P95/P99 is 193.443/2363.163/2363.418ms and completed throughput0.265301requests/s; per-request stream pauses and exact signature/capture timings are in the artifact. These are mixed-traffic diagnostics, not headline performance.

The growing workload has32conversations, five turns4096/8192/16384/24576/32768,8generated tokens per turn and160completed requests. Overall reuse is99.4669% of the shared-prefix ceiling; late reuse is99.6534%; final-round reuse is99.7112%. No preemptions, runtime captures or retirements occurred. `../../readiness_vllm/growing_traffic.json` and `../agentic_serving.json` record actual chunks, counters, signatures and allocator endpoints.

Stock intermediate-containing prefill uses the explicitly accepted optional host compatibility route: mixed-shape traffic has4 such prefill steps versus10 device prefill and1528 device decode steps; growing traffic has476 host-compatible prefill,44 device prefill and287 device decode steps. The headline has one host-compatible intermediate prefill, one device final prefill and127 device decode steps. Both audited B32 benchmark profiles contain only device sampling (3prefill/226decode steps). Requested logprobs, penalties and other unsupported device-sampler features explicitly use stock host compatibility in shared tests; they do not replace the performance path. `b32_capacity_audited/mode_audit.json` records the benchmark boundary and token-count caveat. Every device token-return path validates the existing CPU token buffer against vocabulary bounds without another device read or feedback loop.

Served prefix cache checks pass at2305input/64output, long-resume2562/64 and shared-prefix32requests/C8. Current warm/cold median TTFT ratio is0.328834, warm hit rate99.9566%, shared hit rate70.5882%, and all64 greedy continuation tokens agree (`prefix_cache_check.json`).

The serial prefill burst is128input/32output,32requests/C32, cold cache: TTFT P50/P99 1416.885/2638.772ms, output throughput283.738tokens/s. `vllm_prefill_burst_serial_benchmark.json` is the retained serial reference. Dense batched prefill is not implemented; its equality tests and ratio gate belong to stage10 under the explicit `prefill-features.md` stage split. No artificial off/on ratio is reported.

## Qualitative verdict and limitations

All six greedy and six sampled chat responses were read in full: coherent, on topic, no mechanical repetition, gibberish, wrong-language drift or request contamination in the latest output set. HF and standalone controls use the pinned chat template; all six served prompt tokenizations match. Long explanations reach the256-token cap, and thermodynamics wording is oversimplified in both controls. Evidence: `vllm_qualitative_outputs.json`, `qualitative_review.json`, `qualitative_prompt_format.json`, `controls/qualitative_hf_review.json` and `tt_qualitative_control/` under readiness_vllm.

An earlier sampled thermodynamics heading contained Japanese “英語”. Exact-prefix standalone selected-policy logits reproduce the preference and match all20 served relative candidate gaps within3e-8; HF prefers English. This is a retained selected-policy numerical/model-quality limitation, not an HF behavior or a serving-only defect (`qualitative_language_comparison.json`).

No profiling or reset was performed. Completed server runtime audits record source hashes, sampling modes and exited processes. The final sampling server exited cleanly: runner108112, API108144 and EngineCore108190 are gone, and no process holds a Tenstorrent device. `final_full/runtime_audit.json` records final server lifetime sampling modes separately from benchmark counts. Independent stage review returned **clean-pass** (`stage_review.md`). Local checkpoint SHAs are recorded in `work_log.md`; nothing is pushed.

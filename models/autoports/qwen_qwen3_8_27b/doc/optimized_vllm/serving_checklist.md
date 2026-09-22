# Serving optimization evidence

This stage optimizes the completed vLLM adapter with the selected datatype
configuration. It preserves the decoder, LM head geometry, residual layout and
collective policy. Paths below are relative to this directory.

| Relevant optimize item | Implementation and evidence |
| --- | --- |
| Optimize the measured serving path | `../../tt/generator_vllm.py` calls the generator-owned `serving_prefill_tokens`; the installed `readiness_check.run_vllm_server` launches the actual TT plugin. Matched primary and CI commands and raw results are indexed in README/work_log. |
| Preserve selected optimized math | `before/source_manifest.json` and `after/source_hashes.json` identify unchanged model/decoder source. `../datatype_sweep/selected_confirmation.json` runtime tensors and `selected_token_out.json` verify the inherited selected uploaded BFP4/LoFi projections/head, BF16 activations/residual/CCL, BFP8 KV and FP32 recurrence. Native server startup prints the selected policy. |
| Matmul geometry and topology | Inherited completed searches: `../optimized_full_model/full_path_checklist.md`, `../optimized_multichip_decoder/collective_contracts.md`, `cumulative_contract.md`, `candidate_summary.csv`, and datatype-stage head policy. Stage8 BFP4/LoFi head supersedes stage7's BFP8/HiFi2 head. No new geometry, dtype or residual-topology rejection is inferred from this serving experiment. Decode already matches the selected full-model token-out path; this stage targets prefill dispatch. |
| Entry, final norm, logits and sampling | Preserve full-model embedding, sharded final norm and vocab-sharded head. Common split sampling uses physical TopK32/candidate gather with semantic greedy k1,p0,T1. `AUTODEBUG_prefill.md` and `AUTOFIX_prefill.md` explain first-token sampling ownership. No adapter argmax or replacement generic sampler. |
| Traced prefill and token-out decode | One bounded owned prefill shape through 4096; coordinated model/decode-sampler/prefill/prefill-sampler traces. Long, multirow and nonzero-slot prefills use the existing chunked model followed by persistent staged sampling. Warmed sampling replays in both branches; cold first-use sampling warms its graph once. All serving replays use `blocking=False`. |
| Persistent inputs and capture ownership | Token, position, RoPE, page table, cache, sampler parameters/seeds and logits inputs are persistent. New persistent allocations precede capture. Packing temporaries die before fallback sampling replay. `prefill_full.json`, `fallback_reduced.json`, `test_serving_prefill_trace_host.py`, and allocation tracking validate exact outputs and stable buffer/trace identities. |
| Scheduler refresh / stale state | `adapter_final.json` tests68-step device/control paths, page growth, changed current position/token, remapping, deferred reads and inactive slots. `prefill_full.json` covers changed prompt and reversed physical mapping; unchanged mapping causes zero copies. Device token/position/RoPE/seed feedback avoids per-token upload. |
| Async plugin contract | Native server logs enable async scheduling. `decode_forward(read_from_device=False)` returns device tensors; `read_decode_output(async_read=True)` enqueues one replica's32 UINT32 tokens and an event. `process_decode_output_host` follows event completion. Host tests exercise the actual plugin split and pending snapshots; reset drains only when scheduler state requires it. |
| Context and non-aligned support | Context262144, block32 and external vLLM pool remain unchanged. Full 64-layer exact prefill comparisons cover31,33,128,129,4095,4096,4097. Reduced complementary tests include32,127 and multirow31/45 in slots1/3. The4096 threshold limits trace retention, not valid requests. `../context_contract.json` records at most8,421,376 additional persistent bytes/device and no added KV pool. |
| Async CCL safety | Separate `watcher_prefill.json/.log`: real layers0/3, B4, S33/4097 and multirow31/45, watcher10, NOINLINE1/fabricO3 and allocation tracking. All four devices checked, no ETH disablement, clean exit. Full 64-layer uninstrumented metrics are separate. |
| Runtime data movement | No new decode reshard, tilize/untilize, tensor upload or host logits read. Fallback prefill needs one logits staging copy per scheduler prefill, never per decode token. Native counters record minimal token reads and no full-logits reads. Explicit compatibility sampling is reported separately from native serving. |
| Appropriate verification | Python-only source changes; no C++/CMake build needed. Final59 host tests pass, including the updated fake generator interface in seed-continuity tests. Full-model boundary checks and adapter device check exit0. Formatting/compilation and final HTTP gates are recorded in work_log. |
| Profiler exclusion | No Tracy, tt-perf-report, adapter/live-server device profiler, or ReadDeviceProfiler collected. Device time, DRAM utilization and roofline are unreported for this stage; benchmark wall metrics and correctness evidence support the result. Inherited profile artifacts explain existing math decisions only. |
| Inapplicable items | No MoE/router/sparse matmul, new precision sweep, new C++ kernel, activation policy or CCL family. These are not introduced as serving-stage experiments. |

The README and independent stage review record final HTTP sampling, qualitative,
benchmark and cleanup gates. This checklist does not substitute for that review.

Final gates:73 compatibility tests,12 shared/8 extended native outputs,12 exact
concurrency streams,6 seeded continuations,12 lifecycle requests, both matched
benchmark profiles, process cleanup and bounded four-chip health listing pass.
Independent `stage_review.md` returns clean-pass with no required work.

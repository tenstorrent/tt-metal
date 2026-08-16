# Work log

Started from datatype-sweep commit `a60cab960bc02bba4b9f5c846a7819ee25a42c41`.

- Added `tt/generator_vllm.py`, external vLLM-owned hybrid KV-cache plumbing, async traced decode, device sampling delegation, precision-config loading, and exact context-contract capability.
- Registered Gemma 4 aliases in `vllm_tt_plugin.platform.register_tt_models()`.
- Added canonical generator hooks and propagated absolute/chunked prefill positions through the model.
- Proved non-aligned prompt lengths 47 and 2051 via direct serving requests.
- Ran the full shared sampling profile: 72 passed, 1 skipped. Corrected a model-agnostic repetition-penalty test whose factor 2 did not change Gemma's greedy winner; factor 10 validates the intended behavior.
- AutoDebug/AutoFix found two burst failures: cumulative prefill ends were treated as chunk lengths, and narrowed sliding-cache page tables violated the 1024-token modulo geometry. The adapter now slices `[start:end]`, preserves absolute positions, and passes the full scheduler-owned table.
- Initial raw-completion qualitative prompts bypassed Gemma's chat template and produced base-autocomplete contamination. The readiness runner now uses OpenAI chat completions. Final six-prompt greedy/sampled output is coherent; degeneracy check reports none.
- Reordered page-table staging so decode traces are released before any staging allocation. Final server log has no unsafe-active-trace warning.
- Final runner command: `python -m models.common.readiness_check.run_vllm_server --stages serve,benchmark --model-dir models/autoports/google_gemma_4_26b_a4b_it --hf-model google/gemma-4-26B-A4B-it --mesh-device P300x2 --max-num-seqs 32 --max-model-len 262144 --sampling-profile full --tt-config '{"trace_region_size": 220000000, "fabric_config": "FABRIC_1D_RING"}' --benchmark-prompt-len 128 --benchmark-output-len 128 --benchmark-num-requests 1 --benchmark-concurrency 1 --ci-benchmark-prompt-len 100 --ci-benchmark-output-len 100 --ci-benchmark-num-requests 32`.
- Stage review found padded-batch decode, visible duplicated subwords, incomplete performance comparison, context-capacity ambiguity, and weak fallback evidence. AutoFix showed the padded 32-row execution was the common correctness/performance cause. Logical batch slicing raised serving decode from 2.45 to 21.78 t/s/u and removed the corrupt joins; full-model canonical token-out is 23.76 t/s/u.
- Final primary metrics: TTFT P50/P99 532.8/532.8 ms, TPOT mean/P99 45.92/45.92 ms, ITL P50/P99 44.08/45.66 ms, 21.78 TPOT-derived t/s/u, 20.11 aggregate output tok/s (128/128/1, concurrency 1).
- Final CI metrics: TTFT P50/P99 3949.9/5956.4 ms, TPOT mean/P99 418.9/450.7 ms, ITL P50/P99 369.2/1520.8 ms, 69.6 aggregate output tok/s (100/100/32 burst).
- Final qualitative and benchmarks ran with fallback exceptions enabled. A separate live 262,143-input/1-output request completed in 317.6 s, directly proving the 262,144 context contract.
- Final sampling rerun exposed and fixed padded token slicing plus unsupported arbitrary multi-user shard shapes. Batch one executes logically as one; multi-user requests retain the canonical 32 sampling lanes with inactive rows gated by `current_pos=-1`. Final full profile: 72 passed, 1 skipped in 732.98 s.
- Focused async overlap command: `python models/autoports/google_gemma_4_26b_a4b_it/tests/run_vllm_async_overlap.py --model google/gemma-4-26B-A4B-it --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/async_overlap_state_test.json`. The final chat-templated controls are nondegenerate: the coherent long response produces 96 tokens and crosses the 64-token page boundary, the exact two-word response reaches EOS after 3 tokens while the long request remains active, and both staggered outputs byte-match isolated controls. The retained artifact records empty degeneracy findings for isolated and overlapped text; the runner explicitly fails doubled tokens, adjacent repeated phrases, and dominant-token collapse.

Stage review and local commit SHAs are appended after the independent clean-pass.

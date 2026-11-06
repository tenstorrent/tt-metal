# Gemma 4 31B text autoport

**vLLM serving,4096 input /128 generated /batch1 /concurrency1: TTFT754.43 ms; decode35.32 tokens/s/user.** This is full 60-layer TT plugin serving through this run's autoport, with context 262144, mesh1x4, selected head_bfp4 precision, prefix caching and async device sampling. One measured request per arm; decode is1000/meanTPOT. All required serving profiles pass; independent stage review returned `clean-pass` ([report](doc/optimized_vllm/stage_review.md)).

Short batched prefill reduces cold128-input/32-output,C32 meanTTFT from 1981.70 ms to 857.83 ms while preserving the128/128,B1/C1 latency gate. Secondary local CI-shaped100/100/32 capacity throughput improves from 551.30 to 741.36outputtokens/s; it is not headline decode speed or a remoteCI result.

HF/tokenizer revision:`842da3794eaa0b77d5f08bae87a17459d91ff475`. This is a new text-only implementation built from the HF reference and generic TTNN operations. Non-aligned prompts, chunked continuation, shared vLLM KV ownership and allocator-driven page growth remain supported.

The operator-authorized served default is top_k=32, retaining checkpoint temperature 1/top_p0.95. Stock intermediate-containing prefill and explicitly requested host-only features use documented host compatibility; measured decode and eligible all-final prefill stay on the canonical traced device sampler. Qualitative controls retain the selected-policy prose/language and256-token output-cap limitations described in the report.

[Optimized vLLM report](doc/optimized_vllm/README.md), [metrics](doc/optimized_vllm/perf_summary.json), [work log](doc/optimized_vllm/work_log.md), [context contract](doc/context_contract.json), [serving contract](doc/full_model/serving_contract.json), [datatype selection](doc/datatype_sweep/README.md), [historical full-model report](doc/full_model/README.md). No serving profiler collection or pushes.

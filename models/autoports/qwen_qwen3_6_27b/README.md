# Qwen/Qwen3.6-27B TTNN autoport

Real vLLM TT-plugin serving on Blackhole TP4 measures **291.277 ms primary TTFT and 37.625576 decode tokens/s/user** for 128 input / 128 output tokens, one request at concurrency 1, greedy sampling. Context remains 262144, with max_num_seqs 32, scheduler budget 512, async scheduling and device split sampling. The same-config baseline is 276.979 ms and 37.632082 tokens/s/user: decode is essentially unchanged, while this final TTFT observation is 5.162% higher. The retained continuation optimization improves the measured mixed workload with documented latency tradeoffs.

See the [optimized serving report](doc/optimized_vllm/README.md) for primary and secondary capacity measurements, fresh full-model comparison, sampling, trace/async and eval evidence, limitations and local checkpoints. Sampling, qualitative, trace/async, long-context timing, 32-user long-answer load and cleanup checks are complete. Independent [stage review](doc/optimized_vllm/stage_review_round1.md) returned **clean-pass**. Local checkpoints are recorded in [commits.json](doc/optimized_vllm/commits.json); nothing is pushed.

## Selected datatype baseline

Selected datatype policy on Blackhole TP4: **warmed batch-1 TTFT75.38 ms; traced resident token-out decode37.82 tokens/s/user (26.44 ms/token)**. Workload:128 input /128 output tokens,262144-token context allocation, all64 real layers, device sampling/feedback with no per-token host readback.

The default uses inner-layer MLP BFP4/LoFi with first/last MLP exceptions, GDN attention and head BFP8/HiFi2, full attention BFP8/LoFi, BF16 activations/CCL and BFP8 KV. Across25 evaluated policies it is fastest while passing full-model and P1 gates. Traced teacher-forcing throughput is37.65 t/s/u; worst-reference top1/top5/top100 is91.796875%/98.828125%/100% against90%/98%/100% thresholds. These teacher-forcing numbers are separate from token-out performance.

Selected-default token feedback, non-aligned prompts through32769tokens, trace allocation tracking, resumed prefill and the shared chat suite pass. Advertised context remains262144. See the [datatype sweep report](doc/datatype_sweep/README.md), [selected precision artifact](doc/datatype_sweep/selected_precision_config.json), and [work log](doc/datatype_sweep/work_log.md). Independent [datatype stage review](doc/datatype_sweep/stage_review_round1.md) returned **clean-pass**; [local checkpoints](doc/datatype_sweep/commit_receipt.json) record the committed stage. The completed integration is documented in the [vLLM integration report](doc/vllm_integration/README.md).

## Prior optimized full-model baseline

Optimized full model, Blackhole TP4: **warmed batch-1 TTFT 86.85 ms; traced resident token-out decode 31.00 tokens/s/user (32.26 ms/token)**, with device sampling and feedback and no per-token host readback. Workload: 128 input / 128 output tokens, 262144-token context allocation, all 64 real layers.

The same readback-inclusive harness measures **TTFT 87.031 → 86.889 ms** and **decode 30.976 → 30.957 tokens/s/user** before/after. These are small, near-noise changes. Separate AIME24 teacher-forcing decode is **31.04 tokens/s/user**, including reference-token updates and readback; logits-only is 31.504 steps/s and is not autoregression.

All 20 acceptance jobs pass: AIME24 prefill and teacher-forcing top-5/top-100 are 100/100; nonaligned 32769-token context smoke, all-layer B32, mixed prompts, resumed state, trace tracking, Watcher, runtime boundary, and generated-text checks pass. The context contract remains 262144, with unchanged cache precision/layout and decoder policy. Independent stage review returned [clean-pass](doc/optimized_full_model/stage_review_round1.md). Local checkpoint SHAs are recorded in [commit_receipt.json](doc/optimized_full_model/commit_receipt.json).

See the [optimized full-model report](doc/optimized_full_model/README.md), [work log](doc/optimized_full_model/work_log.md), [hard-gate receipts](doc/optimized_full_model/hard_gates.json), and [accepted optimized decoder](doc/optimized_multichip_decoder/README.md). The completed integration is documented in the [vLLM integration report](doc/vllm_integration/README.md).

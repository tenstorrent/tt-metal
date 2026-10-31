# Optimized full model

Warmed full60-layer autoregressive generation on TP4 Blackhole: **4096 input tokens,128 generated tokens**. Batch1 is one concurrent request; other rows have one request per batch slot. Three repeats, diagnostics disabled, exact token equality against streaming.

| Batch | Before TTFT ms | After TTFT ms | Before tokens/s/user | After tokens/s/user | Decode change |
|---:|---:|---:|---:|---:|---:|
| 1 | 659.37 | 660.79 | 34.215 | 34.291 | +0.222% |
| 32 | 20677.09 | 20670.45 | 23.182 | 23.242 | +0.260% |
| 16 | 10327.03 | 10322.35 | 29.133 | 29.213 | +0.275% |
| 8 | 5162.31 | 5159.80 | 32.308 | 32.430 | +0.377% |

The throughput improvement is small. The main orchestration change removes per-token host reads: each measured request has127 nonblocking model replays,128 sampling replays, one initial position update, zero token/page refreshes and zero captures. TTFT includes reset, all prefill chunks, terminal work, sampling and the first-token read; decode includes one final history read. Baseline has a token read each step. See performance_b*.json and ../full_model/performance_b*.json.

Teacher-forcing accuracy timing is separate: the161-token AIME prompt with100 checked continuation positions reports37.460 tokens/s/user for99 decode steps, including sampling and reference-token feedback. It is not the4096/128 autoregressive result. Prefill top1/top5/top100 are94/100/100%; decode93/100/100%, unchanged from baseline. This is100 token positions from one prompt, not full-dataset accuracy.

## Delivered path

Model google/gemma-4-31B-it and tokenizer remain pinned to842da3794eaa0b77d5f08bae87a17459d91ff475. The completed TP4 model supplies the baseline. Decoder BFP4/LoFi weights, BFP8 KV/selected activation/CCL policy and replicated mesh residual with local8-core L1 sharding are unchanged. No inter-layer collectives were added. The inherited rejection ledger remains in ../optimized_multichip_decoder and ../multichip_decoder.

The vocabulary-sharded BFP8/HiFi2 head now uses eight8192-column DRAM-sharded weight splits per chip, two readers per bank and tuned phase-specific matmul configs. Larger isolated winners failed integrated L1 capacity; the safe8K choice and exact all-logit boundary comparisons are documented in AUTODEBUG.md, AUTOFIX.md, lm_head_candidates.json and head_boundaries.json. The targeted LoFi head control gives no meaningful speed benefit, so existing fidelity remains.

The common split sampler retains local physicaltop32 and semantic greedyk1, candidate-only collectives, top-k/top-p controls and persistent tt_out_tok feedback. Device code advances positions, RoPE lookup, seeds and history index. The model, streaming sampler and buffered sampler traces are all prepared before capture. Fixed-step generate/generate_batch calls use buffering by default; EOS-sensitive calls retain streaming. See runtime_audit.md.

## Capability and evidence

Full-stack262143-token nonaligned and262144-token exact context execution pass, including decode at position262143. The advertised262144 context is unchanged. Logical tails are padded internally to32 and prefill chunks remain1024. The default1024-token output-history buffer is configurable before capture; longer generation retains streaming capability. Fixed slots, mixed prompts, inactive rows, page remapping and seeded trace reuse are checked by generator_state_b*.json.

The six shared qualitative streamed outputs equal the accepted previous-stage outputs. Buffered prefixes match them. Post-EOS forced continuation in the translation stress is explicitly classified in qualitative_verdict.md; it is not clean conversational output. Readiness/reference metadata and autoregressive output paths are retained locally.

## Optimization accounting

See layer_stack_budget.json, optimization_audit.md and perf_summary.json. The selected stage05 clean traced layer sum is26.104ms for B1 (50sliding+10full), versus29.162ms complete token-out decode. Clean terminal and sampler controls add1.211ms and0.482ms, giving27.797ms; complete token-out latency is4.91% above this engineering budget. Other batches are under7% above the layer sum even before terminal credit. Thus no batch exceeds the10–15% closure threshold. These cross-run controls are not exact same-run attribution. Fresh reduced attribution is recorded separately. Profiler-instrumented layer sums are a different regime and are not subtracted from clean host timings.

Full60-layer profiling is prohibited for this run. Device-time/roofline fields for the full model remain unknown. Reduced real layers0/5 profile windows include terminal, sampling and token feedback for attribution only; their device times are not full-model measurements. The analytic lower bound is one projection-weight read plus active KV at TP4 theoretical DRAM bandwidth, with the head added; it excludes other traffic, communication and compute.

## Status and reproduction

State, watcher, sampler, memory and reduced-profile checks pass. Independent stage review returned **clean-pass** with no required work; see stage_review.md. Local checkpoint SHAs are recorded in work_log.md. Commands and exact local artifacts are listed in commands.json and work_log.md. No vLLM integration or push.

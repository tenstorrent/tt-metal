# Stage 10 contract

Optimize the completed google/gemma-4-31B-it vLLM integration in place, preserving
selected datatype-sweep policy, text-only scope, HF/tokenizer revision
842da3794eaa0b77d5f08bae87a17459d91ff475 and context262144. Use this new autoport;
never copy the removed Gemma implementations or Gemma-specific framework branches.

The measured path must be real vLLM TT plugin serving through this run's
`tt/generator_vllm.py`. Headline workload is exactly4096 input,128 generated,
batch1 and concurrency1. Compare warmed before/after TTFT, TPOT, ITL, aggregate
throughput and TPOT-derived decode tokens/s/user with the same runner, generation
mode, max-num-seqs, max-model-len, mesh, sampling and TT configuration. Compare
local CI-shaped serving-burst profiles separately as capacity/nightly-parity
measurements. Do not use burst TPOT as headline decode speed. Decode must be
comparable to the selected full-model traced token-out autoregressive path.

Preserve arbitrary non-aligned logical prompts, prefix caching, chunked prefill,
scheduler freedom and context. Batched prefill must pass exact solo/batched
logits/KV, watcher, allocation tracking, burst and single-user regression gates.
Repeat mixed-shape trace reuse with chunked-prefill continuations, concurrent
decode, allocator page growth, row retirement/reordering, sampling switches,
latency, stable trace IDs and capture/retirement counts. Fixed-shape warm tests
cannot replace it.

Async decode returns device tensors with read_from_device=False, calls
execute_trace(blocking=False), uses minimal deferred reads after that boundary,
and is exercised by vLLM async scheduling before supports_async_decode=True is
accepted. Token, position, RoPE, page, cache and sampler inputs are persistent;
refresh only for scheduler changes. Check stale host tokens/current positions,
changed and unchanged page tables and token feedback. Reuse canonical greedy
split sampling with device token feedback. No host argmax, full-logit readback,
eager/generic greedy sampling or force-argmax in the measured device path.

Operator decisions in AGENTS.md allow the stock plugin's intermediate-containing
prefill host-sampling compatibility mode, which must be logged per step, counted
and documented. Decode and eligible all-final prefill remain device sampled.
Serve with the authorized top_k32 generation override, checkpoint temperature1
and top_p0.95; every server command records it. Plugin runtime stays stock at the
pinned revision. Existing authorized test-only local plugin changes are retained;
any further changes must meet vllm-integration plugin-change rules.

Full sampling tests, prompt-correct shared qualitative outputs, primary
single-user benchmarks, CI-shaped burst measurements and default batch1/8/16/32
serving checks must pass. Record agentic growing-prefix reuse, prefill rates by
length, decode during long prefill, and host contention. Clean up all owned
server/EngineCore/stress processes. No Tracy, tt-perf-report, live-device/adapter
profiling or ReadDeviceProfiler; use serving JSON and functional evidence.

Complete relevant optimize checklist evidence, runtime/memory/context contracts,
README/work log, before/after metrics, full-model comparison, trace/async
choices, rejected candidates, limitations, exact commands and artifact references.
Both model and optimized_vllm READMEs lead with primary single-user vLLM metrics
and put workload shapes next to numbers. Obtain fresh independent stage-review
clean-pass, fixing and rereviewing findings. Commit stage-owned source/docs changes
locally and log SHAs. Never push or send messages. Finish the exact-ID compact
telemetry packet, leaving unavailable device/roofline values null with reasons.

Failed gates and review findings remain work. Only failed AutoFix or an
unauthorized required plugin change permits stopping incomplete.

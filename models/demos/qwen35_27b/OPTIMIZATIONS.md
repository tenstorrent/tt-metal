# Qwen3.5-27B on 4×P150 Blackhole — Optimization Analysis

This document catalogs optimization opportunities for the Qwen3.5-27B model running on 4 P150 Blackhole devices (TP=4). Each item describes the current state, the optimization target, expected impact, and relevant reference implementations.

**Current baseline (from `test_e2e_l1_rolling.py`):** ~68.6ms/step decode, ~14.6 tok/s/user, batch=32

---

## Table of Contents

1. [Prefill Optimizations](#1-prefill-optimizations)
2. [Decode Optimizations](#2-decode-optimizations)
3. [Memory Optimizations](#3-memory-optimizations)
4. [Kernel & Compute Optimizations](#4-kernel--compute-optimizations)
5. [Infrastructure & Testing](#5-infrastructure--testing)
6. [Architecture-Specific (GDN) Optimizations](#6-architecture-specific-gdn-optimizations)
7. [Summary Table](#7-summary-table)

---

## 1. Prefill Optimizations

### 1.1 Parallel Prefill for Full Attention Layers (Critical)

**Current state:** The e2e tests drive prefill through the decode path — processing one token at a time autoregressively for the entire prompt. This means a 4096-token prompt requires 4096 sequential decode steps just for prefill.

**Optimization:** The model already has `forward_prefill()` methods in both `Qwen35Attention` (Flash Attention / SDPA with `is_causal=True`) and `TtGatedDeltaNet` (batched projections). The full attention layers can process the entire sequence in parallel using 2D matmul and Flash Attention. This path exists in the code (`attention.py:forward_prefill`) but is not exercised by the main e2e tests.

**Expected impact:** Orders-of-magnitude improvement in TTFT (time-to-first-token). For a 4096-token prompt, instead of 4096 decode steps, you'd have ~1 prefill pass for attention layers (GDN still sequential — see 1.2).

**Reference:** `llama3_70b_galaxy/tt/generator.py` — `prefill_forward_text()` processes full sequences in parallel with batched SDPA.

**Files to modify:**
- `tt/tests/test_e2e_generate.py` — change prefill loop to call `model.ttnn_prefill_forward()`
- `tests/test_e2e_generate.py` — same change in expanded test
- `tt/tests/test_ttft.py` — benchmark with true prefill path

---

### 1.2 Chunked GDN Prefill with Batched Projections

**Current state:** GDN `forward_prefill()` computes QKV/AB projections in a single batched 2D matmul (good), but the recurrence loop is strictly sequential per-token (inherent to DeltaNet architecture: `state[t]` depends on `state[t-1]`).

**Optimization:** While full sequence parallelism is impossible for the recurrence, several sub-optimizations apply:
- **Batch the pre-recurrence work**: Projections + conv1d can be computed for the full sequence before entering the sequential recurrence loop. The current code already does this for projections but processes conv1d token-by-token.
- **Chunked prefill**: For very long sequences (>4096), process in chunks to avoid memory blowup from storing all intermediate projections. The framework supports this via `max_prefill_chunk_size`.
- **Fused prefill kernel**: Write a custom kernel that fuses the per-token conv1d + recurrence into a single kernel dispatch per token, eliminating Python loop overhead and inter-kernel launch latency.

**Expected impact:** 2-5× improvement in GDN prefill throughput (projections already batched; the loop overhead and kernel dispatch domination can be reduced).

**Reference:** The decode path already uses fused kernels (`gdn_full_fused_inplace`). A similar fused kernel for the sequential prefill recurrence would reduce dispatch overhead.

---

### 1.3 Longer Input Sequence Testing

**Current state:** The primary e2e test (`tt/tests/test_e2e_generate.py`) uses a hardcoded short prompt: `"The capital of France is"` (6 tokens). The expanded test (`tests/test_e2e_generate.py`) supports variable lengths but isn't the default.

**Optimization:** Test and validate with realistic prompt lengths:
- Short: 128 tokens
- Medium: 2048, 4096 tokens
- Long: 8192, 16384, 32768 tokens
- Very long: 65536, 131072 tokens (max supported: 262144)

This exposes performance bottlenecks at different scales and validates correctness at longer contexts.

**Expected impact:** Uncovers scaling issues in KV cache management, GDN state accumulation, memory pressure, and prefill performance.

**Reference:** `llama3_70b_galaxy` tests parametrize across 128, 1K, 4K, 8K, 32K, 64K, 128K tokens with documented TTFT numbers in `PERF.md`.

---

### 1.4 Batched Multi-User Prefill

**Current state:** Prefill processes B=1 (single user), then replicates state to all 32 batch slots via `replicate_kv_cache_to_batch()` and `replicate_prefill_state_to_batch()`.

**Optimization:** For attention layers, support batched prefill where multiple users with similar-length prompts are processed simultaneously. The framework's `prefill_forward_text()` already supports this when `batch_size > 1` and all prompts have the same padded length.

**Constraint:** GDN recurrence states are user-specific and sequential, so multi-user batched GDN prefill may not be straightforward. However, attention layers (16 of 64) can still benefit from batched processing.

**Expected impact:** Reduced aggregate prefill time when serving multiple users with similar prompt lengths.

**Reference:** `llama3_70b_galaxy` supports batched prefill for up to 32 users simultaneously.

---

### 1.5 Prefill Trace Caching

**Current state:** Prefill is not traced. Each prefill call recompiles the computation graph.

**Optimization:** Capture and cache prefill traces for common sequence length buckets (128, 512, 1024, 2048, 4096, ..., 131072). Replay traces during inference to eliminate compilation overhead.

**Expected impact:** Eliminates first-prefill compilation cost on repeated use. Essential for production serving.

**Reference:** `llama3_70b_galaxy/tt/generator.py` — `_capture_trace_prefill()` caches traces keyed by `"{seq_len}_{model_id}_{batch_size}"`.

---

## 2. Decode Optimizations

### 2.1 Speculative Decoding / Multi-Token Prediction (MTP)

**Current state:** Single-token generation per decode step. Each step processes one token through all 64 layers.

**Optimization:** Implement multi-token prediction where a lightweight draft head predicts N future tokens, then verifies them in a single batched pass. This amortizes the per-step overhead across multiple tokens.

**Expected impact:** 1.5-3× throughput improvement depending on accept rate and draft quality.

**Reference:** `deepseek_v3/tt/generator.py` implements MTP with:
- Multiple prediction heads (`nextn_predict_layers`)
- Page table aliasing for cache reuse during verification
- Accept rate tracking for throughput analysis

**Consideration:** GDN recurrence state management during speculative rollback needs careful design — rejected tokens require state restoration.

---

### 2.2 Weight Prefetching

**Current state:** No weight prefetching. Weights are loaded from DRAM cache on-demand during forward pass.

**Optimization:** Implement a weight prefetcher that overlaps DRAM→L1 weight transfers with compute. While layer N is executing, prefetch weights for layer N+1.

**Expected impact:** Hides weight loading latency, particularly beneficial for memory-bound decode operations (M=1 matmuls). Could reduce per-step latency by 10-20%.

**Reference:** `llama3_70b_galaxy/tt/prefetcher_common.py` implements:
- Dedicated prefetcher sub-device with 12 sender cores
- Global circular buffer (728×1088 tiles) for double-buffered weights
- NOC-optimized core selection for bandwidth

---

### 2.3 Continuous Batching

**Current state:** Static batch of 32 sequences all progressing in lockstep. No support for dynamic user arrival/departure.

**Optimization:** Implement continuous batching where:
- New users can join mid-generation (filling empty slots)
- Completed users free their slots immediately
- Different users can be at different generation stages

**Expected impact:** Higher GPU utilization in production serving. Essential for practical deployment.

**Reference:** The vLLM adapter (`generator_vllm.py`) provides the skeleton. The framework generator supports slot-based placement via `empty_slots` parameter.

**Constraint:** GDN state slots need per-user management — when a user completes, their GDN state slot must be reset before reuse.

---

### 2.4 Split Sampling Trace Optimization

**Current state:** `test_e2e_generate_traced.py` captures a single trace for the full decode step including device sampling.

**Optimization:** Split the decode trace into two parts:
1. Model forward (attention + GDN + norms)
2. Sampling (LM head + argmax/top-k)

This allows independent trace management and enables scenarios where sampling parameters change without recompiling the model trace.

**Expected impact:** More flexible trace management; enables dynamic sampling strategy changes.

**Reference:** `llama3_70b_galaxy` uses split sampling by default (`split_sampling=True`), with separate trace capture for model and sampling.

---

## 3. Memory Optimizations

### 3.1 Expand L1 Rolling Window Coverage

**Current state:** `test_e2e_l1_rolling.py` demonstrates rolling-window L1 state for the first 4 GDN layers. The architecture has 48 GDN layers total in groups of 3 (3 GDN + 1 full attention, repeated 16×).

**Optimization:** Extend L1 rolling to cover all 48 GDN layers dynamically:
- Keep 3 GDN layers' rec_states in L1 at a time
- Swap to DRAM when crossing a full-attention barrier
- Pre-allocate DRAM backup buffers to avoid runtime allocation

The infrastructure for this exists in `gdn.py` (`enable_l1_state()`, `_swap_l1_state()`). The optimization is extending it from 4 layers to full model coverage.

**Expected impact:** GDN recurrence kernel runs faster from L1 vs DRAM. The `test_e2e_l1_rolling.py` shows this path works; scaling to all layers should yield consistent decode speedup.

---

### 3.2 Paged Attention for Longer Contexts

**Current state:** Internal per-head KV caches for attention layers. Paged attention infrastructure exists (`use_paged_kv_cache`, `paged_update_cache`) but is primarily exercised through vLLM adapter.

**Optimization:** Enable paged attention by default for all inference paths:
- Reduces memory fragmentation for variable-length sequences
- Enables efficient memory sharing across users with shared prefixes
- Required for scaling to 128K+ context lengths with batch>1

**Expected impact:** Enables larger batch sizes at longer context lengths without OOM.

**Reference:** `llama3_70b_galaxy` uses paged attention by default with `block_size=64`, `max_num_blocks=1024`.

---

### 3.3 FP8 Inference (Beyond Weight Storage)

**Current state:** Weights are stored in FP8 (float8_e4m3fn) with block-wise quantization but are dequantized to bfloat16 at load time. All compute runs in bfloat16.

**Optimization:** Explore FP8 compute for select operations:
- GDN projections (large matmuls that are memory-bandwidth bound in decode)
- MLP gate/up/down projections
- Keep attention QKV and recurrence in bfloat16 for numerical stability

**Expected impact:** ~2× memory bandwidth reduction for FP8 matmuls, translating to faster decode for bandwidth-bound operations.

**Consideration:** Requires hardware FP8 matmul support on Blackhole P150 and careful accuracy validation.

---

## 4. Kernel & Compute Optimizations

### 4.1 Fused Attention Output + All-Reduce

**Current state:** Attention output projection and cross-device all-reduce are separate operations.

**Optimization:** Fuse the output matmul with the subsequent all-reduce/all-gather, eliminating an intermediate buffer write.

**Expected impact:** Reduces memory bandwidth and kernel launch overhead for the TP communication step.

**Reference:** `llama3_70b_galaxy` implements `USE_FUSED_ALL_GATHER_MATMUL` which fuses attention output gather with subsequent matmul on compatible grid configurations.

---

### 4.2 GDN Conv1d Parallelization

**Current state:** 4-tap causal conv1d in GDN is implemented as a shift register with sequential per-token processing.

**Optimization:** For prefill, the conv1d operation over the full sequence can be implemented as a 1D depthwise convolution (parallel across the sequence dimension). Only the recurrence needs to remain sequential.

**Expected impact:** Eliminates conv1d from the sequential prefill loop, leaving only the DeltaNet recurrence as the sequential bottleneck.

---

### 4.3 Optimized QK L2 Normalization

**Current state:** L2 normalization for Q and K in both attention and GDN is computed as: `x / (||x||_2 + eps) * learned_scale`. In GDN, normalization is done before `repeat_interleave` (3× less compute — good optimization already in place).

**Optimization:** Fuse L2 norm with the preceding projection matmul. Instead of separate matmul → L2_norm → scale, output the normalized+scaled result directly.

**Expected impact:** Eliminates a kernel launch and intermediate buffer for each QK normalization.

---

### 4.4 Async All-Reduce with Compute Overlap

**Current state:** All-reduce operations for TP synchronization appear to be synchronous.

**Optimization:** Use asynchronous all-reduce with double-buffering to overlap communication with the next layer's computation.

**Expected impact:** Hides TP communication latency behind compute. More impactful as TP degree increases.

**Reference:** `deepseek_v3` uses async AllGather/ReduceScatter with minimal config for low latency.

---

## 5. Infrastructure & Testing

### 5.1 Performance Documentation (PERF.md)

**Current state:** No performance documentation. Only ad-hoc numbers in test comments (14.6 tok/s/user baseline).

**Optimization:** Create a `PERF.md` file documenting:
- Baseline metrics at various sequence lengths and batch sizes
- TTFT (time-to-first-token) benchmarks
- Decode throughput (tok/s/user and aggregate tok/s)
- Memory usage per configuration
- Regression tracking across commits

**Reference:** `llama3_70b_galaxy/PERF.md` provides a comprehensive performance table with reproducibility instructions.

---

### 5.2 Comprehensive Benchmark Suite

**Current state:** Tests exist for profiling (`test_profile.py`, `test_profile_breakdown.py`) but lack systematic coverage across configurations.

**Optimization:** Create parametrized performance benchmarks covering:
- Batch sizes: 1, 8, 16, 32
- Sequence lengths: 128, 1K, 4K, 8K, 32K, 64K, 128K
- Modes: prefill-only TTFT, decode throughput, end-to-end latency
- Configurations: with/without L1 rolling, with/without device sampling, with/without tracing

**Reference:** `llama3_70b_galaxy/demo/demo_performance.py` provides a systematic performance benchmark framework.

---

### 5.3 CI Performance Regression Tests

**Current state:** No automated performance regression detection.

**Optimization:** Add CI tests that:
- Run decode at fixed configuration (e.g., batch=32, seq=2048)
- Assert throughput meets minimum threshold
- Fail if performance regresses beyond tolerance (e.g., >5%)

**Reference:** `llama3_70b_galaxy/tests/` includes performance-tagged tests with expected throughput ranges.

---

### 5.4 Warmup Infrastructure

**Current state:** No systematic warmup phase. First decode step includes compilation overhead.

**Optimization:** Implement a `warmup_model()` method that:
- Pre-compiles traces for all supported sequence length buckets
- Warms up both prefill and decode paths
- Caches program configs for subsequent fast inference

**Expected impact:** Eliminates cold-start latency in production serving.

**Reference:** `llama3_70b_galaxy/tt/generator.py` — `prefill_warmup()` and `decode_warmup()` systematically warm up all configurations.

---

## 6. Architecture-Specific (GDN) Optimizations

### 6.1 GDN State Compression for Long Contexts

**Current state:** GDN recurrence state is `[B*Nv_TP, Dk, Dv]` = `[32*12, 128, 128]` per layer per device. With 48 GDN layers, total state is ~48 × 32 × 12 × 128 × 128 × 2 bytes ≈ 576 MB per device.

**Optimization:** Explore low-rank approximation of the recurrence state matrix. Since the state is a running sum of outer products (`k ⊗ delta`), it may admit low-rank compression without significant quality loss.

**Expected impact:** Reduced state memory, faster L1↔DRAM swaps, potentially enabling more layers in L1 simultaneously.

**Risk:** May affect model quality — requires careful accuracy evaluation.

---

### 6.2 GDN Chunkwise Parallel Recurrence

**Current state:** GDN recurrence is strictly sequential per-token.

**Optimization:** Implement chunkwise parallel processing for the DeltaNet recurrence:
1. Divide sequence into chunks of C tokens
2. Within each chunk, compute the recurrence with a parallel scan (the decay+update can be formulated as an associative scan)
3. Chain chunks sequentially (chunk boundaries carry forward state)

This trades some extra compute for parallelism within chunks.

**Expected impact:** C× speedup for GDN prefill recurrence (where C is chunk size, e.g., 16-64).

**Reference:** The "Gated Linear Attention" and "Mamba-2" papers describe chunkwise parallel formulations for linear recurrences that could apply to DeltaNet.

**Consideration:** Requires reformulating the decay+update as an associative operator, which is possible for the linear recurrence but needs validation for the specific DeltaNet update rule with sigmoid gating.

---

### 6.3 GDN Prefill State Pipelining

**Current state:** All 48 GDN layers process each token sequentially. For a 4096-token prefill, each GDN layer waits for the previous layer to finish all 4096 tokens.

**Optimization:** Pipeline GDN layers: once layer L finishes token t, layer L+1 can start processing token t immediately (don't wait for layer L to finish all tokens).

**Expected impact:** Overlaps computation across layers, reducing total prefill wall time. Most beneficial when kernel launch overhead is significant relative to per-token compute.

**Constraint:** Requires careful buffer management to avoid data hazards between layers.

---

### 6.4 Conv1d State Optimization

**Current state:** 4-tap causal conv1d uses a shift register of 4 separate tensors, each `[1, B, qkv_dim_tp]`. Shift operation copies tensor 2→3, 1→2, 0→1, new→0.

**Optimization:** Use a circular buffer instead of shifting:
- Single tensor `[4, B, qkv_dim_tp]` with a position pointer
- Write new input at pointer, advance pointer (mod 4)
- Conv taps indexed relative to pointer

**Expected impact:** Eliminates 3 tensor copies per GDN layer per decode step (48 layers × 3 copies = 144 avoided copies per step).

---

## 7. Summary Table

| # | Optimization | Category | Impact | Complexity | Dependencies |
|---|---|---|---|---|---|
| 1.1 | Parallel prefill (attention) | Prefill | **Critical** — orders of magnitude TTFT improvement | Low | Code path exists, needs test integration |
| 1.2 | Chunked GDN prefill | Prefill | High — 2-5× GDN prefill speedup | Medium | Fused kernel for sequential loop |
| 1.3 | Longer input testing | Prefill | Medium — exposes scaling issues | Low | Test parametrization |
| 1.4 | Batched multi-user prefill | Prefill | Medium — aggregate prefill throughput | Medium | Framework support exists |
| 1.5 | Prefill trace caching | Prefill | Medium — eliminates recompilation | Medium | Framework infrastructure exists |
| 2.1 | Speculative decoding / MTP | Decode | High — 1.5-3× throughput | High | GDN state rollback design |
| 2.2 | Weight prefetching | Decode | Medium — 10-20% latency reduction | High | New prefetcher infrastructure |
| 2.3 | Continuous batching | Decode | High — production serving essential | High | GDN per-user state management |
| 2.4 | Split sampling traces | Decode | Low — flexibility improvement | Low | Trace infrastructure change |
| 3.1 | Full L1 rolling window | Memory | Medium — consistent decode speedup | Low | Infrastructure exists, extend coverage |
| 3.2 | Paged attention default | Memory | Medium — longer context support | Low | Infrastructure exists |
| 3.3 | FP8 compute | Memory | High — 2× bandwidth for select ops | High | Hardware support validation |
| 4.1 | Fused output + all-reduce | Kernel | Medium — reduced overhead | Medium | Grid compatibility check |
| 4.2 | Conv1d parallelization | Kernel | Low-Medium — prefill only | Medium | Custom kernel |
| 4.3 | Fused QK L2 norm | Kernel | Low — kernel launch reduction | Medium | Custom fused kernel |
| 4.4 | Async all-reduce | Kernel | Medium — hides comm latency | Medium | CCL infrastructure |
| 5.1 | PERF.md documentation | Infra | N/A — essential for tracking | Low | Benchmark runs |
| 5.2 | Benchmark suite | Infra | N/A — essential for optimization | Low | Test framework |
| 5.3 | CI regression tests | Infra | N/A — prevents regressions | Low | CI integration |
| 5.4 | Warmup infrastructure | Infra | Medium — production readiness | Medium | Trace infrastructure |
| 6.1 | GDN state compression | GDN | Medium — memory reduction | High | Accuracy evaluation |
| 6.2 | Chunkwise parallel recurrence | GDN | **High** — C× prefill speedup | High | Associative scan formulation |
| 6.3 | GDN prefill pipelining | GDN | Medium — layer overlap | High | Buffer management |
| 6.4 | Conv1d circular buffer | GDN | Low — eliminates 144 copies/step | Low | Buffer pointer management |

---

## Recommended Priority Order

### Phase 1: Quick Wins (Low complexity, high impact)
1. **1.1** — Enable parallel prefill for attention layers (path exists, just needs wiring)
2. **1.3** — Test with longer input sequences (parametrize existing tests)
3. **3.1** — Extend L1 rolling window to all 48 GDN layers
4. **5.1** — Create PERF.md with baseline numbers
5. **6.4** — Conv1d circular buffer (simple change, removes 144 copies/step)

### Phase 2: Medium-Term (Moderate complexity, significant impact)
6. **1.2** — Chunked GDN prefill with fused sequential kernel
7. **1.5** — Prefill trace caching
8. **5.4** — Warmup infrastructure
9. **4.4** — Async all-reduce with compute overlap
10. **3.2** — Enable paged attention by default

### Phase 3: Major Features (High complexity, transformative impact)
11. **6.2** — Chunkwise parallel GDN recurrence (requires algorithm R&D)
12. **2.1** — Speculative decoding / MTP
13. **2.3** — Continuous batching
14. **2.2** — Weight prefetching
15. **3.3** — FP8 compute paths

---

## Reference Implementations

| Model | Path | Key Features to Study |
|---|---|---|
| Llama-3.3-70B Galaxy | `models/demos/llama3_70b_galaxy/` | Batched prefill, prefix caching, weight prefetching, PERF.md, 128K context |
| DeepSeek-V3 | `models/demos/deepseek_v3/` | MTP/speculative decoding, MoE routing, async CCL, multi-host mesh |
| Framework Generator | `models/tt_transformers/tt/generator.py` | Trace infrastructure, batched prefill, split sampling, warmup |

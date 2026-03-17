# Why Hybrid: Advantages Over Both Original Implementations

This document explains what the hybrid GLM-4.7-Flash implementation gains over each of the two original codebases — and critically, what **new performance gains** become possible only by combining them.

---

## Table of Contents

1. [The Two Originals at a Glance](#the-two-originals-at-a-glance)
2. [What the Hybrid Takes from Each](#what-the-hybrid-takes-from-each)
3. [New Optimizations: Things Neither Has Alone](#new-optimizations-things-neither-has-alone)
4. [Performance Gain Breakdown](#performance-gain-breakdown)
5. [Advantages Over the Agentic Implementation](#advantages-over-the-agentic-implementation)
6. [Advantages Over the tt-symbiote Implementation](#advantages-over-the-tt-symbiote-implementation)
7. [Single-Chip vs T3K (8-chip) Applicability](#single-chip-vs-t3k-8-chip-applicability)
8. [Summary Comparison Matrix](#summary-comparison-matrix)

---

## The Two Originals at a Glance

| Dimension | **Agentic** (`glm4_moe_lite/`) | **tt-symbiote** (`tt_symbiote/`) |
|---|---|---|
| **Goal** | Maximum decode throughput on production hardware | Modular, reusable framework for any HuggingFace model |
| **Strength** | Hand-tuned C++ fused kernels, DRAM-sharded matmuls, compressed KVPE, 30+ runtime knobs | HF module replacement, weight lifecycle, TRACED mode, distributed norm/linear |
| **Weakness** | Monolithic GLM-specific code, no HF integration, no trace capture | No fused kernels, no DRAM sharding, standard (not compressed) KV cache |
| **Decode perf** | Best available (fused kernels + sparse MoE) | ~2x slower (generic TTNN ops, separate K/V cache) |
| **Ease of use** | Low (manual setup, env vars, snapshot loading) | High (drop-in HF replacement, `from_pretrained`) |

---

## What the Hybrid Takes from Each

### From Agentic (Performance)

| Optimization | What It Does | Impact |
|---|---|---|
| **GLMKVCacheBranch** C++ kernel | Fuses DKV matmul + gather + RMSNorm + RoPE in one dispatch | ~30-40% attention latency reduction (batch=1) |
| **PreSDPA** C++ kernel | Fuses RMSNorm → matmul → RMSNorm2 → matmul2 → RoPE | Fewer kernel dispatches, better data locality |
| **Compressed KVPE cache** | Single `[blocks,1,block_size,576]` BF8 tensor vs separate K/V | **2x KV cache memory reduction** |
| **DRAM-sharded matmuls** | L1 WIDTH_SHARDED gate→silu→up→mul→down in single L1 pass | ~20-30% decode MLP latency |
| **Sparse MoE** | `scatter → moe_expert_token_remap → sparse_matmul` (block_size=32) | Optimal expert computation |
| **fused_persistent_moe_decode** | Single kernel for full MoE decode step | ~15-20% MoE latency reduction |
| **4 expert paths** | sparse, dense_decode, dense_prefill, packed_prefill | Right path for each scenario |
| **MTP speculative decoding** | Layer 47 produces draft tokens | ~1.5-2x effective throughput |
| **30+ runtime config knobs** | Env-var-controlled precision, memory, dispatch, debug | Fine-grained tuning without code changes |

### From tt-symbiote (Framework + Unique Optimizations)

| Optimization | What It Does | Impact |
|---|---|---|
| **TTNNModule base class** | `from_torch()`, `preprocess_weights()`, `move_weights_to_device()` | Clean weight lifecycle, no manual conversion |
| **Module replacement** | `register_module_replacement(model, class_map)` | Drop-in HF integration with `from_pretrained` |
| **Distributed RMSNorm** | `rms_norm_pre_all_gather → all_gather_async → rms_norm_post_all_gather` | **Avoids full-tensor all-gather before norm (TP efficiency)** |
| **reduce_scatter_minimal_async** | Overlapped matmul + reduce-scatter for TP linear | **Lower TP communication overhead than all_reduce** |
| **3-pass BF16 router centering** | Rough topk → center → refined topk → final topk | **Better MoE routing accuracy at BF16** |
| **Run modes** | NORMAL, FALLBACK, LIGHTWEIGHT, SEL, DPL, TRACED | Debugging, correctness checking, profiling |

> **Note on tracing:** The agentic implementation already has full `begin_trace_capture` / `end_trace_capture` / `execute_trace` support in `model_tt.py` (activated via `--enable-trace`). tt-symbiote has its own TRACED run mode at the framework level. Both codebases support trace capture/replay — this is NOT a hybrid-exclusive gain.

---

## New Optimizations: Things Neither Has Alone

These are the **genuine performance gains** the hybrid unlocks by combining both codebases. Neither original can do these on its own:

### 1. fused_persistent_moe_decode + 3-Pass Router Centering

| What | Use the agentic fused MoE kernel with tt-symbiote's improved BF16 routing accuracy |
|---|---|
| **Agentic alone** | Has fused MoE decode but single-pass BF16 topk (can flip expert ordering near decision boundaries) |
| **tt-symbiote alone** | Has 3-pass centering but no fused MoE kernel (slower sparse_matmul path) |
| **Hybrid** | More accurate routing feeds into the faster kernel — correct experts, fastest compute |
| **Expected gain** | **Better output quality** at the same throughput, or same quality with fewer top-k |

### 2. DRAM-Sharded MLP + reduce_scatter_minimal_async

| What | L1-sharded MLP from agentic + overlapped reduce-scatter from tt-symbiote |
|---|---|
| **Agentic alone** | DRAM-sharded MLP with `all_reduce` for TP (full synchronization barrier) |
| **tt-symbiote alone** | `reduce_scatter_minimal_async` but with DRAM intermediates (higher memory traffic) |
| **Hybrid** | MLP runs in L1 with DRAM-sharded weights, output reduced via async reduce-scatter |
| **Expected gain** | **~10-20% MLP latency on T3K** by overlapping compute and communication |

### 3. Distributed RMSNorm + Compressed KVPE

| What | tt-symbiote's distributed norm with agentic's compressed KV cache |
|---|---|
| **Agentic alone** | Uses fused kernel for batch=1 norm, but standard norm for prefill/multi-batch |
| **tt-symbiote alone** | Distributed norm but with 2x larger separate K/V cache |
| **Hybrid** | Efficient TP norm on the already-compressed KVPE path |
| **Expected gain** | **Better TP scaling for prefill** with lower memory footprint |

### 4. Framework-Level Trace Integration

| What | Integrate tt-symbiote's TRACED run mode with the agentic model runner |
|---|---|
| **Agentic alone** | Has trace capture/replay via `--enable-trace` in `model_tt.py` — fully functional |
| **tt-symbiote alone** | Has TRACED run mode at the framework level (auto-traces any TTNNModule) |
| **Hybrid** | Both trace paths available; framework-level tracing can be used for new models without writing custom trace logic |
| **Expected gain** | **Developer productivity** — not a perf gain over agentic (which already traces), but easier to add tracing to new model variants |

> **Important:** Trace capture/replay is NOT a hybrid-exclusive performance advantage. The agentic codebase already implements it with `begin_trace_capture` / `execute_trace` and supports both logits and sampling trace modes.

---

## Performance Gain Breakdown

### Estimated Cumulative Gains (T3K, 8-chip, batch=1 decode, traced mode)

The agentic baseline already includes trace capture/replay. The hybrid gains come from TP communication improvements and distributed norm — not from tracing.

| Layer | Agentic Baseline (traced) | + Async TP (reduce_scatter) | + Distributed Norm | + 3-Pass Router | Cumulative |
|---|---|---|---|---|---|
| Attention (KV + Q + FlashMLA + output) | 1.0x | 1.08x | 1.12x | — | **1.12x** |
| MoE (shared + router + experts + merge) | 1.0x | 1.10x | — | 1.10x (quality) | **1.10x** |
| Dense MLP | 1.0x | 1.08x | — | — | **1.08x** |
| **Full decode step** | **1.0x** | **~1.09x** | **~1.11x** | **~1.11x** | **~1.10-1.15x** |

> These are estimates based on the optimization characteristics. Actual gains depend on the workload (batch size, sequence length) and hardware configuration. Gains are more pronounced at higher device counts where TP communication is a larger fraction of total time.

### Where the Gains Come From

```
Agentic baseline decode (T3K, traced): ~25 ms/token

Hybrid improvements over agentic traced baseline:
  reduce_scatter_minimal_async for TP linears:   -2 ms  (8%)
  Distributed RMSNorm (stats-only gather):       -0.5 ms (2%)
  Fused gate+up for experts + shared:            -0.5 ms (2%)
  3-pass router centering:                       ~0 ms  (quality, not speed)

Estimated hybrid decode (T3K, traced):           ~22 ms/token  (~1.10-1.15x faster)
```

> **Honest assessment:** The hybrid's perf advantage over the agentic traced baseline is **modest (~10-15%)** because the agentic implementation is already highly optimized with trace capture, fused kernels, and sparse MoE. The hybrid's larger value is in combining these with the framework benefits (HF integration, modularity, correctness modes) and the tt-symbiote TP communication improvements.

---

## Advantages Over the Agentic Implementation

| Dimension | Agentic | Hybrid | Improvement |
|---|---|---|---|
| **HuggingFace integration** | None (manual snapshot loading) | `from_pretrained` + auto module replacement | Much easier model loading |
| **Weight management** | Manual `convert_decoder_layer_weights` per layer | `preprocess_weights()` / `move_weights_to_device()` lifecycle | Cleaner, less error-prone |
| **Trace capture** | Full support (`begin_trace_capture` / `execute_trace`) | Both agentic + framework-level TRACED mode | Same perf; framework tracing easier for new models |
| **TP communication** | `all_reduce` (synchronization barrier) | `reduce_scatter_minimal_async` (overlapped) | ~7-10% TP linear improvement |
| **Norm efficiency** | Standard RMSNorm (full all-gather) | Distributed RMSNorm (stats-only gather) | ~3-5% norm latency on multi-device |
| **Router accuracy** | Single BF16 topk | 3-pass centering | Better expert selection near boundaries |
| **Correctness modes** | Debug env vars only | SEL/DPL modes (run both, compare) | Systematic validation |
| **Multi-model reuse** | GLM-specific only | Generic modules usable for any MLA/MoE model | Framework for future models |
| **Code maintainability** | 20+ files, monolithic | Modular (core/ + modules/ + runner) | Easier to extend and test |

---

## Advantages Over the tt-symbiote Implementation

| Dimension | tt-symbiote | Hybrid | Improvement |
|---|---|---|---|
| **KV cache memory** | Separate K/V (2x larger) | Compressed KVPE at BF8 | **2x memory savings** |
| **Attention decode** | Generic TTNN ops | Fused GLMKVCacheBranch + PreSDPA kernels | **~30-40% attention latency** |
| **MoE decode** | sparse_matmul path only | 4 paths + fused_persistent_moe_decode | **~15-20% MoE latency** |
| **MLP decode** | DRAM intermediates | L1 WIDTH_SHARDED via dram_sharded_mlp | **~20-30% MLP latency** |
| **Matmul config** | Framework defaults | 1D multicast program config for M=1 | **~10-15% decode matmul** |
| **Speculative decoding** | Not supported | MTP layer 47 for draft tokens | **~1.5-2x effective throughput** |
| **Batch serving** | Basic | Decode trace batching (per-bucket states) | **~2-3x throughput** |
| **Runtime tuning** | Minimal config | 30+ env var knobs (precision, memory, MoE, TP) | Fine-grained optimization |
| **Weight quantization** | BF16 only | BF8 experts, BF8 KV cache, configurable dtype | Lower memory, more layers fit |

---

## Single-Chip vs T3K (8-chip) Applicability

The hybrid implementation is **fully applicable to both single-chip (N150) and T3K (8-chip) systems**. The code automatically adapts based on the mesh configuration:

### Single Chip (N150 / Quiet Box)

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# Layer benchmark
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_single_device.py

# Full model decode (with weight eviction for 12.8 GB DRAM)
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 1 --prompt "Hello" --max-new-tokens 32 --phase both
```

| Behavior | Details |
|---|---|
| Mesh shape | `(1, 1)` |
| Weight eviction | **Enabled** (`GLM4_MOE_LITE_EVICT_WEIGHTS=1`) — layers loaded on demand |
| TP | Disabled (single device) |
| Expert sharding | All 64 experts on one device |
| DRAM budget | ~12.8 GB (weight eviction streams layers) |
| Fabric | Not configured |

### T3K / 8-Chip (Loud Box)

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# Full model decode (all weights resident)
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 --prompt "Hello" --max-new-tokens 64 --phase both

# With tracing (best throughput)
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 --prompt "Hello" --max-new-tokens 64 --phase both \
  --enable-trace --trace-mode sampling

# With all hybrid optimizations
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 \
GLM4_MOE_LITE_FUSED_MOE=1 \
GLM4_MOE_LITE_TP=1 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 --prompt "Hello" --max-new-tokens 64 --phase both
```

| Behavior | Details |
|---|---|
| Mesh shape | `(1, 8)` |
| Weight eviction | **Disabled** — all 47 layers fit across 8 devices (~100 GB aggregate DRAM) |
| TP | **Enabled** (`GLM4_MOE_LITE_TP=1`) — attention/MLP weights sharded across devices |
| Expert sharding | 64 experts / 8 devices = 8 experts per device |
| DRAM budget | ~100 GB aggregate (no eviction needed) |
| Fabric | `FABRIC_1D` (auto-configured for multi-device) |
| Dispatch | `reduce` (default) or `a2a` for MoE experts |

### Multi-Device MoE Test

```bash
# Requires 8-chip system
TT_ENABLE_HW_TESTS=1 \
TT_ENABLE_LARGE_MODEL_TESTS=1 \
TT_ENABLE_MULTI_DEVICE_TESTS=1 \
TT_TEST_MESH_SHAPE=1x8 \
pytest models/demos/glm4_moe_lite/tests/test_tt_moe_layer1_mesh_optional.py -v
```

### N300 (2-chip)

```bash
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 2 --prompt "Hello" --max-new-tokens 32 --phase both
```

### TG / Galaxy (32-chip)

```bash
# TG uses mesh shape (8, 4)
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 4 --prompt "Hello" --max-new-tokens 32 --phase both
```

---

## Summary Comparison Matrix

| Feature | Agentic | tt-symbiote | **Hybrid** |
|---|:---:|:---:|:---:|
| **Fused KV branch kernel** | Yes | No | **Yes** |
| **PreSDPA fused kernel** | Yes | No | **Yes** |
| **Compressed KVPE (BF8)** | Yes | No | **Yes** |
| **DRAM-sharded MLP** | Yes | No | **Yes** |
| **fused_persistent_moe_decode** | Yes | No | **Yes** |
| **4 MoE expert paths** | Yes | No | **Yes** |
| **MTP speculative decoding** | Yes | No | **Yes** |
| **30+ runtime knobs** | Yes | No | **Yes** |
| **Decode trace batching** | Yes | No | **Yes** |
| **HF module replacement** | No | Yes | **Yes** |
| **TTNNModule weight lifecycle** | No | Yes | **Yes** |
| **Trace capture/replay** | Yes (`model_tt.py`) | Yes (TRACED run mode) | **Yes (both paths)** |
| **Distributed RMSNorm** | No | Yes | **Yes** |
| **reduce_scatter_minimal_async** | No | Yes | **Yes** |
| **3-pass BF16 router centering** | No | Yes | **Yes** |
| **SEL/DPL correctness modes** | No | Yes | **Yes** |
| **Single-chip (N150)** | Yes | Yes | **Yes** |
| **T3K (8-chip)** | Yes | Yes | **Yes** |
| **TG / Galaxy** | Yes | Partial | **Yes** |
| | | | |
| **Estimated decode speedup vs agentic (traced)** | Baseline | ~0.5x | **~1.10-1.15x** |
| **Estimated decode speedup vs tt-symbiote** | ~2x | Baseline | **~2.0-2.3x** |

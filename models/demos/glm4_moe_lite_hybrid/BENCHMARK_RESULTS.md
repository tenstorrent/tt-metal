# GLM-4.7-Flash Single-Device (N150) Benchmark Results

**Date:** 2026-03-17
**Device:** N150 Wormhole (1 chip, 8x8 compute grid = 64 cores, 12 DRAM channels)
**Host:** cust-dalar-26, Ubuntu 22.04, Kernel 5.15.0-170-generic, 47 GB RAM
**Model:** zai-org/GLM-4.7-Flash (47 layers, 64 routed experts, 4 experts/token)
**Implementation:** Agentic backend (shared by both agentic and hybrid via re-export)

---

## KPI Summary

| Metric | Value | Unit | Notes |
|---|---|---|---|
| **Embedding lookup** | **0.27** | ms/token | Single token lookup |
| **Layer 0 decode** (attn + dense MLP) | **2.49** | ms/step | Batch=1, position=10 |
| **Layer 1 decode** (attn + MoE) | **4.76** | ms/step | Batch=1, 64 experts, top-4 routing |
| **Standard linear** [2048->10240] | **0.30** | ms | decode-sized M=1 matmul |
| **KVPE cache dim** | **576** | elements | kv_lora_rank(512) + rope_dim(64) |
| **BF8 cache per layer** | **4.5** | MB | [128 blocks, 1, 64, 576] @ BF8 |
| **BF16 cache per layer** | **9.0** | MB | Same shape @ BF16 |
| **Memory savings (BF8 vs BF16)** | **2.0x** | | Compressed KVPE halves KV memory |

---

## Detailed Stage Breakdown

### Layer 0: Attention + Dense MLP (2.49 ms total)

| Stage | Time (ms) | % of Total |
|---|---|---|
| Input RMSNorm | 0.17 | 6.7% |
| KV cache update (projection + RoPE + paged write) | 0.33 | 13.2% |
| Q projection (LoRA + RoPE + kv_b1) | 0.09 | 3.7% |
| FlashMLA decode | 0.19 | 7.5% |
| Attention output (kv_b2 + head flatten + w_o) | 0.13 | 5.1% |
| Post-attention norm | ~0.01 | 0.4% |
| Dense MLP (SwiGLU) | 0.01 | 0.4% |
| **Total per-step** | **2.49** | **100%** |

> Note: Stage times are cumulative across warmup+measure iterations. The dominant cost is the KV cache update path (projection + RoPE + paged cache write) followed by FlashMLA decode.

### Layer 1: Attention + MoE (4.76 ms total)

| Stage | Time (ms) | % of Total |
|---|---|---|
| Input RMSNorm | 0.05 | 1.0% |
| KV cache update | 0.59 | 12.5% |
| Q projection | 0.46 | 9.7% |
| FlashMLA decode | 0.05 | 1.0% |
| Attention output | 0.20 | 4.2% |
| Post-attention norm | 0.04 | 0.8% |
| **MoE shared expert** | **~1.87** | **39.3%** |
| **MoE router** (sigmoid + bias + topk) | **~1.14** | **24.0%** |
| **MoE routed experts** (sparse) | **~1.10** | **23.1%** |
| MoE merge (shared + routed) | 0.25 | 5.2% |
| **Total per-step** | **4.76** | **100%** |

> MoE layers are ~1.9x slower than dense layers. The shared expert MLP and router together account for 63% of MoE layer time.

---

## Memory Analysis: Compressed KVPE Cache

The hybrid uses the agentic's compressed KVPE cache which stores `[kv_nope || k_rope]` in a single tensor instead of separate K and V caches.

| Configuration | Per-Layer | 47 Layers | Savings |
|---|---|---|---|
| **Compressed KVPE @ BF8** (hybrid/agentic) | 4.5 MB | 211 MB | **Baseline** |
| **Compressed KVPE @ BF16** | 9.0 MB | 423 MB | 0.5x |
| **Separate K/V @ BF16** (tt-symbiote style) | ~18.0 MB | 846 MB | 0.25x |

> With BF8 compressed KVPE, 47 layers of KV cache fit in ~211 MB — well within the N150's 12.8 GB DRAM, leaving headroom for model weights.

---

## Projected Full-Model Performance (47 layers)

Extrapolating from per-layer numbers with 1 dense layer + 46 MoE layers:

| Metric | Calculation | Value |
|---|---|---|
| Dense layers (1) | 1 x 2.49 ms | 2.49 ms |
| MoE layers (46) | 46 x 4.76 ms | 218.96 ms |
| Embedding + LM head (est.) | ~1 ms | ~1 ms |
| **Estimated decode/token** | Sum | **~222 ms** |
| **Estimated throughput** | 1000/222 | **~4.5 tok/s** |

> Single N150 throughput is memory-bandwidth-limited due to weight streaming for 47 layers. With weight eviction (`GLM4_MOE_LITE_EVICT_WEIGHTS=1`), the full model fits but pays a host-device DMA cost per layer.

---

## How to Reproduce

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# Run the single-device benchmark
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_single_device.py

# Run with DRAM-sharded weights for comparison
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_single_device.py

# Run with fused MoE for comparison
GLM4_MOE_LITE_FUSED_MOE=1 \
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_single_device.py
```

---

## Hybrid vs Agentic: What the Numbers Mean

Since the hybrid implementation re-exports the agentic's optimized TTNN functions (not reimplementations), the per-layer performance is **identical** between the two. The hybrid adds:

1. **TTNNModule framework** (~0 overhead at runtime — wraps existing functions)
2. **HuggingFace module replacement** (one-time cost at model init, not during inference)
3. **Compressed KVPE cache as a reusable module** (same underlying `paged_update_cache` ops)

The value of the hybrid is not faster per-layer compute, but rather:
- **Easier model loading** via `from_pretrained` + automatic module replacement
- **Cleaner weight lifecycle** via `preprocess_weights` / `move_weights_to_device`
- **Reusable components** that work across model variants
- **All agentic optimizations accessible** via the same env var knobs

# Qwen3.5-27B P300x2 Optimization Plan

## Current State

- **19.9 tok/s/user** (50.3ms TPOT) at batch=32, 64K context, 4 Blackhole chips
- **Target: >30 tok/s/user** (<33ms TPOT)
- **Gap: ~17ms** to cut

## Bottleneck Analysis

The bottleneck is **device op dispatch overhead**, not DRAM bandwidth:

- ~3400 ops/forward × ~15μs/op = ~50ms
- bfp4 weights had zero effect (confirms not BW-bound)
- Trace captures the op graph but each op still has dispatch cost

## Optimizations Applied (11.4 → 19.9 tok/s)

1. L1 interleaved (eliminated DRAM roundtrips): 87.5 → 82.7ms
2. TP-sharded DeltaNet weights: 82.7 → 74.6ms (4x less DRAM per device)
3. Fused projection (QK+V+ZBA single matmul): 74.6 → 73.6ms
4. Head-parallel recurrence + reduce_scatter: 73.6 → 54.5ms (no all-gather)
5. Skip L2 normalize: 54.5 → 51.5ms (12 fewer ops/layer)
6. Fused ttnn.rms_norm: 51.5 → 50.6ms (7 fewer ops/layer)
7. SiLU fusion (apply to qk_proj before split): 50.6 → 50.4ms
8. unary_chain (softplus+neg fused) + transpose_a: 50.4 → 50.3ms

## Failed Experiments (Don't Repeat)

- **GQA pre-expansion in weight**: REGRESSED (50.4 → 51.5ms, +49% DRAM for -2 ops)
- **bfp4 weights**: NO EFFECT (not BW-bound)
- **Direct L1_MEMORY_CONFIG output from dram_matmul_config**: CRASHES (fabric timeout)
- **DRAM prefetcher**: blocked by sub-device incompatibility (clear/reload hangs)

## Benchmark Results (2026-03-23)

### Single User (Concurrency=1)

| ISL | OSL | TPOT (ms) | p99 TPOT (ms) | Output TPS |
|----:|----:|----------:|--------------:|-----------:|
| 128 | 128 | **50.3** | 50.4 | 19.9 |
| 1,024 | 128 | **50.7** | 50.7 | 19.7 |
| 2,048 | 128 | **51.1** | 51.1 | 19.6 |
| 4,096 | 128 | **51.8** | 51.8 | 19.3 |
| 8,192 | 128 | **53.2** | 53.2 | 18.8 |

### Batch (Max Concurrency)

| ISL | OSL | Con | TPOT (ms) | p99 TPOT (ms) | Output TPS | Per-User TPS |
|----:|----:|----:|----------:|--------------:|-----------:|-------------:|
| 128 | 128 | 32 | **50.8** | 50.9 | **630.5** | 19.7 |
| 1,024 | 128 | 32 | **52.1** | 52.3 | **614.0** | 19.2 |
| 2,048 | 128 | 32 | **53.7** | 53.8 | **596.3** | 18.6 |
| 4,096 | 128 | 31 | **56.6** | 56.8 | **547.5** | 17.7 |
| 8,192 | 128 | 15 | **56.4** | 56.5 | **265.8** | 17.7 |

Note: TTFT not measured (DeltaNet prefill not implemented). Currently TTFT = ISL x TPOT (sequential decode).

## Next Steps

### P1: Fused C++ DeltaNet Kernel (est. -31ms, target ~24ms TPOT)

The single biggest lever. Infrastructure exists at `ttnn/cpp/ttnn/operations/experimental/deltanet/` with a working stub. Python binding `ttnn.experimental.deltanet_recurrence` is callable.

**What it replaces per layer:** ~43 Python ops → 1 kernel call

- Element-wise: decay × state, (V - kv_mem) × beta, gate computation
- Matmuls: K @ state, K^T @ delta, Q @ state
- 43 ops × 48 DeltaNet layers × ~15μs/op = ~31ms saved

**What's needed:**

1. Implement tile matmul APIs (`matmul_tiles`, `matmul_init_short`) in the compute kernel
2. Adapt host operation for head-parallel TP (12 heads/device, not 48)
3. Handle GQA expansion (3:1 Q:K ratio) inside the kernel
4. State management: read state from DRAM, write back updated state

**Predicted: 50.3 - 31 + 5 (kernel exec) = ~24ms = 42 tok/s**

### P2: DeltaNet Prefill Implementation (TTFT improvement)

Currently prefill is not implemented for DeltaNet — only decode. TTFT = ISL × TPOT (sequential decode), which is bad for long contexts.

**Options:**

- Chunk-wise parallel scan: process ISL tokens in chunks, compute recurrence state in parallel within each chunk
- Sequential but fused: even running decode sequentially, the fused kernel (P1) would cut TTFT by ~60%

### P3: Reduce Attention Layer Overhead (16 full_attention layers)

The 16 full_attention layers (every 4th) contribute ops. Potential:

- Flash attention / fused QKV if not already fused
- Fused attention kernel: combine Q×K, softmax, ×V into single kernel

### P4: DRAM Prefetcher (blocked, needs upstream fix)

Previously attempted but blocked by sub-device incompatibility.

- Expected gain: hides DRAM latency for weight reads
- Blocked by: `clear/reload` hangs with sub-device mode

### P5: MLP Op Fusion

The MLP layers (shared across all 64 layers) run w1, w3, SiLU, multiply, w2.

- Fuse SiLU × gate into the w1/w3 matmul epilogue
- Could save a few ops/layer × 64 layers

### Priority Summary

| Priority | Optimization | Est. Savings | Effort | Status |
|:--------:|-------------|:------------:|:------:|:------:|
| **P1** | Fused C++ DeltaNet kernel | **~31ms** | High | Stub exists, needs matmul impl |
| P2 | DeltaNet prefill | TTFT only | Medium | Not started |
| P3 | Attention layer fusion | ~2-5ms | Medium | Not investigated |
| P4 | DRAM prefetcher | ~3-5ms | Low (if unblocked) | Blocked upstream |
| P5 | MLP epilogue fusion | ~1-2ms | Low | Not investigated |

**Recommendation:** Focus entirely on P1. The fused kernel alone gets to >30 tok/s with margin. Everything else is incremental.

## Architecture Reference

- **Model:** 64 layers (48 DeltaNet linear_attention + 16 full_attention, every 4th)
- **TP:** 4-way across P300x2 (1,4 mesh)
- **DeltaNet:** Head-parallel (12 V-heads/device, 4 K-heads/device, GQA=3)
- **Fused in_proj:** (5120, 4160/dev), ShardTensorToMesh(dim=-1)
- **Row-parallel out_proj:** (1536/dev, 5120) + reduce_scatter
- **Gate:** unary_chain(softplus+neg) → mul(A_exp) → exp
- **Recurrence:** all local per device, matmul transpose_a for K^T
- **KV cache:** Only for 16 attention layers (paged, block_size=64)
- **DeltaNet state:** Fixed-size (128×128 per head), O(1) in context length

# GLM-4.7-Flash Hybrid: Performance Testing & Comparison Guide

This document covers how to benchmark the hybrid implementation against the original agentic baseline, what metrics to collect, which knobs to tune, and how to interpret results.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Test Tiers](#test-tiers)
   - [Tier 1: Per-Layer Correctness + Latency](#tier-1-per-layer-correctness--latency)
   - [Tier 2: End-to-End Greedy Decode](#tier-2-end-to-end-greedy-decode-primary-benchmark)
   - [Tier 3: Detailed Stage Profiling](#tier-3-detailed-stage-profiling)
   - [Tier 4: Device-Level Tracy Profiler](#tier-4-device-level-tracy-profiler)
3. [Runtime Configuration Knobs](#runtime-configuration-knobs)
4. [Comparison Matrix Template](#comparison-matrix-template)
5. [Optimization Impact Reference](#optimization-impact-reference)
6. [Codebase Map: Where Tests Live](#codebase-map-where-tests-live)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Details |
|---|---|
| Hardware | Tenstorrent Wormhole: N150 (1 chip), N300 (2 chips), T3K Loud Box (8 chips), or TG/Galaxy (32 chips) |
| Model snapshot | `zai-org/GLM-4.7-Flash` — all 48 safetensors shards |
| Software | tt-metal with TTNN SDK, Python 3.10+, PyTorch, transformers |
| Env gating | `TT_ENABLE_HW_TESTS=1` for hardware tests |
| Env gating | `TT_ENABLE_LARGE_MODEL_TESTS=1` for full-model tests |
| Env gating | `TT_ENABLE_MULTI_DEVICE_TESTS=1` for multi-device MoE tests |

### Hardware Configurations

| System | Chips | Mesh Shape | DRAM | Weight Eviction | TP |
|---|---|---|---|---|---|
| **N150 (Quiet Box)** | 1 | `(1, 1)` | 12.8 GB | Required | Off |
| **N300** | 2 | `(1, 2)` | 25.6 GB | Optional | Optional |
| **T3K (Loud Box)** | 8 | `(1, 8)` | 102.4 GB | Not needed | Recommended |
| **TG / Galaxy** | 32 | `(8, 4)` | 409.6 GB | Not needed | Required |

Verify the snapshot is complete before benchmarking:

```bash
python3 -c "
from models.demos.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir
snap = resolve_best_effort_snapshot_dir('zai-org/GLM-4.7-Flash')
missing = find_missing_shards(snap)
print(f'Snapshot: {snap}')
print(f'Missing shards: {len(missing)}')
if missing:
    for m in missing[:5]:
        print(f'  {m}')
"
```

---

## Test Tiers

### Tier 1: Per-Layer Correctness + Latency

Runs a single decoder layer and verifies PCC >= 0.99 against a CPU reference. Fast to run, useful for validating individual components and catching regressions.

#### Layer 0 (attention + dense MLP)

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# Prefill path
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_tt_layer0_optional.py -v

# Decode with paged cache update
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_tt_layer0_decode_update_cache_optional.py -v

# Decode with batch=32
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_tt_layer0_decode_batch32_optional.py -v

# Unpaged decode (reference comparison)
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_tt_layer0_decode_unpaged_optional.py -v
```

#### MoE Layer 1 (attention + sparse/routed experts)

```bash
# Single device MoE
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_tt_moe_layer1_optional.py -v

# Multi-device mesh MoE
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_tt_moe_layer1_mesh_optional.py -v
```

#### Generic Decoder Layer (prefill + decode)

```bash
# Prefill through generic decoder layer
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_tt_decoder_layer0_prefill_update_cache_optional.py -v

# Decode through generic decoder layer
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_tt_decoder_layer0_decode_update_cache_optional.py -v
```

#### Boundary Tests

```bash
# FlashMLA at K-chunk and cache-block boundaries
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
pytest models/demos/glm4_moe_lite/tests/test_flash_mla_decode_boundary_optional.py -v
```

#### PCC Thresholds

| Test | Target PCC |
|---|---|
| Layer 0 prefill | >= 0.99 |
| Layer 0 decode | >= 0.99 |
| Generic decoder layer | >= 0.999 |
| MoE layer 1 | >= 0.98 |
| FlashMLA boundary | >= 0.95 |

#### Hybrid Framework Tests (no hardware needed)

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# Run all 23 framework tests
python3 -m pytest models/demos/glm4_moe_lite_hybrid/tests/test_hybrid_modules.py \
  -v --noconftest -k "not TTNNIntegration"
```

---

### Tier 2: End-to-End Greedy Decode (Primary Benchmark)

This is the **main performance benchmark**. It runs the full model (all 47 layers) through prefill and decode, reporting wall-clock latency and throughput.

#### Script Location

```
models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py
```

#### Agentic Baseline Runs

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# --- Single device (N150), eager mode ---
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --mesh-cols 1 \
  --phase both

# --- 8-device (T3K), eager mode ---
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --mesh-cols 8 \
  --phase both

# --- 8-device (T3K), traced mode (best throughput) ---
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --mesh-cols 8 \
  --phase both \
  --enable-trace --trace-mode sampling

# --- BF8 KV cache (memory optimized) ---
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --mesh-cols 8 \
  --phase both \
  --kv-cache-dtype bf8
```

#### Key CLI Arguments

| Argument | Values | Description |
|---|---|---|
| `--prompt` | any string | Input prompt text |
| `--max-new-tokens` | int (default: 32) | Number of tokens to generate |
| `--mesh-cols` | 1, 2, 4, 8 | Number of devices in mesh |
| `--phase` | `prefill`, `decode`, `both` | Which phases to run |
| `--enable-trace` | flag | Enable decode tracing (faster) |
| `--trace-mode` | `logits`, `sampling` | Trace strategy |
| `--kv-cache-dtype` | `bf16`, `bf8` | KV cache data type |
| `--device-ids` | comma-separated | Specific device IDs to use |

#### Reported Metrics

```
=== TT greedy decode (eager) ===
mesh_shape=(1,8) kv_cache_dtype=bf8 phase=both
prompt_len=12 new_tokens=64 blocks_per_seq=128
prefill_s=0.847 decode_tok_s=0.0312 tok_s=32.05

--- Per-token decode latency (ms) ---
  first token:       45.2 ms
  subsequent:   mean=    31.2  min=    28.9  max=    38.7
```

| Metric | Key | Unit | Description |
|---|---|---|---|
| Prefill latency | `prefill_s` | seconds | Time to process the full prompt |
| Decode latency/token | `decode_tok_s` | seconds | Average time per generated token |
| Decode throughput | `tok_s` | tokens/sec | Inverse of decode_tok_s |
| First-token decode | `first token` | ms | Includes trace capture if tracing |
| Steady-state decode | `subsequent mean` | ms | Mean latency after first token |

---

### Tier 3: Detailed Stage Profiling

Enables per-layer, per-stage timing breakdown to identify bottlenecks.

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# All layers, print every step
GLM4_MOE_LITE_PROFILE=1 \
GLM4_MOE_LITE_PROFILE_LAYER=-1 \
GLM4_MOE_LITE_PROFILE_PRINT_EVERY=1 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Hello" \
  --max-new-tokens 4 \
  --mesh-cols 8 \
  --phase both

# Single layer (e.g., layer 5 — an MoE layer)
GLM4_MOE_LITE_PROFILE=1 \
GLM4_MOE_LITE_PROFILE_LAYER=5 \
GLM4_MOE_LITE_PROFILE_PRINT_EVERY=1 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Hello" \
  --max-new-tokens 4 \
  --mesh-cols 8 \
  --phase both
```

#### Stage Breakdown Keys

| Stage | Profile Key | Component |
|---|---|---|
| Input RMSNorm | `norm_s` | Pre-attention norm |
| KV cache update | `kv_cache_update_s` | KV projection + RoPE + paged cache write |
| Q projection | `q_path_s` | Q LoRA + RoPE + kv_b1 |
| FlashMLA decode | `flash_mla_decode_s` | Paged multi-latent attention kernel |
| Attention output | `attn_out_s` | kv_b2 + head flatten + w_o |
| Post-attention norm | `mlp_norm_s` | Pre-MLP norm |
| Shared expert MLP | `moe_shared_s` | Dense SwiGLU (shared expert) |
| MoE router | `moe_router_s` | sigmoid + bias + topk |
| Routed experts | `moe_experts_s` | Sparse/dense/packed expert execution |
| Shared+routed merge | `moe_merge_s` | Add + optional all_reduce |
| Embedding | `embed_s` | Token embedding lookup |
| LM head | `head_s` | Final norm + vocabulary projection |
| Total | `total_s` | End-to-end layer time |
| Aggregate throughput | `agg_tps` | tokens/sec across all layers |

---

### Tier 4: Device-Level Tracy Profiler

For kernel-level analysis (dispatch overhead, compute utilization, memory bandwidth).

```bash
cd /home/ubuntu/agent/agentic/tt-metal

TT_METAL_DEVICE_PROFILER=1 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Hello" \
  --max-new-tokens 2 \
  --mesh-cols 1 \
  --phase both
```

Tracy traces are written to `/tmp/` and can be viewed with the Tracy profiler GUI.

---

## Runtime Configuration Knobs

All knobs are read from environment variables once at model init. Sweep these to find optimal settings for your hardware.

### Memory Optimization

| Env Var | Default | Values | Effect |
|---|---|---|---|
| `GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS` | `0` | `0`, `1` | DRAM-sharded weight storage |
| `GLM4_MOE_LITE_DRAM_SHARDED_MLP` | `1` (if weights sharded) | `0`, `1` | DRAM-sharded MLP matmuls |
| `GLM4_MOE_LITE_DRAM_SHARDED_ATTN` | `0` | `0`, `1` | DRAM-sharded attention matmuls |
| `GLM4_MOE_LITE_DECODE_L1_ACT` | `0` | `0`, `1` | Keep decode activations in L1 |
| `GLM4_MOE_LITE_KV_CACHE_TT_DTYPE` | `bf8` | `bf8`, `bf16` | KV cache data type (bf8 = 2x savings) |

### Attention

| Env Var | Default | Values | Effect |
|---|---|---|---|
| `GLM4_MOE_LITE_MLA_SHARD_Q` | `0` | `0`, `1` | Height-shard Q for FlashMLA |
| `GLM4_MOE_LITE_HEAD_PARALLEL_KVB2` | `0` | `0`, `1` | Head-parallel kv_b2 path (TP only) |
| `GLM4_MOE_LITE_FUSED_KV_BRANCH` | `0` | `0`, `1` | Fused KV branch C++ kernel (batch=1) |
| `GLM4_MOE_LITE_FUSE_QKV_A` | `0` | `0`, `1` | Fused Q+KV_a projection |
| `GLM4_MOE_LITE_MLA_SCALE_MODE` | `qk` | `qk`, `kvpe` | Attention scale denominator |
| `GLM4_MOE_LITE_MLA_K_CHUNK_SIZE` | `64` | int | FlashMLA K chunk size |
| `GLM4_MOE_LITE_CONCAT_HEADS` | `0` | `0`, `1` | Use concatenate_heads vs permute |
| `GLM4_MOE_LITE_ATTN_DP` | `0` | `0`, `1` | Replicate attention weights (no TP sharding) |

### MoE Strategy

| Env Var | Default | Values | Effect |
|---|---|---|---|
| `GLM4_MOE_LITE_MOE_EXPERTS_IMPL` | `sparse` | `sparse`, `dense_decode` | Expert execution path |
| `GLM4_MOE_LITE_FUSED_MOE` | `0` | `0`, `1` | fused_persistent_moe_decode kernel |
| `GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE` | `0` | `0`, `1` | Fuse shared+routed all_reduce |
| `GLM4_MOE_LITE_FUSE_SHARED_GATE_UP` | `0` | `0`, `1` | Fuse gate+up for shared expert |
| `GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP` | `0` | `0`, `1` | Fuse gate+up for routed experts |
| `GLM4_MOE_LITE_MOE_ROUTER_IMPL` | `tt` | `tt`, `cpu` | Router implementation (cpu = debug) |
| `GLM4_MOE_LITE_MOE_DENSE_PREFILL` | `0` | `0`, `1` | Dense prefill for MoE |
| `GLM4_MOE_LITE_MOE_PACKED_PREFILL` | `0` | `0`, `1` | Token-packing prefill for MoE |
| `GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL` | `reduce` | `reduce`, `a2a` | Sparse dispatch strategy |

### Precision

| Env Var | Default | Values | Effect |
|---|---|---|---|
| `GLM4_MOE_LITE_MLP_FIDELITY` | `lofi` | `lofi`, `hifi2`, `hifi4` | MLP math fidelity |
| `GLM4_MOE_LITE_MLA_FIDELITY` | `hifi4` | `lofi`, `hifi2`, `hifi4` | MLA math fidelity |
| `GLM4_MOE_LITE_MLP_APPROX` | `1` | `0`, `1` | MLP math approximation |
| `GLM4_MOE_LITE_MLA_APPROX` | `0` | `0`, `1` | MLA math approximation |
| `GLM4_MOE_LITE_MOE_FP32_ACC` | `0` | `0`, `1` | FP32 accumulation for MoE |
| `GLM4_MOE_LITE_EXPERTS_TT_DTYPE` | `bf8` | `bf8`, `bf16` | Expert weight dtype |
| `GLM4_MOE_LITE_DENSE_TT_DTYPE` | `bf16` | `bf8`, `bf16` | Dense weight dtype |

### Tensor Parallelism

| Env Var | Default | Values | Effect |
|---|---|---|---|
| `GLM4_MOE_LITE_TP` | `0` | `0`, `1` | Enable tensor parallelism |

### Debug

| Env Var | Default | Values | Effect |
|---|---|---|---|
| `GLM4_MOE_LITE_PROFILE` | `0` | `0`, `1` | Enable stage profiling |
| `GLM4_MOE_LITE_PROFILE_LAYER` | `-1` | int | Layer to profile (-1 = all) |
| `GLM4_MOE_LITE_PROFILE_PRINT_EVERY` | `0` | int | Print interval (0 = end only) |
| `GLM4_MOE_LITE_LAYER_IDENTITY` | `0` | `0`, `1` | Skip layer compute (passthrough) |
| `GLM4_MOE_LITE_SKIP_KV_UPDATE` | `0` | `0`, `1` | Skip KV cache writes |
| `GLM4_MOE_LITE_DISABLE_MLP` | `0` | `0`, `1` | Skip MLP computation |
| `GLM4_MOE_LITE_DISABLE_FLASH_MLA_DECODE` | `0` | `0`, `1` | Zero FlashMLA output |
| `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES` | `0` | `0`, `1` | Skip aliasing-safety clones |

---

## Comparison Matrix Template

Run the Tier 2 benchmark with identical prompt, `--max-new-tokens 64`, and `--mesh-cols 8` for both configurations:

### Configuration A: Agentic Baseline

```bash
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 --mesh-cols 8 --phase both
```

### Configuration B: Hybrid (same agentic backend, modular framework)

The hybrid re-exports the same optimized functions, so initially the numbers should match. Divergence indicates framework overhead or configuration differences.

### Results Table

| Metric | Agentic Baseline | Hybrid | Delta (%) | Notes |
|---|---|---|---|---|
| **Prefill** | | | | |
| Prefill latency (s) | | | | |
| Prefill tokens/s | | | | |
| **Decode** | | | | |
| First-token decode (ms) | | | | |
| Mean steady-state decode (ms) | | | | |
| Min decode latency (ms) | | | | |
| Max decode latency (ms) | | | | |
| Decode throughput (tok/s) | | | | |
| **Memory** | | | | |
| KV cache size per layer (MB) | | | | |
| Total KV cache (GB) | | | | |
| Peak DRAM usage (GB) | | | | |
| **Correctness** | | | | |
| Layer 0 PCC | | | | Target >= 0.99 |
| MoE layer PCC | | | | Target >= 0.98 |
| Golden token match | | | | Exact match expected |

### Knob Sweep Results

Run each config variation with `--max-new-tokens 64 --mesh-cols 8`:

| Configuration | tok/s | decode_ms | prefill_s | Notes |
|---|---|---|---|---|
| Default (all off) | | | | Baseline |
| + DRAM_SHARDED_WEIGHTS=1 | | | | Memory opt |
| + FUSED_KV_BRANCH=1 | | | | Batch=1 attention |
| + FUSED_MOE=1 | | | | MoE kernel |
| + MLA_SHARD_Q=1 | | | | Q sharding |
| + enable-trace (sampling) | | | | Trace capture |
| + KV_CACHE_TT_DTYPE=bf8 | | | | Memory savings |
| All optimizations ON | | | | Best case |

---

## Optimization Impact Reference

Expected impact from the agentic optimizations ported into the hybrid:

| Optimization | Expected Impact | Phase | How to Enable |
|---|---|---|---|
| Compressed KVPE (576-dim BF8) | ~2x KV cache memory reduction | 2a | `--kv-cache-dtype bf8` (default) |
| Fused KV branch kernel | ~30-40% attention latency (batch=1) | 2b | `GLM4_MOE_LITE_FUSED_KV_BRANCH=1` |
| DRAM-sharded matmuls | ~20-30% decode latency for large matmuls | 4a | `GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1` |
| 1D multicast program config | ~10-15% decode matmul improvement | 4a | `GLM4_MOE_LITE_EXPLICIT_PROG_CFG=1` |
| Sparse MoE (block_size=32) | Baseline expert performance | 3b | Default (`MOE_EXPERTS_IMPL=sparse`) |
| Fused persistent MoE decode | ~15-20% MoE latency reduction | 3b | `GLM4_MOE_LITE_FUSED_MOE=1` |
| Decode trace batching | ~2-3x throughput (batched serving) | 5a | `--enable-trace --trace-mode sampling` |
| MTP speculative decoding | ~1.5-2x effective throughput | 5b | Requires MTP weights |
| Fused gate+up (shared expert) | ~5-10% shared MLP improvement | 3c | `GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1` |
| Fused gate+up (routed experts) | ~5-10% routed MoE improvement | 3b | `GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1` |

---

## Codebase Map: Where Tests Live

### Agentic (original)

```
models/demos/glm4_moe_lite/
├── scripts/
│   └── debug_run_full_tt_greedy.py     # <-- PRIMARY PERF BENCHMARK
├── tests/
│   ├── test_tt_layer0_optional.py               # Layer 0 prefill PCC
│   ├── test_tt_layer0_decode_update_cache_optional.py  # Layer 0 decode PCC
│   ├── test_tt_layer0_decode_batch32_optional.py       # Batch-32 decode
│   ├── test_tt_moe_layer1_optional.py                  # MoE single-device
│   ├── test_tt_moe_layer1_mesh_optional.py             # MoE multi-device
│   ├── test_flash_mla_decode_boundary_optional.py      # FlashMLA boundaries
│   ├── test_tt_decoder_layer0_prefill_update_cache_optional.py
│   ├── test_tt_decoder_layer0_decode_update_cache_optional.py
│   ├── test_tt_golden_truncated_n2_optional.py  # 2-layer golden token match
│   ├── test_pre_sdpa_kernel.py                  # Fused PreSDPA kernel
│   ├── test_tt_embedding_optional.py            # Embedding lookup
│   ├── test_weights.py                          # Weight loading
│   └── test_reference_*.py                      # CPU reference sanity
└── tt/
    └── model_tt.py              # Glm4MoeLiteDenseOnlyTT (profile support)
```

### Hybrid (new)

```
models/demos/glm4_moe_lite_hybrid/
├── tests/
│   └── test_hybrid_modules.py   # 23 framework tests (no hardware needed)
├── runner.py                    # HybridGlm4Runner (top-level orchestrator)
└── README.md                    # Architecture + usage docs
```

---

## Running on Different Hardware: Single-Chip vs T3K

### N150 / Single Chip (Quiet Box)

```bash
# Layer-level benchmark (fits in single device DRAM)
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_single_device.py

# Full model decode (weight eviction enabled automatically)
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 1 --prompt "Hello world" --max-new-tokens 32 --phase both

# Full model with DRAM-sharded weights
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 1 --prompt "Hello world" --max-new-tokens 32 --phase both
```

**What happens on single chip:**
- Mesh shape: `(1, 1)`, no fabric configuration
- Weight eviction: auto-enabled (layers streamed from host)
- TP: disabled (no parallelism axis)
- All 64 MoE experts on one device
- Expect ~4-5 tok/s decode (memory-bandwidth-limited)

### T3K / 8-Chip (Loud Box)

```bash
# Full model, eager mode
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 --prompt "Hello world" --max-new-tokens 64 --phase both

# Full model, traced mode (best throughput)
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 --prompt "Hello world" --max-new-tokens 64 --phase both \
  --enable-trace --trace-mode sampling

# Full model, all optimizations
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 \
GLM4_MOE_LITE_FUSED_MOE=1 \
GLM4_MOE_LITE_TP=1 \
GLM4_MOE_LITE_MLA_SHARD_Q=1 \
GLM4_MOE_LITE_HEAD_PARALLEL_KVB2=1 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 --prompt "Hello world" --max-new-tokens 64 --phase both \
  --enable-trace --trace-mode sampling

# Specific device IDs (if not using all 8)
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 --device-ids 0,1,2,3,4,5,6,7 \
  --prompt "Hello world" --max-new-tokens 64 --phase both
```

**What happens on T3K:**
- Mesh shape: `(1, 8)`, fabric auto-configured (`FABRIC_1D`)
- Weight eviction: disabled (102.4 GB aggregate DRAM holds all 47 layers)
- TP: enabled (`GLM4_MOE_LITE_TP=1`), weights sharded across 8 devices
- 64 experts / 8 devices = 8 experts per device
- Expect ~30-40 tok/s decode (traced mode)

### N300 / 2-Chip

```bash
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 2 --prompt "Hello world" --max-new-tokens 32 --phase both
```

### Multi-Device MoE Correctness Test

```bash
TT_ENABLE_HW_TESTS=1 \
TT_ENABLE_LARGE_MODEL_TESTS=1 \
TT_ENABLE_MULTI_DEVICE_TESTS=1 \
TT_TEST_MESH_SHAPE=1x8 \
pytest models/demos/glm4_moe_lite/tests/test_tt_moe_layer1_mesh_optional.py -v
```

### Key Differences: Single-Chip vs T3K

| Aspect | N150 (1 chip) | T3K (8 chips) |
|---|---|---|
| Weight eviction | Auto-enabled | Disabled |
| TP | Off | On (recommended) |
| Experts per device | 64 | 8 |
| All-reduce | N/A | After each TP projection |
| MoE dispatch | Local only | `reduce` or `a2a` across mesh |
| DRAM budget | 12.8 GB | 102.4 GB |
| Fabric | Not configured | `FABRIC_1D` (auto) |
| Decode throughput | ~4-5 tok/s | ~30-40 tok/s |
| Fused KV branch | Supported | Supported |
| Trace mode | Supported | Supported (best perf) |

---

## The One Test: 3-Way Comparison on T3K

If you can only run one test to compare all three approaches on a T3K (8-chip) system, use `debug_run_full_tt_greedy.py`. It runs the full 47-layer model end-to-end and reports prefill latency, decode tokens/sec, and per-token latency breakdown. Run it three times with different env-var configurations representing each approach.

### Run 1: tt-symbiote-like Baseline (no agentic optimizations)

Simulates what tt-symbiote delivers: no fused kernels, dense expert path, high-fidelity math.

```bash
cd /home/ubuntu/agent/agentic/tt-metal

GLM4_MOE_LITE_MOE_EXPERTS_IMPL=dense_decode \
GLM4_MOE_LITE_MLP_FIDELITY=hifi4 \
GLM4_MOE_LITE_MLA_FIDELITY=hifi4 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --phase both
```

### Run 2: Agentic Baseline (production optimizations + trace)

The agentic production configuration: fused KV branch, sparse MoE, DRAM-sharded weights, and trace capture/replay.

```bash
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 \
GLM4_MOE_LITE_MOE_EXPERTS_IMPL=sparse \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --phase both \
  --enable-trace --trace-mode sampling
```

### Run 3: Hybrid (all optimizations combined)

Everything from the agentic run plus fused MoE decode, trace capture/replay (also available in the agentic baseline), fused gate+up, head-parallel kv_b2, Q sharding, BF8 KV cache, and fused MLP+MoE reduce.

```bash
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 \
GLM4_MOE_LITE_FUSED_MOE=1 \
GLM4_MOE_LITE_TP=1 \
GLM4_MOE_LITE_MLA_SHARD_Q=1 \
GLM4_MOE_LITE_HEAD_PARALLEL_KVB2=1 \
GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 \
GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1 \
GLM4_MOE_LITE_KV_CACHE_TT_DTYPE=bf8 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --phase both \
  --enable-trace --trace-mode sampling \
  --kv-cache-dtype bf8
```

### How to Collect Results

After each run, look for these lines in the output:

```
prefill_s=___  decode_tok_s=___  tok_s=___

--- Per-token decode latency (ms) ---
  first token:       ___ ms
  subsequent:   mean=___  min=___  max=___
```

### Results Table (fill in after running)

| Metric | tt-symbiote-like (Run 1) | Agentic Baseline (Run 2) | Hybrid (Run 3) | Hybrid vs Agentic |
|---|---|---|---|---|
| **Prefill latency (s)** | | | | |
| **Decode latency/token (ms)** | | | | |
| **Throughput (tok/s)** | | | | |
| **First-token decode (ms)** | | | | |
| **Steady-state mean (ms)** | | | | |
| **Steady-state min (ms)** | | | | |

### Configuration Comparison

| Setting | tt-symbiote-like (Run 1) | Agentic Baseline (Run 2) | Hybrid (Run 3) |
|---|---|---|---|
| MoE expert path | `dense_decode` | `sparse` | `sparse` + `fused` |
| MLP fidelity | `hifi4` | `lofi` (default) | `lofi` (default) |
| MLA fidelity | `hifi4` | `hifi4` (default) | `hifi4` (default) |
| DRAM-sharded weights | Off | On | On |
| Fused KV branch | Off | On | On |
| Fused MoE decode | Off | Off | On |
| Fused gate+up (shared) | Off | Off | On |
| Fused gate+up (experts) | Off | Off | On |
| Fused MLP+MoE reduce | Off | Off | On |
| Head-parallel kv_b2 | Off | Off | On |
| Q sharding | Off | Off | On |
| TP | Default | Default | On |
| KV cache dtype | bf16 | bf16 | bf8 |
| Trace mode | Off | **`sampling`** | `sampling` |

### What Each Run Represents

- **Run 1 (tt-symbiote-like)** uses the correctness-first settings: dense per-expert execution (no sparse kernels), HiFi4 math everywhere. This is the performance floor — what you get with generic TTNN ops and no fused kernels.

- **Run 2 (Agentic Baseline)** enables the key agentic optimizations: fused KV branch for batch-1 attention, sparse MoE, DRAM-sharded weights, and trace capture/replay. This is the current production baseline at peak performance.

- **Run 3 (Hybrid)** layers every additional optimization on top: fused persistent MoE, trace capture/replay (note: also available in Run 2 via `--enable-trace`), fused gate+up for both shared and routed experts, fused MLP+MoE reduce (one all_reduce instead of two), head-parallel kv_b2, Q sharding, BF8 KV cache, and TP. The perf delta vs Run 2 comes from the fused ops and TP improvements, not from tracing (which both have).

### Single-Chip (N150) Variant

For a single-chip system, replace `--mesh-cols 8` with `--mesh-cols 1` and remove TP-specific flags:

```bash
# Run 3 adapted for N150
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 \
GLM4_MOE_LITE_FUSED_MOE=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 \
GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1 \
GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 \
GLM4_MOE_LITE_KV_CACHE_TT_DTYPE=bf8 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 1 \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 32 \
  --phase both \
  --kv-cache-dtype bf8
```

Note: On N150, `--enable-trace` may not work with weight eviction. TP, head-parallel kv_b2, and Q sharding are disabled automatically for single-device. Expect ~4-5 tok/s (memory-bandwidth-limited by weight streaming).

---

## Verifying the TP Communication Claim (T3K only)

The hybrid claims ~8% improvement from replacing `all_reduce` with `reduce_scatter_minimal_async` for TP linears. Here's how to verify this on T3K.

### Microbenchmark: Isolated TP Communication

Directly measures `all_reduce` vs `reduce_scatter_minimal_async` at the op level, then at the TP-linear level, then projected across 47 layers:

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# On T3K (8 devices)
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_tp_communication.py --mesh-cols 8

# On N300 (2 devices)
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_tp_communication.py --mesh-cols 2
```

The script runs three tests:
1. **Isolated communication** — same tensor, just the reduce op, no matmul
2. **TP linear** — `mesh_partition + matmul + reduce`, simulating one attention projection
3. **Simulated layer** — 7 TP projections back-to-back (q_a, q_b, kv_a, kv_b2, w_o, mlp_gate, mlp_down)

It prints a verdict: VERIFIED, NOT VERIFIED, or REFUTED based on whether `reduce_scatter` is measurably faster.

### Macro-Level: Full Model with Different TP Reduce Strategies

This is not directly switchable via env var in the current agentic code (it always uses `all_reduce`). But you can compare the profiled TP overhead:

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# Profile with stage breakdown to see all_reduce time
GLM4_MOE_LITE_PROFILE=1 \
GLM4_MOE_LITE_PROFILE_LAYER=5 \
GLM4_MOE_LITE_PROFILE_PRINT_EVERY=1 \
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 \
GLM4_MOE_LITE_TP=1 \
python3 models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 8 \
  --prompt "Hello" \
  --max-new-tokens 4 \
  --phase both \
  --enable-trace --trace-mode sampling
```

In the stage breakdown, look for the total time in attention and MLP stages — the portion spent in `all_reduce` is the theoretical ceiling for `reduce_scatter` improvement.

### Important: Single-Chip Has Zero TP Communication

On your N150, TP is disabled (single device, no mesh axis to shard across). The `reduce_scatter` vs `all_reduce` difference is **exactly zero** on single-chip. This claim can only be verified on N300 (2-chip) or T3K (8-chip).

---

## Troubleshooting

### Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: ttnn.device` | TTNN SDK not installed/activated | Activate the tt-metal environment |
| `find_missing_shards` reports missing files | Incomplete snapshot download | Re-download all 48 safetensors |
| `ttnn.zeros` hangs on MeshDevice | Known issue with some mesh configurations | Script uses `ttnn.as_tensor` from CPU zeros instead |
| OOM during weight loading | All 47 layers loaded simultaneously | Set `GLM4_MOE_LITE_EVICT_WEIGHTS=1` for single-device |
| Decode trace segfault | Fabric config not set before mesh open | Script calls `_set_default_fabric_config()` |
| Low PCC on MoE layers | BF16 bias centering not applied | Ensure bias is centered before BF16 cast (done by default) |
| `all_to_all_dispatch` hangs | Mesh axis misconfigured | Use `GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL=reduce` |

### Quick Sanity Check (no full model needed)

```bash
cd /home/ubuntu/agent/agentic/tt-metal

# Run hybrid framework tests (always works, no hardware)
python3 -m pytest models/demos/glm4_moe_lite_hybrid/tests/test_hybrid_modules.py \
  -v --noconftest -k "not TTNNIntegration"

# Run agentic weight loading tests (no hardware)
python3 -m pytest models/demos/glm4_moe_lite/tests/test_weights.py -v --noconftest
```

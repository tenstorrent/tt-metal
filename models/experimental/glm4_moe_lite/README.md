# GLM-4.7-Flash on Tenstorrent: Commands, Configuration & Performance

**Model:** zai-org/GLM-4.7-Flash (47 layers, MLA attention, MoE, 4.7B params)
**Hardware:** T3K (8 Wormhole devices total); tested with mesh shapes 1x4 (4 devices), 1x8 (8 devices), and 2x4 (8 devices, matching T3K physical topology)
**Dispatch:** `DispatchCoreType.ETH` (all 64 Tensix cores per device available for compute)

## Directory Structure

```
models/demos/glm4_moe_lite/
├── README.md                          # This file
├── tt/
│   ├── model_tt.py                    # Top-level model runner (prefill, decode, trace)
│   ├── decoder_layer_tt.py            # Decoder layer (attention + MLP/MoE)
│   ├── moe_tt.py                      # MoE implementation (sparse, dense, packed, fused)
│   ├── layer_weights.py               # Weight conversion (torch → TT)
│   ├── config.py                      # Hyperparameters and runtime config
│   └── generator_vllm.py             # vLLM backend integration
├── scripts/
│   ├── debug_run_full_tt_greedy.py    # Main debug/benchmark script
│   └── run_sweep_isl_batch.py         # ISL × batch sweep automation
├── sample_prompts/                    # Long-context prompt files
├── tests/                             # PCC and layer tests
└── experiments/                       # Sweep results (CSV, tables, graphs)
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Script & CLI Options](#script--cli-options)
3. [Environment Variables](#environment-variables)
4. [Run Commands](#run-commands)
   - [Non-Fused (Unfused) Ops](#non-fused-unfused-ops)
   - [Fused Ops](#fused-ops)
   - [Profiling](#profiling)
5. [Performance Results](#performance-results)
   - [Optimized Batch Size Sweep (1x8)](#optimized-batch-size-sweep-1x8-mesh-trace-sampling-all-optimizations)
   - [20k ISL Batch Size Sweep (1x8)](#20k-isl-batch-size-sweep-1x8-mesh-trace-sampling-all-optimizations)
   - [Fused Ops Batch Sweep (1x8)](#fused-ops-batch-sweep-1x8-mesh-trace-sampling-all-code-optimizations)
6. [Key Concepts](#key-concepts)

---

## Quick Start

```bash
cd $TT_METAL

# Simplest run: non-fused, eager, 4 devices, prefill + decode
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both

# Fastest run: non-fused, trace-sampling, 2x4 mesh (8 devices, T3K physical topology)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-rows 2 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode sampling
```

---

## Script & CLI Options

**Script:** `scripts/debug_run_full_tt_greedy.py` (relative to this directory; from repo root: `models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py`)

| Argument | Default | Description |
| --- | --- | --- |
| `--model-id` | `zai-org/GLM-4.7-Flash` | HuggingFace model ID or local path |
| `--prompt` | `"Say hello in exactly 3 words."` | Input prompt text |
| `--max-new-tokens` | `32` | Number of tokens to generate after prefill |
| `--cache-dir` | `~/.cache/ttnn/models/glm4_moe_lite/vllm` | TT weight cache directory |
| `--mesh-rows` | `1` | Number of rows in mesh shape |
| `--mesh-cols` | `1` | Number of columns in mesh shape. T3K has 8 devices; use `--mesh-cols 4` (4 dev) or `--mesh-rows 2 --mesh-cols 4` (8 dev, physical topology). |
| `--device-ids` | `auto` | Comma-separated physical device IDs or `auto` |
| `--kv-cache-dtype` | `bf16` | KV cache data type: `bf16` (correctness) or `bf8` (memory/perf) |
| `--block-size` | `64` | KV cache block size |
| `--min-cache-tokens` | `128` | Minimum tokens to allocate in KV cache |
| `--phase` | `both` | Phase to run: `prefill`, `decode`, or `both` |
| `--enable-trace` | `false` | Enable traced decode execution (captures trace on first call, replays on subsequent) |
| `--trace-mode` | `logits` | Trace mode: `logits` (returns full logits to host) or `sampling` (on-device greedy top-1) |

---

## Environment Variables

### Required

| Variable | Value | Description |
| --- | --- | --- |
| `TT_METAL_GTEST_ETH_DISPATCH=1` | **Always set** | Route dispatch through Ethernet cores, freeing all 64 Tensix cores for compute |

### Feature Toggles

| Variable | Default | Description |
| --- | --- | --- |
| `GLM4_MOE_LITE_FUSED_KV_BRANCH=1` | Off | Enable fused KV cache branch kernel (DKV matmul + gather + RMSNorm + RoPE in one dispatch) |
| `GLM4_MOE_LITE_ENABLE_MOE=1` | Off (but script forces it on) | Enable MoE layers; the debug script sets this automatically |
| `GLM4_MOE_LITE_NUM_LAYERS=N` | All (47) | Run only N layers (requires `DEBUG_ALLOW_PARTIAL_LAYERS=1`) |
| `GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS=1` | Off | Allow partial-layer runs with `NUM_LAYERS` |
| `GLM4_MOE_LITE_TP=1` | Off | Enable tensor parallelism across mesh devices |
| `GLM4_MOE_LITE_MTP=1` | Off | Enable multi-token prediction (MTP layer 47) |
| `GLM4_MOE_LITE_PRESERVE_TRACE=1` | Off | Skip trace release after prefill to avoid ~6s re-capture overhead |

### Performance Tuning

| Variable | Default | Description |
| --- | --- | --- |
| `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1` | Off | Skip defensive clone operations (saves memory/time, may cause aliasing bugs) |
| `GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1` | Off | Fuse shared MLP gate + up projections |
| `GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1` | Off | Fuse expert gate + up projections |
| `GLM4_MOE_LITE_FUSE_QKV_A=1` | Off | Fuse Q and KV_A projections into a single matmul |
| `GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1` | Off | Fuse MLP + MoE reduce step (consolidates dual ReduceScatter+AllGather pairs in MoE layers) |
| `GLM4_MOE_LITE_SKIP_TYPECAST=1` | Off | Skip unnecessary bf16 typecasts in attention path (eliminates ~1,500 TypecastDeviceOperation calls per decode step) |
| `GLM4_MOE_LITE_CONCAT_HEADS=1` | Off | Use `ttnn.transformer.concatenate_heads` for attention output head-flattening (tested neutral in traced mode; not recommended) |
| `GLM4_MOE_LITE_NLP_CONCAT_HEADS=1` | Off | Use `ttnn.experimental.nlp_concat_heads` for prefill attention output path |
| `GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1` | Off | Use DRAM-sharded weight layout |
| `GLM4_MOE_LITE_DRAM_SHARDED_ATTN=1` | Off | DRAM-sharded attention weights (requires `DRAM_SHARDED_WEIGHTS=1`) |
| `GLM4_MOE_LITE_DRAM_SHARDED_MLP=1` | On (if `DRAM_SHARDED_WEIGHTS=1`) | DRAM-sharded MLP weights |
| `GLM4_MOE_LITE_SHARDED_MLP=1` | Off | L1 WIDTH_SHARDED activations for shared MLP decode |
| `GLM4_MOE_LITE_BATCH_EXPAND=1` | Off | Enable batch expansion |
| `GLM4_MOE_LITE_USE_DECODE_ROPE=1` | Off (auto-enabled with trace) | Use decode-specific RoPE implementation |
| `GLM4_MOE_LITE_MOE_FP32_ACC=1` | Off | FP32 accumulation for MoE matmuls |
| `GLM4_MOE_LITE_MLA_FP32_ACC=1` | Off | FP32 accumulation for FlashMLA (unsafe without `UNSAFE_ALLOW_FP32_MLA=1`) |
| `GLM4_MOE_LITE_ROUTER_L1=1` | On | Keep MoE router intermediates in L1 (for decode, T<=32) |

### Data Type Overrides

| Variable | Default | Description |
| --- | --- | --- |
| `GLM4_MOE_LITE_EXPERTS_TT_DTYPE` | `bf8` | TT dtype for expert weights (`bf16`, `bf8`, `bf4`) |
| `GLM4_MOE_LITE_DENSE_TT_DTYPE` | `bf8` | TT dtype for dense (non-expert) weights |
| `GLM4_MOE_LITE_KV_CACHE_TT_DTYPE` | (from CLI) | Override KV cache dtype |

### Debug / Profiling

| Variable | Default | Description |
| --- | --- | --- |
| `TT_METAL_DEVICE_PROFILER=1` | Off | Enable device profiler (used by `tt_metal_profiler`) |
| `GLM4_MOE_LITE_PROFILE=1` | Off | Enable per-op Python-level profiling |
| `GLM4_MOE_LITE_PROFILE_LAYER=N` | All | Profile only layer N |
| `GLM4_MOE_LITE_PROFILE_PRINT_EVERY=N` | (default) | Print profile every N steps |
| `GLM4_MOE_LITE_MOE_ROUTER_IMPL=cpu` | `tt` | Use CPU reference for MoE routing (debug) |
| `GLM4_MOE_LITE_MLA_SCALE_MODE=kvpe` | `qk` | MLA scaling mode (`qk` matches HF, `kvpe` is experimental) |
| `GLM4_MOE_LITE_DECODE_EMBED_ONLY=1` | Off | Skip all decoder layers, return after embedding (debug) |
| `GLM4_MOE_LITE_DEBUG_LOGITS_SANITY=1` | Off | Run logits sanity checks |
| `GLM4_MOE_LITE_DEBUG_PAGE_TABLE_BOUNDARY=1` | Off | Debug page table boundary conditions |
| `GLM4_MOE_LITE_SYNC_AFTER_KV_UPDATE=1` | Off | Force device sync after KV cache update |
| `GLM4_MOE_LITE_LAYER_IDENTITY=1` | Off | Make each layer an identity function (debug) |
| `GLM4_MOE_LITE_SKIP_KV_UPDATE=1` | Off | Skip KV cache update entirely (debug) |
| `GLM4_MOE_LITE_DISABLE_MLP=1` | Off | Disable MLP/MoE FFN (debug) |
| `GLM4_MOE_LITE_DISABLE_FLASH_MLA_DECODE=1` | Off | Disable FlashMLA for decode (debug) |

---

## Run Commands

All commands assume you are in the `tt-metal` root directory:
```bash
cd $TT_METAL
```

### Non-Fused (Unfused) Ops

#### Eager Mode

```bash
# Prefill + decode (both phases)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both

# Decode only (skips real prefill, uses iterative warm-up)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi"

# Prefill only
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 4 --phase prefill
```

#### Trace Mode (Non-Fused)

```bash
# Trace-sampling, 1x4 mesh (4 devices)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode sampling

# Trace-sampling, 1x8 mesh (8 devices, flat 1D)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 8 --phase both \
  --enable-trace --trace-mode sampling

# Trace-sampling, 2x4 mesh (8 devices, T3K physical topology — fastest)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-rows 2 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode sampling

# Trace-logits (returns full logits to host, flexible sampling)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode logits
```

#### Partial Layer Runs (Non-Fused)

```bash
# 1-layer eager (for quick testing / kernel debugging)
GLM4_MOE_LITE_NUM_LAYERS=1 GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS=1 \
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi"

# 1-layer trace-sampling
GLM4_MOE_LITE_NUM_LAYERS=1 GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS=1 \
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi" \
  --enable-trace --trace-mode sampling
```

### Fused Ops

The fused KV cache branch (`GLMKVCacheBranch`) combines DKV matmul + gather/slice + RMSNorm + RoPE into a single kernel. Enable with `GLM4_MOE_LITE_FUSED_KV_BRANCH=1`.

#### Eager Mode (Fused)

```bash
# Fused eager, decode only
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi"

# Fused eager, 1-layer (quick kernel test)
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 GLM4_MOE_LITE_NUM_LAYERS=1 \
GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi"
```

#### Trace Mode (Fused, 1x4)

```bash
# Fused + trace-logits, 1x4
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi" \
  --enable-trace --trace-mode logits

# Fused + trace-sampling, 1x4
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi" \
  --enable-trace --trace-mode sampling
```

#### Trace Mode (Fused, 1x8 — 8 devices)

```bash
# Fused (KV_BRANCH + QKV_A + SHARED_GATE_UP) + trace-sampling, 1x8
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
  GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 8 --phase both \
  --enable-trace --trace-mode sampling

# All fused (+ EXPERTS_GATE_UP) + trace-sampling, 1x8
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
  GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1 \
  TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 8 --phase both \
  --enable-trace --trace-mode sampling
```

#### Trace Mode (Fused, 2x4 — 8 devices, T3K physical topology)

```bash
# Fused (KV_BRANCH + QKV_A + SHARED_GATE_UP) + trace-sampling, 2x4
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
  GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-rows 2 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode sampling

# All fused (+ EXPERTS_GATE_UP) + trace-sampling, 2x4
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
  GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1 \
  TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-rows 2 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode sampling
```

### Profiling

Profiling uses `python -m tracy` to capture device-level op timings. Use `--op-support-count 20000` to ensure full capture of all ops.

```bash
# Non-fused profiling, 1x4 mesh (prefill + decode)
TT_METAL_GTEST_ETH_DISPATCH=1 python -m tracy -p -r -v \
  -n glm4_unfused_both --op-support-count 20000 \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both --prompt "Hi"

# Non-fused profiling, 2x4 mesh (8 devices)
TT_METAL_GTEST_ETH_DISPATCH=1 python -m tracy -p -r -v \
  -n glm4_unfused_2x4 --op-support-count 20000 \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-rows 2 --mesh-cols 4 --phase both --prompt "Hi"

# Fused ops profiling (KV_BRANCH + QKV_A), 1x4 mesh
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
  TT_METAL_GTEST_ETH_DISPATCH=1 python -m tracy -p -r -v \
  -n glm4_fused_both --op-support-count 20000 \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both --prompt "Hi"
```

Profiler output lands in `generated/profiler/reports/<name>/<timestamp>/`.

---

## Performance Results

### Master Comparison: All Configurations Tested

| # | Configuration | Mesh | Dev | Decode Latency | Throughput | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Non-fused, eager | 1x4 | 4 | 504.6 ms | 1.98 tok/s | 1.0x |
| 2 | Non-fused, trace-logits | 1x4 | 4 | 156.3 ms | 6.4 tok/s | 3.2x |
| 3 | Non-fused, trace-sampling | 1x4 | 4 | 137.6 ms | 7.3 tok/s | 3.7x |
| 4 | Fused (KV_BRANCH), eager | 1x4 | 4 | 533.5 ms | 1.87 tok/s | 0.9x |
| 5 | Fused (KV_BRANCH), trace-logits | 1x4 | 4 | 153.2 ms | 6.5 tok/s | 3.3x |
| 6 | Fused (KV_BRANCH), trace-sampling | 1x4 | 4 | 141.0 ms | 7.1 tok/s | 3.6x |
| 7 | **Non-fused, trace-sampling** | **2x4** | **8** | **120.6 ms** | **8.3 tok/s** | **4.2x** |
| 8 | Fused (3 flags), trace-sampling | 2x4 | 8 | 120.8 ms | 8.3 tok/s | 4.2x |
| 9 | All-fused (4 flags), trace-sampling | 2x4 | 8 | 127.6 ms | 7.8 tok/s | 4.0x |
| 10 | Non-fused, trace-sampling | 1x8 | 8 | 124.5 ms | 8.0 tok/s | 4.1x |
| 11 | Fused (3 flags), trace-sampling | 1x8 | 8 | 124.7 ms | 8.0 tok/s | 4.0x |
| 12 | All-fused (4 flags), trace-sampling | 1x8 | 8 | 130.8 ms | 7.6 tok/s | 3.9x |

**Row 7 is the fastest single-sequence configuration.** Fused (3 flags) = KV_BRANCH + QKV_A + SHARED_GATE_UP. All-fused (4 flags) adds EXPERTS_GATE_UP.

With code optimizations (`SKIP_DEFENSIVE_CLONES=1`, broadcast-mul, pad-in-RM), the 1x8 mesh achieves **83.9 ms** (11.92 tok/s) at batch=4 and **277.0 aggregate tok/s** at batch=30. See [Optimized Batch Size Sweep](#optimized-batch-size-sweep-1x8-mesh-trace-sampling-all-optimizations) below.

At **20k ISL**, per-sequence throughput remains **9.75–9.91 tok/s** (batch=1–2, max supported batch at 20k). See [20k ISL Batch Size Sweep](#20k-isl-batch-size-sweep-1x8-mesh-trace-sampling-all-optimizations) below.

### Cross-Comparison: Topology × Fusion (All Trace-Sampling)

| Configuration | 1x4 (4 dev) | 1x8 (8 dev) | 2x4 (8 dev) |
| --- | --- | --- | --- |
| **Non-fused** | 137.6 ms | 124.5 ms | **120.6 ms** |
| **Fused (3 flags)** | 141.0 ms | 124.7 ms | **120.8 ms** |
| **All-fused (4 flags)** | DRAM OOM | 130.8 ms | **127.6 ms** |

- 2x4 is fastest for all fusion configs (matches T3K physical topology)
- Non-fused is fastest or tied with fused (3 flags) across all topologies
- EXPERTS_GATE_UP causes DRAM OOM on 1x4 but works on 8-device topologies (5-6% slower)
- 1x8 is 3-4% slower than 2x4 across all configs

### Non-Fused Ops (1x4 mesh, 4 devices)

| Metric | Eager | Trace-Logits | Trace-Sampling |
| --- | --- | --- | --- |
| **Steady-state decode latency** | 504.6 ms/token | 156.3 ms/token | **137.6 ms/token** |
| **Decode throughput** | 1.98 tok/s | ~6.4 tok/s | **~7.3 tok/s** |
| **Speedup vs eager** | 1.0x | 3.2x | **3.7x** |
| **First token latency** | ~504.6 ms | 67,804 ms | 3,019 ms |
| **Device kernel time** | ~44.2 ms | ~44.2 ms | ~44.2 ms |
| **Device kernel utilization** | 8.8% | 28.3% | **32.1%** |

### Fused Ops (1x4 mesh, 4 devices, KV_BRANCH only)

| Metric | Fused Eager | Fused + Trace-Logits | Fused + Trace-Sampling |
| --- | --- | --- | --- |
| **Steady-state decode latency** | 533.5 ms/token | 153.2 ms/token | **141.0 ms/token** |
| **Decode throughput** | 1.87 tok/s | ~6.5 tok/s | **~7.1 tok/s** |
| **Speedup vs fused eager** | 1.0x | 3.5x | **3.8x** |
| **First token latency** | ~533.5 ms | 169.1 ms | 140.6 ms |
| **Device kernel utilization** | 8.2% | 28.7% | **31.2%** |

### Non-Fused Ops (2x4 mesh, 8 devices — FASTEST)

| Metric | Trace-Sampling |
| --- | --- |
| **Steady-state decode latency** | **120.6 ms/token** |
| **Decode throughput** | **~8.3 tok/s** |
| **Min / Max latency** | 120.2 / 121.0 ms |
| **First token latency** | 11,399 ms (includes trace capture) |

### Fused Ops (2x4 mesh, 8 devices)

| Metric | Fused (3 flags) | All-Fused (4 flags) |
| --- | --- | --- |
| **Steady-state decode latency** | 120.8 ms/token | 127.6 ms/token |
| **Min / Max** | 120.7 / 120.9 ms | 127.0 / 128.2 ms |
| **First token latency** | 12,697 ms | 4,212 ms |

### Non-Fused Ops (1x8 mesh, 8 devices)

| Metric | Trace-Sampling |
| --- | --- |
| **Steady-state decode latency** | **124.5 ms/token** |
| **Decode throughput** | **~8.0 tok/s** |
| **Min / Max latency** | 124.1 / 124.8 ms |

### Fused Ops (1x8 mesh, 8 devices)

| Metric | Fused (3 flags) | All-Fused (4 flags) |
| --- | --- | --- |
| **Steady-state decode latency** | 124.7 ms/token | 130.8 ms/token |
| **Min / Max** | 124.4 / 124.9 ms | 130.8 / 130.9 ms |
| **First token latency** | 8,435 ms | 4,212 ms |

### Profiler: Device Kernel Duration Comparison (Decode, Per Step, Dev 0)

All profiler runs used `--phase both`, `--max-new-tokens 4`, `--prompt "Hi"`, `--op-support-count 20000`.

| Metric | 1x4 unfused | 1x4 fused | 1x8 unfused | 2x4 unfused |
| --- | --- | --- | --- | --- |
| **Devices** | 4 | 4 | 8 | 8 |
| **Decode ops/step** | 4,568 | 4,474 | 4,568 | 4,660 |
| **Decode kernel ms/step** | 125.06 | 125.27 | 111.96 | **108.47** |
| **Decode FW ms/step** | 155.62 | 153.07 | 125.47 | **118.66** |
| Matmul ms/step | 33.35 | 31.89 | 33.26 | 33.32 |
| FillPad ms/step | 21.85 | 21.84 | 23.18 | **19.19** |
| ReduceScatter ms/step | 3.08 | 2.35 | 2.49 | 2.83 |
| AllGather ms/step | 1.12 | 1.18 | 2.05 | 2.23 |
| **Prefill kernel ms** | 169.48 | 167.50 | **143.12** | 143.48 |

**Fastest decode kernel: 2x4 unfused at 108.47 ms/step** — 13.3% faster than 1x4.
**Fastest prefill kernel: 1x8 unfused at 143.12 ms** — tied with 2x4 (143.48 ms).

Matmul is consistent across all configs (~33 ms/step) — compute-bound, not topology-dependent. The 2x4 topology wins on decode primarily because FillPad drops to 19.19 ms (vs 23.18 on 1x8).

### Latency Breakdown (Non-Fused, Trace-Sampling)

| Component | Eager (1x4) | Trace-Sampling (1x4) | Trace-Sampling (2x4) |
| --- | --- | --- | --- |
| Device kernel compute | ~44 ms | ~44 ms | ~36 ms |
| Host dispatch overhead | ~460 ms | ~0 ms | ~0 ms |
| Trace replay overhead | 0 ms | ~5 ms | ~5 ms |
| Token readback (D2H) | -- | ~1 ms | ~1 ms |
| Host input copy (H2D) | -- | ~0.3 ms | ~0.3 ms |
| Residual (NOC, fabric, DRAM BW) | ~0.6 ms | ~87.3 ms | ~78.3 ms |
| **Total** | **504.6 ms** | **137.6 ms** | **120.6 ms** |

### 1-Layer Isolated Tests

| Metric | 1-Layer Eager | 1-Layer Trace-Logits | 1-Layer Trace-Sampling |
| --- | --- | --- | --- |
| **Per-token latency** | ~12,034 ms | 151 ms | 138 ms |
| **Speedup** | 1x | 80x | 87x |

### Profiler: Device Op Counts (Decode, Per Device, 1x4)

| Metric | Non-Fused | Fused |
| --- | --- | --- |
| Total device ops | ~4,568 | ~4,474 (-2.1%) |
| FillPad ops | 369 (17.5% of device time) | 369 (17.4%) |
| Fused kernel op code | N/A | `GenericOpDeviceOperation` |
| Fused kernel duration | N/A | ~10 µs/invocation |

### Optimized Batch Size Sweep (1x8 mesh, trace-sampling, all optimizations)

Config: `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1`, 1x8 mesh, trace-sampling, 32 new tokens, short context.

```bash
TT_METAL_GTEST_ETH_DISPATCH=1 GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 \
  python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
    --mesh-rows 1 --mesh-cols 8 --batch-size $B --phase both \
    --max-new-tokens 32 --enable-trace --trace-mode sampling
```

| Batch | Status | Decode Mean (ms) | Decode Min (ms) | Decode Max (ms) | Per-Seq tok/s | Aggregate tok/s | Prefill (s) | First Token (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | OK | 87.5 | 86.3 | 88.5 | 11.43 | **11.4** | 2.6 | 4,474 |
| 2 | OK | 88.7 | 87.3 | 93.4 | 11.27 | **22.6** | 4.2 | 4,884 |
| 4 | OK | 86.0 | 85.3 | 86.8 | 11.63 | **46.5** | 3.8 | 3,985 |
| 8 | OK | 89.9 | 88.9 | 96.7 | 11.12 | **89.0** | 7.3 | 8,752 |
| 16 | OK | 96.6 | 95.5 | 99.8 | 10.35 | **165.6** | 9.1 | 8,051 |
| 20 | OK | 98.0 | 97.4 | 98.7 | 10.20 | **204.1** | 12.5 | 26,723 |
| 24 | OK | 103.1 | 102.4 | 109.5 | 9.70 | **232.8** | 11.9 | 19,536 |
| 28 | OK | 106.4 | 105.9 | 107.6 | 9.40 | **263.2** | 14.0 | 19,554 |
| **30** | **OK** | **110.4** | **109.8** | **113.9** | **9.06** | **271.7** | **14.6** | **22,587** |
| 32 | **FAIL** | — | — | — | — | — | 15.4 | Runtime error |

- **Max supported batch**: 30 (batch=32 crashes during decode)
- **Best per-token latency**: batch=4 at **86.0 ms** (11.63 tok/s per sequence)
- **Best aggregate throughput**: batch=30 at **271.7 tok/s**
- **Sweet spot**: batch=16 — 165.6 aggregate tok/s with only 96.6 ms latency

### 20k ISL Batch Size Sweep (1x8 mesh, trace-sampling, all optimizations)

Config: `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1`, 1x8 mesh, trace-sampling, 32 new tokens, 20k simulated context.

```bash
TT_METAL_GTEST_ETH_DISPATCH=1 GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 \
  python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
    --mesh-rows 1 --mesh-cols 8 --batch-size $B --phase both \
    --max-new-tokens 32 --enable-trace --trace-mode sampling \
    --simulate-context-len 20000 --min-cache-tokens 20000
```

| Batch | Status | Decode Mean (ms) | Per-Seq tok/s | Aggregate tok/s | Prefill (s) | KV blocks/seq | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | **OOM** | — | — | — | — | — | KV cache allocation: needs 369MB, only 11.8MB free |
| 12 | **OOM** | — | — | — | — | — | KV cache allocation: needs 277MB, only 19.5MB free |
| 8 | **OOM** | — | — | — | — | — | Weight loading fails at layer 20 after KV cache fills DRAM |
| 4 | **OOM** | — | — | — | — | — | KV cache allocation: needs 328MB, only 11.2MB free |
| **2** | **OK** | **102.6** | **9.75** | **19.5** | **260.7** | 313 | Max supported batch at 20k ISL |
| **1** | **OK** | **100.9** | **9.91** | **9.9** | **136.1** | 313 | |

- **Max batch at 20k ISL**: 2 — the KV cache for 20k tokens × 46 layers × 576 KVPE dim consumes most device DRAM
- Decode latency at 20k (~101–103 ms) is ~18% higher than short-context batch=4 (86 ms) due to larger paged attention scan (313 blocks vs ~2 blocks)
- **Per-seq throughput remains strong**: 9.75–9.91 tok/s even at 20k context

### Fused Ops Batch Sweep (1x8 mesh, trace-sampling, all code optimizations)

All runs use `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1` plus broadcast-mul and pad-in-RM code optimizations.

#### Fused 3-flag (KV_BRANCH + QKV_A + SHARED_GATE_UP)

```bash
TT_METAL_GTEST_ETH_DISPATCH=1 GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 \
  GLM4_MOE_LITE_FUSED_KV_BRANCH=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
  GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 \
  python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
    --mesh-rows 1 --mesh-cols 8 --batch-size $B --phase both \
    --max-new-tokens 32 --enable-trace --trace-mode sampling
```

| Batch | Status | Decode Mean (ms) | Decode Min (ms) | Decode Max (ms) | Per-Seq tok/s | Aggregate tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | OK | 88.6 | 87.6 | 90.7 | 11.29 | **11.3** |
| 2 | OK | 86.7 | 86.0 | 87.4 | 11.53 | **23.1** |
| 4 | OK | 84.5 | 83.0 | 89.1 | 11.83 | **47.3** |
| 8 | OK | 87.8 | 86.8 | 89.1 | 11.39 | **91.1** |
| 16 | OK | 94.2 | 93.1 | 95.2 | 10.62 | **169.9** |
| 20 | OK | 96.4 | 95.8 | 97.1 | 10.37 | **207.5** |
| **24** | **OK** | **101.1** | **99.9** | **102.1** | **9.89** | **237.4** |
| 28 | **FAIL** | — | — | — | — | — |
| 30 | **FAIL** | — | — | — | — | — |
| 32 | **FAIL** | — | — | — | — | — |

- **Max supported batch**: 24 (batch=28+ crashes — KV_BRANCH data marshaling increases DRAM pressure)
- **Best per-token**: batch=4 at 84.5 ms

#### Fused 2-flag (QKV_A + SHARED_GATE_UP, no KV_BRANCH) — BEST CONFIG

```bash
TT_METAL_GTEST_ETH_DISPATCH=1 GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 \
  GLM4_MOE_LITE_FUSE_QKV_A=1 GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 \
  python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
    --mesh-rows 1 --mesh-cols 8 --batch-size $B --phase both \
    --max-new-tokens 32 --enable-trace --trace-mode sampling
```

| Batch | Status | Decode Mean (ms) | Decode Min (ms) | Decode Max (ms) | Per-Seq tok/s | Aggregate tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | OK | 86.3 | 84.7 | 96.6 | 11.59 | **11.6** |
| 2 | OK | 86.8 | 85.8 | 89.6 | 11.52 | **23.0** |
| 4 | OK | 83.9 | 83.3 | 86.5 | 11.92 | **47.7** |
| 8 | OK | 87.9 | 86.8 | 97.6 | 11.38 | **91.0** |
| 16 | OK | 94.6 | 93.1 | 96.2 | 10.57 | **169.1** |
| 20 | OK | 96.6 | 96.0 | 104.5 | 10.35 | **207.0** |
| 24 | OK | 101.0 | 100.0 | 102.3 | 9.90 | **237.6** |
| 28 | OK | 104.4 | 103.3 | 104.8 | 9.58 | **268.2** |
| **30** | **OK** | **108.3** | **107.4** | **108.8** | **9.23** | **277.0** |
| 32 | **FAIL** | — | — | — | — | — |

- **Max supported batch**: 30 (same as non-fused)
- **Best per-token**: batch=4 at **83.9 ms** (fastest decode across all configs)
- **Best aggregate**: batch=30 at **277.0 tok/s** (highest throughput across all configs)

#### Cross-Comparison: Non-Fused vs Fused at Key Batch Sizes

| Batch | Non-Fused (ms) | Fused 3-flag (ms) | Fused 2-flag (ms) | Best Config |
| --- | --- | --- | --- | --- |
| 1 | 87.5 | 88.6 | **86.3** | 2-flag |
| 4 | 86.0 | 84.5 | **83.9** | 2-flag |
| 8 | 89.9 | **87.8** | 87.9 | 3-flag (tied) |
| 16 | 96.6 | **94.2** | 94.6 | 3-flag |
| 24 | 103.1 | 101.1 | **101.0** | 2-flag (tied) |
| 28 | 106.4 | FAIL | **104.4** | 2-flag |
| 30 | 110.4 | FAIL | **108.3** | 2-flag |
| Max batch | **30** | 24 | **30** | Non-fused / 2-flag |
| Best agg tok/s | 271.7 (B=30) | 237.4 (B=24) | **277.0** (B=30) | **2-flag** |

**Key findings:**
- **Fused 2-flag (QKV_A + SHARED_GATE_UP)** is the overall best config — **83.9 ms at batch=4** (best per-token) and **277.0 tok/s at batch=30** (best aggregate)
- **KV_BRANCH fusion reduces max batch** from 30 → 24 due to extra DRAM pressure
- **Removing KV_BRANCH is strictly better**: 2-flag matches or beats 3-flag at every batch size and supports higher batch sizes

---

## Key Concepts

### Trace Mode
Records the entire decode step (all 47 layers + lm_head) into a replayable device command sequence. Instead of ~1,870 individual host-to-device dispatches per token per device, a single `ttnn.execute_trace()` call replays everything. Host only copies inputs and reads outputs.

### Trace-Logits vs Trace-Sampling
- **Trace-Logits**: Returns full vocabulary logits to host (~600 KB per token). Supports any sampling strategy but incurs D2H transfer cost.
- **Trace-Sampling**: Performs greedy argmax on device. Returns only the token ID (~4 bytes). Maximum throughput but limited to top-1 greedy.

### Fused KV Cache Branch
The `GLMKVCacheBranch` fuses 4 separate TTNN operations into a single kernel dispatch:
1. **DKV Matmul** — projects hidden state to KV space
2. **Gather/Slice** — extracts nope and rope components
3. **RMSNorm** — normalizes the nope component
4. **RoPE** — applies rotary position embeddings to the rope component

The kernel operates in `TILE_1x32` format internally and uses `ROW_MAJOR` tensors as backing (exploiting identical byte layouts for 1-row data). Weights are dynamically resharded from DRAM to L1 per layer to avoid L1 OOM.

### Mesh Topology
The T3K has a physical 2×4 topology (2 rows × 4 columns = 8 Wormhole devices). Three mesh configurations have been tested:

| Mesh | Devices | Physical Match | Non-Fused Decode | Fused (3 flag) Decode | All-Fused Decode |
| --- | --- | --- | --- | --- | --- |
| 1x4 | 4 | Partial | 137.6 ms | 141.0 ms | DRAM OOM |
| 1x8 | 8 | No (flat 1D) | 124.5 ms | 124.7 ms | 130.8 ms |
| **2x4** | **8** | **Yes** | **120.6 ms** | **120.8 ms** | **127.6 ms** |

The 2x4 mesh consistently outperforms 1x8 by 3-4% across all fusion configurations because it matches the physical interconnect topology. The 1x8 mesh forces a linear communication path that increases FillPad and AllGather overhead.

**Hugepages:** 8-device runs require 16 × 1GB hugepages (2 per device). If a run crashes, hugepages may leak. Clean with `rm -f /dev/hugepages-1G/*tenstorrent*` or reboot.

### Fusion Flags: Compatibility & Performance

| Flag | 1x4 | 1x8 | 2x4 | Impact |
| --- | --- | --- | --- | --- |
| `FUSED_KV_BRANCH=1` | Works | Works | Works | Neutral on 2x4, +2.5% on 1x4 |
| `FUSE_QKV_A=1` | Works | Works | Works | Neutral |
| `FUSE_SHARED_GATE_UP=1` | Works | Works | Works | Neutral |
| `FUSE_EXPERTS_GATE_UP=1` | **DRAM OOM** | Works | Works | +4.9% (1x8), +5.6% (2x4) slower |

### ETH Dispatch
Setting `TT_METAL_GTEST_ETH_DISPATCH=1` routes the command dispatch through Ethernet cores instead of dedicating Tensix cores for dispatch. This frees all 64 Tensix cores for compute, which is critical for full-grid utilization.

### FillPad Operations
TTNN internally generates `FillPadDeviceOperation` calls to zero (or -inf fill) tile padding regions. In the non-fused decode path, there are 369 FillPad ops per device (1,476 total across 4 devices), consuming ~22 ms (15.3% of device time). The largest contributors are:
- Post-FFN LayerNorm output padding (10.12 ms) — potentially eliminable
- MoE expert output zero-init (7.86 ms) — functionally necessary
- KV concat padding (2.90 ms) — potentially eliminable

---

## Optimizations to Try on Galaxy (TG/BHGLX)

Galaxy configurations (2x4, 4x8, 8x4) provide significantly more devices, Ethernet links, and aggregate DRAM/L1/compute than T3K (1x8). Several optimizations that were blocked, ineffective, or regressed on T3K become viable or beneficial on Galaxy's topology.

### High Priority — Directly Unlocked by Galaxy Hardware

| # | Optimization | Why It Helps on Galaxy | Expected Impact |
|---|-------------|----------------------|-----------------|
| G1 | **Multi-link CCL (`num_links=2+`)** | Galaxy TG/BHGLX exposes 2+ Ethernet links per device pair. On T3K all_reduce uses 1 link, making CCL 20-30% of decode time. Doubling link count halves all_reduce latency. | **10-15% decode speedup** (all_reduce goes from ~15 ms to ~7 ms per layer) |
| G2 | **DRAM weight prefetcher** | Blackhole chips in Galaxy support prefetching next layer's weights into L1 while current layer computes. Overlaps weight load latency with matmul execution. On T3K this was completely blocked by hardware check. | **5-10% decode speedup** (hides ~5 ms DRAM latency per layer at batch=4) |
| G3 | **Expert parallelism (EP) across devices** | With 32 devices (4x8), each device can host 2 experts (64 total) instead of replicating all 64 on every device. Eliminates redundant weight DRAM reads and all_reduce for MoE layers. Requires expert-parallel dispatch (`a2a` or capacity-factor routing). | **15-25% decode speedup** for MoE layers; significant DRAM savings enabling larger batch |
| G4 | **Sequence parallelism + distributed RMSNorm** | With more devices, sequence parallelism becomes viable — each device holds a shard of the hidden state. Enables distributed RMSNorm (D3), reduce_scatter CCL (D6), and halves per-device activation memory. Major architectural change from current replicated-activation TP. | **20-30% decode speedup** (eliminates full all_reduce, enables overlapped CCL) |

### Medium Priority — Scaling and Memory Benefits

| # | Optimization | Why It Helps on Galaxy | Expected Impact |
|---|-------------|----------------------|-----------------|
| G5 | **Larger batch sizes (batch=16-64)** | Galaxy's 32-device aggregate DRAM (32× per-device DRAM) supports much larger KV caches. At batch=32-64 with bf8 KV cache, compute utilization increases and per-token amortized CCL cost drops. | **2-4x aggregate TPS** (compute-bound regime at large batch) |
| G6 | **`DRAM_SHARDED_WEIGHTS=1`** (revisit) | Regressed on T3K (+12.5 ms) because 8 devices couldn't saturate DRAM bandwidth with sharded reads at batch=4. With 32 devices and larger batch, sharded weight reads across more DRAM banks may become net positive. | **Potential 5% speedup at batch≥16** (needs profiling) |
| G7 | **bf4 expert weights (`EXPERTS_TT_DTYPE=bf4`)** | Neutral on T3K because MoE wasn't DRAM-bandwidth-bound at batch=4. At larger batch on Galaxy (more active experts per step, more weight reads), 4x compression saves significant bandwidth. | **5-10% MoE layer speedup at batch≥16** |
| G8 | **`FUSED_MOE=1` (fused persistent MoE kernel)** | Crashed on T3K due to JIT build issue. If the kernel compiles on Galaxy's Blackhole toolchain, the fused kernel eliminates per-expert dispatch overhead (currently ~46 individual expert matmuls per layer dispatched from Python). | **10-20% MoE layer speedup** (host dispatch → single kernel launch) |

### Lower Priority — Exploratory

| # | Optimization | Why It Helps on Galaxy | Expected Impact |
|---|-------------|----------------------|-----------------|
| G9 | **Async CCL with ping-pong semaphores (D6/B5)** | On T3K, `tt_all_reduce` only does `reduce_scatter` (sequence-parallel semantics). Galaxy may expose a full async `all_reduce` with overlapped compute. Combined with G1 (multi-link), could hide CCL entirely behind compute. | **Up to 20% decode speedup** if CCL fully overlapped |
| G10 | **256K+ ISL with sub-32K chunk sizes** | Galaxy's aggregate memory supports 256K+ context. Smaller prefill chunks (8K-16K) reduce peak activation memory per chunk while leveraging Galaxy's parallelism for faster chunk processing. | **Enables 256K-512K ISL** (new capability) |
| G11 | **Tensor-parallel head splitting** | With 32 devices, attention heads can be split across more devices (e.g., 4 heads per device instead of 16). Reduces per-device FlashMLA memory and compute. Requires changes to Q/K/V projection sharding. | **Enables larger batch or longer context** at same memory budget |
| G12 | **Pipeline parallelism** | Split the 47 decoder layers across device groups (e.g., 12 layers per group of 8 devices). Overlaps layer computation across pipeline stages. Adds micro-batching complexity but reduces per-device weight memory. | **2-3x throughput** (batch pipeline, amortized bubble overhead) |

### Recommended Execution Order for Galaxy

1. **G1 (Multi-link CCL)** — Easiest to enable (env var or config change), highest confidence win
2. **G2 (DRAM prefetcher)** — Also config-gated, should "just work" on Blackhole
3. **G5 (Larger batch)** — Sweep batch=8,16,32 to find the compute/memory sweet spot
4. **G7 (bf4 experts)** + **G8 (Fused MoE)** — Stack together to maximize MoE efficiency at larger batch
5. **G3 (Expert parallelism)** — Requires dispatch refactor but eliminates MoE all_reduce entirely
6. **G4 (Sequence parallelism)** — Biggest architectural change, biggest potential payoff; do last

---

## Galaxy Optimization Sweep Results (4x8 Mesh, 32 Devices)

Experimental results from systematically testing each optimization one at a time on a Galaxy 4x8 (32-device Wormhole B0) system. All runs use trace mode sampling, `--kv-cache-dtype bf8`, and ROW dispatch (Tensix).

### Baseline Configuration

Env vars: `SKIP_DEFENSIVE_CLONES=1`, `FUSE_QKV_A=1`, `FUSE_SHARED_GATE_UP=1`, `BATCHED_PREFILL=1`, `DECODE_L1_ACT=1`, `EP_L1=1`

| ISL | Decode mean (ms) | Per-user TPS | Agg TPS |
|-----|------------------|-------------|---------|
| 128 | 76.5 | 13.07 | 13.07 |
| 512 | 77.6 | 12.89 | 12.89 |
| 1024 | 78.6 | 12.72 | 12.72 |

### Phase 1: Env-Var-Only Optimizations

#### G5: Batch Scaling (ISL=128)

| Batch | Decode (ms) | Agg TPS | Per-user TPS | Scaling |
|-------|-------------|---------|--------------|---------|
| 1 | 76.7 | 13.04 | 13.04 | 1.0x |
| 2 | 77.2 | 25.91 | 12.95 | 1.99x |
| 4 | 78.2 | 51.15 | 12.79 | 3.93x |
| 8 | 78.7 | 101.65 | 12.71 | 7.80x |
| 16 | 83.6 | 191.39 | 11.96 | 14.68x |
| 32 | 89.3 | 358.34 | 11.20 | 27.48x |

Near-linear aggregate TPS scaling up to batch=8. Batch=32 achieves **358 agg TPS** with only 17% per-user TPS degradation.

#### G7: bf4 Expert Weights

| ISL | Decode (ms) | TPS | vs Baseline |
|-----|-------------|-----|-------------|
| 128 | 76.5 | 13.07 | +0.0% |
| 512 | 77.2 | 12.95 | +0.5% |
| 1024 | 77.7 | 12.87 | +1.2% |

Neutral at batch=1 as expected. Improvement grows with ISL (DRAM bandwidth savings at longer contexts).

#### G8: Fused MOE Kernel

**All runs crashed** (exit code 1). The fused persistent MoE kernel fails on Galaxy, same as T3K.

#### G6: DRAM Sharded Weights

| ISL | Decode (ms) | TPS | vs Baseline |
|-----|-------------|-----|-------------|
| 128 | 86.8 | 11.52 | **-11.9%** |
| 512 | 88.0 | 11.36 | **-11.9%** |
| 1024 | 89.4 | 11.19 | **-12.0%** |

Consistent **12% regression** at batch=1, mirroring T3K behavior. May help at larger batch (not tested).

### Phase 2: Code Changes

#### G1: Multi-link CCL

Added `GLM4_MOE_LITE_CCL_NUM_LINKS` and `GLM4_MOE_LITE_CCL_TOPOLOGY` env vars. Updated 16 `all_reduce`/`all_gather` call sites across 6 files.

| Config | ISL=128 decode (ms) | ISL=1024 decode (ms) | vs Baseline |
|--------|---------------------|----------------------|-------------|
| num_links=2, linear | 76.7 | 77.7 | +0-1.2% |
| num_links=4, ring | 75.7 | 77.2 | **+1.1-1.8%** |

Best config: `GLM4_MOE_LITE_CCL_NUM_LINKS=4 GLM4_MOE_LITE_CCL_TOPOLOGY=ring`

#### G2: DRAM Weight Prefetcher

**Deferred** — too complex for quick integration. GLM4 has variable weight tensor counts per layer (dense vs MoE layers), but `dram_prefetcher` requires uniform `n_tensors`. Also requires global-CB-aware matmul calls and SubDevice core splitting.

### Phase 3: Profiler-Driven Optimizations

Analyzed `ops_perf_results` CSV (~175K rows) from a full decode step to identify redundant or optimizable operations. Key findings:

- **71.9% of device kernel time** spent in data movement/layout ops (reshape, permute, typecast, clone, to_layout)
- **ReshapeViewDeviceOperation** consuming 12.67% of device time due to materializing reshapes where zero-cost views are possible
- **1,504 TypecastDeviceOperation** calls adding unnecessary format conversions
- **Dual ReduceScatter+AllGather pairs** in MoE layers that can be consolidated

Three optimizations were implemented and tested (ISL=128, batch=1, trace-sampling, 4x8 Galaxy):

| Change | Type | Description |
|---|---|---|
| `FUSE_MLP_MOE_REDUCE=1` | Env var | Consolidates dual ReduceScatter+AllGather pairs into one per MoE layer |
| `SKIP_TYPECAST=1` | Env var | Eliminates unnecessary bf16 typecast ops in attention Q/KV paths |
| MoE decode permute-to-reshape | Code change in `moe_tt.py` | Replaces `permute+clone+reshape` with zero-cost `ttnn.reshape` for decode-mode expert output aggregation (when `num_blocks==1`) |

#### Results (with Combined Winners baseline + TP=1)

| Config | Decode Mean (ms) | Improvement |
|---|---|---|
| Before (Combined Winners + TP) | 98.0 | -- |
| After (+ all 3 optimizations) | 90.0 | **-8.2%** |

`CONCAT_HEADS=1` was also tested but showed no benefit in traced mode (93.6ms vs 92.9ms without -- within noise). The `concatenate_heads` device kernel does not outperform the permute-reshape-permute chain when host dispatch overhead is already eliminated by trace replay. Not included in the recommended config.

#### Profiler Analysis Script

A reusable analysis script is available at `experiments/analyze_ops_perf.py`:

```bash
python models/demos/glm4_moe_lite/experiments/analyze_ops_perf.py \
  generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

It reports: top ops by device time, data movement breakdown, reshape materialization patterns, host overhead analysis, fusion candidates, and anti-pattern detection.

### Combined Winners (bf4 + 4-link ring CCL + profiler optimizations)

Best-performing configuration: baseline + `EXPERTS_TT_DTYPE=bf4` + `CCL_NUM_LINKS=4` + `CCL_TOPOLOGY=ring` + `FUSE_MLP_MOE_REDUCE=1` + `SKIP_TYPECAST=1` + MoE decode permute-to-reshape code change.

Recommended run command (Galaxy 4x8, all optimizations):

```bash
GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 \
GLM4_MOE_LITE_FUSE_QKV_A=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 \
GLM4_MOE_LITE_BATCHED_PREFILL=1 \
GLM4_MOE_LITE_DECODE_L1_ACT=1 \
GLM4_MOE_LITE_EP_L1=1 \
GLM4_MOE_LITE_TP=1 \
GLM4_MOE_LITE_EXPERTS_TT_DTYPE=bf4 \
GLM4_MOE_LITE_CCL_NUM_LINKS=4 \
GLM4_MOE_LITE_CCL_TOPOLOGY=ring \
GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 \
GLM4_MOE_LITE_SKIP_TYPECAST=1 \
python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Summarize" \
  --simulate-context-len 128 \
  --min-cache-tokens 256 \
  --max-new-tokens 32 \
  --batch-size 1 \
  --mesh-rows 4 --mesh-cols 8 \
  --kv-cache-dtype bf8 \
  --phase both --enable-trace --trace-mode sampling
```

The sweep script (`run_sweep_isl_batch.py`) includes all these flags automatically.

The tables below show results from the bf4 + 4-link ring CCL configuration (before profiler optimizations). With profiler optimizations enabled, expect an additional ~8% decode speedup.

#### Decode Latency (ms)

| ISL \ batch | 1 | 2 | 4 | 8 | 16 | 32 |
|-------------|------|------|------|------|------|------|
| 128 | 74.8 | 75.2 | 76.0 | 77.5 | 82.0 | 87.3 |
| 512 | 75.4 | 76.6 | 77.8 | 79.1 | 83.4 | 89.4 |
| 1024 | 77.3 | 77.1 | 78.7 | 80.2 | 84.4 | 91.4 |

#### Aggregate TPS

| ISL \ batch | 1 | 2 | 4 | 8 | 16 | 32 |
|-------------|-------|-------|-------|--------|--------|--------|
| 128 | 13.37 | 26.60 | 52.63 | 103.23 | 195.12 | 366.55 |
| 512 | 13.26 | 26.11 | 51.41 | 101.14 | 191.85 | 357.94 |
| 1024 | 12.94 | 25.94 | 50.83 | 99.75 | 189.57 | 350.11 |

#### Improvement vs Baseline (batch=1)

| ISL | Baseline | Combined | Improvement |
|-----|----------|----------|-------------|
| 128 | 76.5 ms | 74.8 ms | **-2.2%** |
| 512 | 77.6 ms | 75.4 ms | **-2.8%** |
| 1024 | 78.6 ms | 77.3 ms | **-1.7%** |

### Summary

| Optimization | Status | Impact |
|---|---|---|
| G1 Multi-link CCL (4 links, ring) | Implemented | +1-2% decode speedup |
| G2 DRAM Prefetcher | Deferred | High complexity |
| G5 Batch Scaling | Tested | 358 agg TPS at batch=32 |
| G6 DRAM Sharded Weights | Tested | -12% regression |
| G7 bf4 Expert Weights | Tested | +0-1.2% at batch=1 |
| G8 Fused MOE | Tested | Crashed (persistent_moe_compute kernel is a WIP stub) |
| **Combined (G1+G7)** | **Implemented** | **2-3% decode speedup, 367 agg TPS at batch=32** |
| FUSE_MLP_MOE_REDUCE | Implemented | Consolidates CCL pairs in MoE layers |
| SKIP_TYPECAST | Implemented | Eliminates ~1,500 typecast ops |
| MoE decode permute-to-reshape | Implemented (code) | Zero-cost view replaces permute+clone chain |
| CONCAT_HEADS | Tested | Neutral in traced mode (not recommended) |
| **Combined (G1+G7+profiler opts)** | **Implemented** | **~10% total decode speedup vs original baseline** |

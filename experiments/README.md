# GLM-4.7-Flash on Tenstorrent: Commands, Configuration & Performance

**Model:** zai-org/GLM-4.7-Flash (47 layers, MLA attention, MoE, 4.7B params)
**Hardware:** T3K (8 Wormhole devices total); tested with mesh shapes 1x4 (4 devices), 1x8 (8 devices), and 2x4 (8 devices, matching T3K physical topology)
**Dispatch:** `DispatchCoreType.ETH` (all 64 Tensix cores per device available for compute)

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

**Script:** `models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py`

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
| `GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1` | Off | Fuse MLP + MoE reduce step |
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
| 10 | Fused (3 flags), trace-sampling | 1x8 | 8 | 124.7 ms | 8.0 tok/s | 4.0x |
| 11 | All-fused (4 flags), trace-sampling | 1x8 | 8 | 130.8 ms | 7.6 tok/s | 3.9x |

**Row 7 is the fastest configuration.** Fused (3 flags) = KV_BRANCH + QKV_A + SHARED_GATE_UP. All-fused (4 flags) adds EXPERTS_GATE_UP.

### Cross-Comparison: Topology × Fusion (All Trace-Sampling)

| Configuration | 1x4 (4 dev) | 1x8 (8 dev) | 2x4 (8 dev) |
| --- | --- | --- | --- |
| **Non-fused** | 137.6 ms | — | **120.6 ms** |
| **Fused (3 flags)** | 141.0 ms | 124.7 ms | **120.8 ms** |
| **All-fused (4 flags)** | DRAM OOM | 130.8 ms | **127.6 ms** |

- 2x4 is fastest for all fusion configs (matches T3K physical topology)
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
| 1x8 | 8 | No (flat 1D) | — | 124.7 ms | 130.8 ms |
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

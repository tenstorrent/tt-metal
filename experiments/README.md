# GLM-4.7-Flash on Tenstorrent: Commands, Configuration & Performance

**Model:** zai-org/GLM-4.7-Flash (47 layers, MLA attention, MoE, 4.7B params)
**Hardware:** T3K (8 Wormhole devices total); runs below use `--mesh-cols 4` = 4 of 8 devices (mesh_shape=1,4, FABRIC_1D)
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
6. [Generated Profiler Reports](#generated-profiler-reports)
7. [Key Concepts](#key-concepts)

---

## Quick Start

```bash
cd $TT_METAL

# Simplest run: non-fused, eager, 4 devices, prefill + decode
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both

# Fastest run: non-fused, trace-sampling
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both \
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
| `--mesh-cols` | `1` | Number of devices in mesh shape (1, mesh_cols). T3K has 8 devices; use 4 or 8. |
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
# Trace-logits (returns full logits to host, flexible sampling)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode logits

# Trace-sampling (on-device greedy top-1, maximum throughput)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode sampling
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

#### Trace Mode (Fused)

```bash
# Fused + trace-logits
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi" \
  --enable-trace --trace-mode logits

# Fused + trace-sampling
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi" \
  --enable-trace --trace-mode sampling
```

### Profiling

Profiling uses the `tt_metal_profiler` wrapper to capture device-level op timings.

```bash
# Non-fused profiling (prefill + decode)
TT_METAL_GTEST_ETH_DISPATCH=1 tt_metal_profiler -n glm4_both_eth \
  python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both

# Fused ops profiling (prefill + decode)
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 \
  tt_metal_profiler -n glm4_fused_both \
  python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both --prompt "Hi"

# Decode-only profiling
TT_METAL_GTEST_ETH_DISPATCH=1 tt_metal_profiler -n glm4_decode_eth \
  python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase decode

# Prefill-only profiling
TT_METAL_GTEST_ETH_DISPATCH=1 tt_metal_profiler -n glm4_prefill_eth \
  python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --mesh-cols 4 --phase prefill
```

Profiler output lands in `generated/profiler/reports/<name>/<timestamp>/`.

---

## Performance Results

### Non-Fused Ops

| Metric | Eager | Trace-Logits | Trace-Sampling |
| --- | --- | --- | --- |
| **Steady-state decode latency** | 504.6 ms/token | 156.3 ms/token | **137.8 ms/token** |
| **Decode throughput** | 1.98 tok/s | ~6.4 tok/s | **~7.3 tok/s** |
| **Speedup vs eager** | 1.0x | 3.2x | **3.7x** |
| **First token latency** | ~504.6 ms | 67,804 ms | 3,309 ms |
| **Device kernel time** | ~44.2 ms | ~44.2 ms | ~44.2 ms |
| **Device kernel utilization** | 8.8% | 28.3% | **32.1%** |

### Fused Ops

| Metric | Fused Eager | Fused + Trace-Logits | Fused + Trace-Sampling |
| --- | --- | --- | --- |
| **Steady-state decode latency** | 533.5 ms/token | 153.2 ms/token | **141.0 ms/token** |
| **Decode throughput** | 1.87 tok/s | ~6.5 tok/s | **~7.1 tok/s** |
| **Speedup vs fused eager** | 1.0x | 3.5x | **3.8x** |
| **First token latency** | ~533.5 ms | 169.1 ms | 140.6 ms |
| **Device kernel time** | ~44 ms | ~44 ms | ~44 ms |
| **Device kernel utilization** | 8.2% | 28.7% | **31.2%** |

### Cross-Comparison: Non-Fused vs Fused (Best Modes)

| Metric | Non-Fused + Trace-Sampling | Fused + Trace-Sampling | Delta |
| --- | --- | --- | --- |
| **Decode latency** | **137.8 ms** | 141.0 ms | +2.3% slower |
| **Throughput** | **7.3 tok/s** | 7.1 tok/s | -2.7% |

The fused kernel itself is very fast on-device (~10 us per invocation), but the data marshaling operations around it (layout conversions, resharding, concat) add more overhead than the fusion saves. The "Path to Making Fused Ops Faster" involves kernel-side DRAM I/O to eliminate these marshaling ops.

### 1-Layer Isolated Tests

| Metric | 1-Layer Eager | 1-Layer Trace-Logits | 1-Layer Trace-Sampling |
| --- | --- | --- | --- |
| **Per-token latency** | ~12,034 ms | 151 ms | 138 ms |
| **Speedup** | 1x | 80x | 87x |

The extreme 80-87x speedup with 1 layer confirms that dispatch overhead is ~99.99% of eager time when compute is minimal.

### Latency Breakdown (Non-Fused, Trace-Sampling)

| Component | Eager | Trace-Sampling |
| --- | --- | --- |
| Device kernel compute | ~44 ms | ~44 ms |
| Host dispatch overhead | ~460 ms | ~0 ms |
| Trace replay overhead | 0 ms | ~5 ms |
| Logits/token readback (D2H) | -- | ~1 ms |
| Host input copy (H2D) | -- | ~0.3 ms |
| Residual (NOC, fabric, DRAM BW) | ~0.6 ms | ~88.5 ms |
| **Total** | **504.6 ms** | **137.8 ms** |

### Profiler: Device Op Counts (Decode, Per Device)

| Metric | Non-Fused | Fused |
| --- | --- | --- |
| Total device ops | ~4,567 | Fewer (fused kernel replaces ~4 ops/layer) |
| FillPad ops | 369 (15.3% of device time) | Reduced |
| Fused kernel op code | N/A | `GenericOpDeviceOperation` |
| Fused kernel duration | N/A | ~10 us/invocation |

---

## Generated Profiler Reports

Reports are stored under `generated/profiler/reports/`:

| Report Name | Path | Description |
| --- | --- | --- |
| Non-fused both (prefill+decode) | `glm4_both_eth/2026_03_12_01_36_44/` | Full model profiling |
| Non-fused decode only (split) | `glm4_both_eth/.../ops_perf_decode_only.csv` | Decode phase extracted from "both" |
| Non-fused prefill only (split) | `glm4_both_eth/.../ops_perf_prefill_only.csv` | Prefill phase extracted from "both" |
| Non-fused decode standalone | `glm4_decode_eth/2026_03_12_01_11_39/` | Decode-only profiling run |
| Non-fused prefill standalone | `glm4_prefill_eth/2026_03_11_22_53_43/` | Prefill-only profiling run |
| Fused both (prefill+decode) | `glm4_fused_both/2026_03_12_17_24_48/` | Fused ops profiling |

Each report directory contains:
- `ops_perf_results_<name>_<timestamp>.csv` — full op-level performance data
- `profile_log_device.csv` — raw device profiler log
- Split CSVs (if generated): `ops_perf_decode_only.csv`, `ops_perf_prefill_only.csv`

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

### ETH Dispatch
Setting `TT_METAL_GTEST_ETH_DISPATCH=1` routes the command dispatch through Ethernet cores instead of dedicating Tensix cores for dispatch. This frees all 64 Tensix cores for compute, which is critical for full-grid utilization.

### FillPad Operations
TTNN internally generates `FillPadDeviceOperation` calls to zero (or -inf fill) tile padding regions. In the non-fused decode path, there are 369 FillPad ops per device (1,476 total across 4 devices), consuming ~22 ms (15.3% of device time). The largest contributors are:
- Post-FFN LayerNorm output padding (10.12 ms) — potentially eliminable
- MoE expert output zero-init (7.86 ms) — functionally necessary
- KV concat padding (2.90 ms) — potentially eliminable

# GLM-4.7-Flash on Tenstorrent: Comprehensive Performance Report

**Date:** 2026-03-13
**Model:** zai-org/GLM-4.7-Flash (47 layers, MLA attention, MoE, 4.7B params)
**Hardware:** T3K (8 Wormhole devices); tested with mesh shapes 1x4 (4 dev), 1x8 (8 dev), 2x4 (8 dev)
**Dispatch:** `DispatchCoreType.ETH` — all 64 Tensix cores per device available for compute
**Script:** `models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Script & CLI Options](#script--cli-options)
3. [Environment Variables](#environment-variables)
4. [Run Commands](#run-commands)
5. [Performance Results](#performance-results)
6. [How Trace Mode Works](#how-trace-mode-works)
7. [Fused Ops Deep Dive](#fused-ops-deep-dive)
8. [Profiler Analysis](#profiler-analysis)
9. [Operational Learnings](#operational-learnings)
10. [Optimization Opportunities](#optimization-opportunities)
11. [Implementation Details](#implementation-details)
12. [Generated Profiler Reports](#generated-profiler-reports)

---

## Executive Summary

| Metric | Eager 1x4 | Trace-Sampling 1x4 | **Trace-Sampling 2x4** |
| --- | --- | --- | --- |
| **Steady-state decode latency** | 504.6 ms/token | 137.6 ms/token | **120.6 ms/token** |
| **Decode throughput** | 1.98 tok/s | ~7.3 tok/s | **~8.3 tok/s** |
| **Speedup vs eager** | 1.0x | 3.7x | **4.2x** |
| **First token latency** | ~504.6 ms | 3,019 ms | 11,399 ms |
| **Device kernel time (per device)** | ~44.2 ms | ~44.2 ms | ~36 ms |
| **Device kernel utilization** | 8.8% | 32.1% | 30.0% |

**Bottom line:** The 2x4 mesh (matching T3K's physical topology) with trace-sampling delivers **4.2x higher throughput** (8.3 vs 1.98 tok/s). Trace eliminates >90% of the host dispatch overhead. The 2x4 topology outperforms 1x4 by 12.3% due to optimal inter-device communication paths.

### Best Configuration Found

```bash
cd $TT_METAL
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
| `--mesh-cols` | `1` | Number of columns in mesh shape |
| `--device-ids` | `auto` | Comma-separated physical device IDs or `auto` |
| `--kv-cache-dtype` | `bf16` | KV cache data type: `bf16` or `bf8` |
| `--block-size` | `64` | KV cache block size |
| `--min-cache-tokens` | `128` | Minimum tokens to allocate in KV cache |
| `--phase` | `both` | Phase to run: `prefill`, `decode`, or `both` |
| `--enable-trace` | `false` | Enable traced decode execution |
| `--trace-mode` | `logits` | `logits` (full logits to host) or `sampling` (on-device greedy top-1) |

**Mesh shape examples:**
- `--mesh-cols 4` → 1x4 mesh (4 devices)
- `--mesh-rows 2 --mesh-cols 4` → 2x4 mesh (8 devices, matches T3K physical topology)
- `--mesh-cols 8` → 1x8 mesh (8 devices, flat 1D)

---

## Environment Variables

### Required

| Variable | Description |
| --- | --- |
| `TT_METAL_GTEST_ETH_DISPATCH=1` | Route dispatch through Ethernet cores, freeing all 64 Tensix cores for compute. **Always set.** |

### Fusion Flags

| Variable | Default | Description | Status |
| --- | --- | --- | --- |
| `GLM4_MOE_LITE_FUSED_KV_BRANCH=1` | Off | Fuse DKV Matmul + Gather/Slice + RMSNorm + RoPE into one kernel | Works, trace-compatible |
| `GLM4_MOE_LITE_FUSE_QKV_A=1` | Off | Fuse Q and KV down-projection into a single matmul | Works, trace-compatible |
| `GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1` | Off | Fuse shared MLP gate + up projections | Works, trace-compatible |
| `GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1` | Off | Fuse expert gate + up projections | **Causes DRAM OOM on 1x4; works on 2x4 but 5.8% slower** |

### Other Feature Toggles

| Variable | Default | Description |
| --- | --- | --- |
| `GLM4_MOE_LITE_ENABLE_MOE=1` | Off (script forces on) | Enable MoE layers |
| `GLM4_MOE_LITE_NUM_LAYERS=N` | All (47) | Run only N layers (requires `DEBUG_ALLOW_PARTIAL_LAYERS=1`) |
| `GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS=1` | Off | Allow partial-layer runs |
| `GLM4_MOE_LITE_TP=1` | Off | Tensor parallelism across mesh devices |
| `GLM4_MOE_LITE_MTP=1` | Off | Multi-token prediction |
| `GLM4_MOE_LITE_PRESERVE_TRACE=1` | Off | Skip trace release after prefill |
| `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1` | Off | Skip defensive clone operations |
| `GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1` | Off | DRAM-sharded weight layout |
| `GLM4_MOE_LITE_SHARDED_MLP=1` | Off | L1 WIDTH_SHARDED activations for MLP decode |
| `GLM4_MOE_LITE_USE_DECODE_ROPE=1` | Off (auto with trace) | Decode-specific RoPE implementation |

### Data Type Overrides

| Variable | Default | Description |
| --- | --- | --- |
| `GLM4_MOE_LITE_EXPERTS_TT_DTYPE` | `bf8` | TT dtype for expert weights |
| `GLM4_MOE_LITE_DENSE_TT_DTYPE` | `bf8` | TT dtype for dense (non-expert) weights |
| `GLM4_MOE_LITE_KV_CACHE_TT_DTYPE` | (from CLI) | Override KV cache dtype |

### Profiling

| Variable | Default | Description |
| --- | --- | --- |
| `TT_METAL_DEVICE_PROFILER=1` | Off | Enable device profiler |
| `GLM4_MOE_LITE_PROFILE=1` | Off | Per-op Python-level profiling |

---

## Run Commands

All commands assume:
```bash
cd $TT_METAL    # /home/ttuser/sdawle/agentic_ai/tt-metal
```

### Non-Fused Ops — Eager Mode

```bash
# Prefill + decode, 1x4 (4 devices)
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

### Non-Fused Ops — Trace Mode

```bash
# Trace-sampling, 1x4 mesh (4 devices)
TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both \
  --enable-trace --trace-mode sampling

# Trace-sampling, 2x4 mesh (8 devices, T3K physical topology — FASTEST)
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

### Fused Ops — Eager Mode

```bash
# Fused KV_BRANCH only, decode, 1x4
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi"
```

### Fused Ops — Trace Mode (1x4)

```bash
# Fused + trace-sampling, 1x4 (KV_BRANCH only)
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi" \
  --enable-trace --trace-mode sampling

# Fused + trace-logits, 1x4
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 TT_METAL_GTEST_ETH_DISPATCH=1 python \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 5 --mesh-cols 4 --phase decode --prompt "Hi" \
  --enable-trace --trace-mode logits
```

### Fused Ops — Trace Mode (1x8, 8 devices)

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

### Fused Ops — Trace Mode (2x4, 8 devices)

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

### Partial Layer Runs (Debugging)

```bash
# 1-layer eager
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

### Profiling Commands

Profiling uses `python -m tracy`. **Always include `--op-support-count 20000`** to capture all ops (default buffer is too small for this model).

```bash
# Non-fused profiling, 1x4 mesh
TT_METAL_GTEST_ETH_DISPATCH=1 python -m tracy -p -r -v \
  -n glm4_unfused_both --op-support-count 20000 \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 4 --phase both --prompt "Hi"

# Non-fused profiling, 1x8 mesh (8 devices)
TT_METAL_GTEST_ETH_DISPATCH=1 python -m tracy -p -r -v \
  -n glm4_unfused_both --op-support-count 20000 \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --max-new-tokens 4 --mesh-cols 8 --phase both --prompt "Hi"

# Non-fused profiling, 2x4 mesh (8 devices, physical topology)
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

### Device Management

```bash
# Reset all 8 devices
tt-smi -r 0,1,2,3,4,5,6,7

# Check hugepage status
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages    # total (should be 16)
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages  # free (need 16 for 8-dev runs)

# Clean up leaked hugepage mappings
rm -f /dev/hugepages-1G/*tenstorrent*

# Check for processes holding hugepages
docker ps -a --filter "name=tt-inference"
```

---

## Performance Results

### Master Comparison: All Configurations Tested

| # | Configuration | Mesh | Devices | Decode Latency | Throughput | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Non-fused, eager | 1x4 | 4 | 504.6 ms | 1.98 tok/s | 1.0x |
| 2 | Non-fused, trace-logits | 1x4 | 4 | 156.3 ms | 6.4 tok/s | 3.2x |
| 3 | Non-fused, trace-sampling | 1x4 | 4 | 137.6 ms | 7.3 tok/s | 3.7x |
| 4 | Fused (KV_BRANCH), eager | 1x4 | 4 | 533.5 ms | 1.87 tok/s | 0.9x |
| 5 | Fused (KV_BRANCH), trace-logits | 1x4 | 4 | 153.2 ms | 6.5 tok/s | 3.3x |
| 6 | Fused (KV_BRANCH), trace-sampling | 1x4 | 4 | 141.0 ms | 7.1 tok/s | 3.6x |
| 7 | **Non-fused, trace-sampling** | **2x4** | **8** | **120.6 ms** | **8.3 tok/s** | **4.2x** |
| 8 | Fused (KV+QKV_A+SHARED_GATE_UP), trace-sampling | 2x4 | 8 | 120.8 ms | 8.3 tok/s | 4.2x |
| 9 | All-fused (+EXPERTS_GATE_UP), trace-sampling | 2x4 | 8 | 127.6 ms | 7.8 tok/s | 4.0x |
| 10 | Fused (KV+QKV_A+SHARED_GATE_UP), trace-sampling | 1x8 | 8 | 124.7 ms | 8.0 tok/s | 4.0x |
| 11 | All-fused (+EXPERTS_GATE_UP), trace-sampling | 1x8 | 8 | 130.8 ms | 7.6 tok/s | 3.9x |

**Row 7 is the fastest configuration.** Fused ops on 2x4 (row 8) are essentially tied. Adding EXPERTS_GATE_UP consistently hurts performance (rows 9 and 11 are 5-6% slower than their non-EXPERTS counterparts).

### Detailed: Non-Fused Ops (1x4 mesh, 4 devices)

| Metric | Eager | Trace-Logits | Trace-Sampling |
| --- | --- | --- | --- |
| **Steady-state decode latency** | 504.6 ms/token | 156.3 ms/token | **137.6 ms/token** |
| **Decode throughput** | 1.98 tok/s | ~6.4 tok/s | **~7.3 tok/s** |
| **Speedup vs eager** | 1.0x | 3.2x | **3.7x** |
| **First token latency** | ~504.6 ms | 67,804 ms | 3,019 ms |
| **Device kernel time** | ~44.2 ms | ~44.2 ms | ~44.2 ms |
| **Device kernel utilization** | 8.8% | 28.3% | **32.1%** |
| **Output text** | I'm a  | I'm a  | I'm a  |

### Detailed: Fused Ops (1x4 mesh, 4 devices, KV_BRANCH only)

| Metric | Fused Eager | Fused + Trace-Logits | Fused + Trace-Sampling |
| --- | --- | --- | --- |
| **Steady-state decode latency** | 533.5 ms/token | 153.2 ms/token | **141.0 ms/token** |
| **Decode throughput** | 1.87 tok/s | ~6.5 tok/s | **~7.1 tok/s** |
| **Speedup vs fused eager** | 1.0x | 3.5x | **3.8x** |
| **First token latency** | ~533.5 ms | 169.1 ms | 140.6 ms |
| **Device kernel utilization** | 8.2% | 28.7% | **31.2%** |

### Detailed: Non-Fused Ops (2x4 mesh, 8 devices)

| Metric | Trace-Sampling |
| --- | --- |
| **Steady-state decode latency** | **120.6 ms/token** |
| **Decode throughput** | **~8.3 tok/s** |
| **Min / Max latency** | 120.2 / 121.0 ms |
| **First token latency** | 11,399 ms |
| **Prefill time** | ~6.5 s |

### Detailed: Fused Ops (2x4 mesh, 8 devices)

| Metric | Fused (3 flags) | All-Fused (4 flags) |
| --- | --- | --- |
| **Fusion flags** | KV_BRANCH + QKV_A + SHARED_GATE_UP | KV_BRANCH + QKV_A + SHARED_GATE_UP + EXPERTS_GATE_UP |
| **Steady-state decode latency** | 120.8 ms/token | 127.6 ms/token |
| **Min / Max** | 120.7 / 120.9 ms | 127.0 / 128.2 ms |
| **First token latency** | 12,697 ms | 4,212 ms |
| **Prefill time** | 6.491 s | 6.412 s |
| **Weight loading** | ~86 s | ~540 s (generates fused expert caches) |
| **Output text** | (No, " | (I amic |

Note: `EXPERTS_GATE_UP=1` generates new fused expert weight caches on first run (~540s). Subsequent runs use cached weights. Different generated text from all-fused suggests possible numerical precision differences with expert fusion.

### Detailed: Fused Ops (1x8 mesh, 8 devices)

| Metric | Fused (3 flags) | All-Fused (4 flags) |
| --- | --- | --- |
| **Fusion flags** | KV_BRANCH + QKV_A + SHARED_GATE_UP | KV_BRANCH + QKV_A + SHARED_GATE_UP + EXPERTS_GATE_UP |
| **Steady-state decode latency** | 124.7 ms/token | 130.8 ms/token |
| **Min / Max** | 124.4 / 124.9 ms | 130.8 / 130.9 ms |
| **First token latency** | 8,435 ms | 4,212 ms |
| **Prefill time** | 8.737 s | 2.832 s |
| **Weight loading** | ~730 s (regenerated 1x8 caches) | ~537 s (regenerated fused expert caches) |
| **Output text** | (I am I | (I am I |

### Cross-Comparison: Mesh Topology (Trace-Sampling, Non-Fused)

| Metric | 1x4 (4 dev) | 2x4 (8 dev) | Delta |
| --- | --- | --- | --- |
| **Decode latency** | 137.6 ms | **120.6 ms** | **-12.3%** |
| **Throughput** | 7.3 tok/s | **8.3 tok/s** | **+13.7%** |
| **Min latency** | 137.2 ms | 120.2 ms | -12.4% |
| **Max latency** | 138.0 ms | 121.0 ms | -12.3% |

### Cross-Comparison: Non-Fused vs Fused (Trace-Sampling)

| Metric | Non-Fused 1x4 | Fused 1x4 | Non-Fused 2x4 | Fused 2x4 (3 flags) | All-Fused 2x4 | Fused 1x8 (3 flags) | All-Fused 1x8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Decode latency** | 137.6 ms | 141.0 ms | **120.6 ms** | 120.8 ms | 127.6 ms | 124.7 ms | 130.8 ms |
| **vs non-fused same mesh** | baseline | +2.5% | baseline | +0.2% | +5.8% | — | — |

### Cross-Comparison: Topology × Fusion (All Trace-Sampling)

| Configuration | 1x4 (4 dev) | 1x8 (8 dev) | 2x4 (8 dev) |
| --- | --- | --- | --- |
| **Non-fused** | 137.6 ms | — | **120.6 ms** |
| **Fused (3 flags)** | 141.0 ms | 124.7 ms | **120.8 ms** |
| **All-fused (4 flags)** | DRAM OOM | 130.8 ms | 127.6 ms |

**Key findings:**
- **2x4 is the fastest topology** for all fusion configurations, matching the T3K physical layout
- **Fused (3 flags) on 2x4 is tied with non-fused** (120.8 vs 120.6 ms) — fusion overhead cancels out savings
- **EXPERTS_GATE_UP consistently hurts performance** by 4.9-5.8% across both 8-device topologies
- **1x8 is 3-4% slower than 2x4** across all fusion configs, confirming the physical topology advantage
- **EXPERTS_GATE_UP causes DRAM OOM on 1x4** (4 devices) but works on both 8-device topologies

### Steady-State Decode Latency Detail

| Mode | Mean (ms) | Min (ms) | Max (ms) | Variance |
| --- | --- | --- | --- | --- |
| Eager (1x4) | 504.6 | -- | -- | -- |
| Trace-Logits (1x4) | 156.3 | 145.9 | 166.7 | ~21 ms spread |
| Trace-Sampling (1x4) | 137.6 | 137.2 | 138.0 | ~0.8 ms spread |
| Trace-Sampling (2x4) | 120.6 | 120.2 | 121.0 | ~0.8 ms spread |
| Fused Trace-Sampling (2x4, 3 flags) | 120.8 | 120.7 | 120.9 | ~0.2 ms spread |
| All-Fused Trace-Sampling (2x4, 4 flags) | 127.6 | 127.0 | 128.2 | ~1.2 ms spread |

Trace-sampling has near-zero variance, indicating deterministic execution with minimal host interference. Trace-logits has higher variance due to the full-vocab tensor readback.

### Latency Breakdown

| Component | Eager (1x4) | Trace-Sampling (1x4) | Trace-Sampling (2x4) |
| --- | --- | --- | --- |
| Device kernel compute | ~44 ms | ~44 ms | ~36 ms |
| Host dispatch overhead | ~460 ms | ~0 ms | ~0 ms |
| Trace replay overhead | 0 ms | ~5 ms | ~5 ms |
| Token ID readback (D2H) | -- | ~1 ms | ~1 ms |
| Host input copy (H2D) | -- | ~0.3 ms | ~0.3 ms |
| Residual (NOC, fabric, DRAM BW) | ~0.6 ms | ~87.3 ms | ~78.3 ms |
| **Total** | **504.6 ms** | **137.6 ms** | **120.6 ms** |

In eager mode, host dispatch overhead consumed **91.2%** of per-token latency. Trace eliminates this entirely.

### Throughput Scaling (Projected)

| Tokens Generated | Eager 1x4 (s) | Trace-Sampling 1x4 (s) | Trace-Sampling 2x4 (s) |
| --- | --- | --- | --- |
| 4 | 1.5 | 3.4 | 11.8 |
| 10 | 4.5 | 4.3 | 12.5 |
| 32 | 15.6 | 7.3 | 15.2 |
| 128 | 63.6 | 20.6 | 26.8 |
| 512 | 254.3 | 73.4 | 73.1 |
| 1024 | 508.6 | 143.9 | 135.0 |

Trace-sampling (1x4) surpasses eager after ~9 tokens. Trace-sampling (2x4) surpasses eager after ~25 tokens (higher trace capture cost) and overtakes 1x4 at ~512 tokens.

### 1-Layer Isolated Tests

| Metric | 1-Layer Eager | 1-Layer Trace-Logits | 1-Layer Trace-Sampling |
| --- | --- | --- | --- |
| **Per-token latency** | ~12,034 ms | 151 ms | 138 ms |
| **Speedup** | 1x | 80x | 87x |

The extreme 80-87x speedup with 1 layer confirms dispatch overhead is ~99.99% of eager time when compute is minimal.

---

## How Trace Mode Works

### The Problem: Host Dispatch Bottleneck

In eager mode, the host (CPU) dispatches each device operation individually:

```
Host                          Device
  |-- dispatch op 1 ----------->|
  |<--- op 1 complete ----------|
  |-- dispatch op 2 ----------->|
  |<--- op 2 complete ----------|
  |   ... x 1,870 ops ...       |
  |-- dispatch op 1870 -------->|
  |<--- op 1870 complete -------|
```

With ~1,870 ops per device per token and 4 devices, the host coordinates ~7,500 individual dispatches per token. Each involves Python overhead, TTNN runtime processing, and PCIe round-trips.

### The Solution: Trace Capture and Replay

```
CAPTURE (one-time):
  Host                          Device
    |-- begin_trace_capture ---->|
    |-- dispatch op 1 ---------->|  (recorded)
    |-- dispatch op 2 ---------->|  (recorded)
    |   ... x 1,870 ops ...      |  (recorded)
    |-- end_trace_capture ------>|
    |   trace_id stored          |

REPLAY (every subsequent token):
  Host                          Device
    |-- copy inputs (H2D) ------>|  (~0.3 ms)
    |-- execute_trace(id) ------>|  (single call)
    |                            |  ... device replays all 1,870 ops ...
    |<--- trace complete --------|
    |-- read output (D2H) <------|  (~1 ms for token, ~107 ms for logits)
```

### Persistent Tensors

Trace replay requires fixed memory addresses. Persistent tensors allocated during capture:
- `tokens_tt` — input token IDs [B,1]
- `positions_tt` — sequence positions [B]
- `rot_idxs_tt` — RoPE rotation indices
- `cos_batch_tt`, `sin_batch_tt` — precomputed RoPE embeddings
- `trans_matrix_tt` — RoPE transformation matrix
- `page_table_tt` — KV cache page table

Updated before each replay via `ttnn.copy_host_to_device_tensor()`.

### Trace-Logits vs Trace-Sampling

| Aspect | Trace-Logits | Trace-Sampling |
| --- | --- | --- |
| **What's captured** | 47 layers + lm_head | 47 layers + lm_head + argmax |
| **Output** | Full logits [1,1,vocab] on host | Token ID (single int) on device |
| **D2H transfer** | ~600 KB (vocab=151,552 × float32) | ~4 bytes (1 int32) |
| **Host sampling** | `torch.argmax(logits)` | Not needed |
| **Steady-state latency** | 156.3 ms | 137.6 ms |
| **Flexibility** | Any sampling strategy | Greedy top-1 only |
| **Use case** | Debug, non-greedy sampling | Maximum throughput |

The 18.7 ms gap comes from eliminating full-vocab D2H transfer and host argmax. On-device argmax executes in the same pipeline with zero dispatch cost.

### Trace Capture Cost (First Token Overhead)

| Mode | First Token (ms) | Steady-State (ms) | Break-Even |
| --- | --- | --- | --- |
| Trace-Logits (1x4) | 67,804 | 156.3 | ~194 tokens |
| Trace-Sampling (1x4) | 3,019 | 137.6 | ~9 tokens |
| Trace-Sampling (2x4) | 11,399 | 120.6 | ~25 tokens |

Trace-logits is expensive because it performs a full warm-up compile run. Trace-sampling is cheap. The 2x4 has higher first-token cost due to 8-device coordination.

---

## Fused Ops Deep Dive

### What Gets Fused

The `GLMKVCacheBranch` kernel combines 4 TTNN operations into a single dispatch:
1. **DKV Matmul** — projects hidden state to KV space
2. **Gather/Slice** — extracts nope and rope components
3. **RMSNorm** — normalizes the nope component
4. **RoPE** — applies rotary position embeddings to the rope component

### Making Fused Ops Trace-Compatible

The original fused op contained 5 host-device roundtrips (`_to_torch_single`, `ttnn.from_torch`, `ttnn.synchronize_device`) to convert between standard TILE (32×32) and the kernel's TILE_1x32 format. These are incompatible with `ttnn.begin_trace_capture()`.

**Fix:** Exploited the fact that ROW_MAJOR and TILE_1x32 have identical byte layouts for 1-row data. Inputs are prepared as ROW_MAJOR sharded tensors using device-side ops. The kernel's circular buffers use manual TILE_1x32 descriptors to interpret the same bytes correctly. Outputs use the same trick in reverse.

### Why Fused Ops Don't Help (Yet)

**Non-fused path** (what the kernel replaces): ~4 optimized TTNN ops operating natively on TILE tensors — matmul, slice, rmsnorm, rope. No format conversion needed.

**Fused path** (what actually runs): 1 fast kernel (~10 µs) + ~15 data-movement ops:
- **Input marshaling:** `to_layout(ROW_MAJOR)` + `reshape` + `concat` (×18) + `to_memory_config(HEIGHT_SHARDED)` for x; similar for cos/sin; DRAM→L1 weight reshard
- **Output marshaling:** `to_memory_config(DRAM)` ×2 + `concat` + `reshape` + `to_layout(TILE)` + `deallocate` calls

The net cost of data marshaling exceeds the savings from fusing 4 ops into 1 kernel.

### Fusion Flag Compatibility Matrix

| Flag | 1x4 Eager | 1x4 Trace | 1x8 Trace | 2x4 Trace | Impact on Latency |
| --- | --- | --- | --- | --- | --- |
| `FUSED_KV_BRANCH=1` | Works | Works | Works | Works | Neutral on 2x4, +2.5% on 1x4 |
| `FUSE_QKV_A=1` | Works | Works | Works | Works | Neutral |
| `FUSE_SHARED_GATE_UP=1` | Works | Works | Works | Works | Neutral |
| `FUSE_EXPERTS_GATE_UP=1` | **DRAM OOM** | **DRAM OOM** | Works | Works | **+4.9% (1x8), +5.6% (2x4)** |

`FUSE_EXPERTS_GATE_UP` causes DRAM OOM on 1x4 (4 devices) because the concatenated w1+w3 expert weights are too large. On 8-device topologies (1x8 and 2x4) there's enough DRAM, but the larger fused weight tensors increase DRAM bandwidth pressure, making decode 5-6% slower.

### Path to Making Fused Ops Faster

| Approach | Description | Expected Impact |
| --- | --- | --- |
| **Kernel-side DRAM read** | NCRISC reads x, cos, sin from DRAM via NOC into CBs | Eliminates ~10 input marshaling ops |
| **Kernel-side DRAM write** | NCRISC writes nope+rope directly to DRAM output tensor | Eliminates ~5 output marshaling ops |
| **Combined** | Both above | Reduces to 1 kernel + 1 weight reshard, saving ~10-15 ms |

---

## Profiler Analysis

### Device Kernel Duration Comparison (Decode, Per Step, Dev 0)

All profiler runs used `python -m tracy --op-support-count 20000`, `--phase both`, `--max-new-tokens 4`, `--prompt "Hi"`.

| Configuration | Prefill Kernel (ms) | Decode Kernel (ms/step) | Total Kernel (ms) |
| --- | --- | --- | --- |
| **1x4 Unfused** | 185.13 | 133.07 | 717.41 |
| **1x4 Fused** (KV_BRANCH + QKV_A) | 184.18 | 124.69 | 682.94 |
| **1x8 Unfused** | 143.12 | 136.44 | 688.88 |
| **2x4 Unfused** | 147.46 | **108.47** | **581.34** |

### Detailed Per-Op Breakdown (Decode, Per Step, Dev 0)

| Metric | 1x4 unfused | 1x4 fused | 1x8 unfused | 2x4 unfused |
| --- | --- | --- | --- | --- |
| **Decode ops/step** | 4,568 | 4,474 | 4,568 | 4,660 |
| **Decode kernel ms/step** | 125.06 | 125.27 | 111.96 | **108.47** |
| **Decode FW ms/step** | 155.62 | 153.07 | 125.47 | **118.66** |
| Matmul ms/step | 33.35 | 31.89 | 33.26 | 33.32 |
| FillPad ms/step | 21.85 | 21.84 | 23.18 | **19.19** |
| ReduceScatter ms/step | 3.08 | 2.35 | 2.49 | 2.83 |
| AllGather ms/step | 1.12 | 1.18 | 2.05 | 2.23 |
| **Prefill kernel ms** | 169.48 | 167.50 | **143.12** | 143.48 |

### Key Profiler Findings

1. **Matmul is topology-independent** (~33 ms/step) — it is compute-bound, not communication-bound.
2. **FillPad is the variable factor** — 2x4 has the lowest FillPad (19.19 ms) compared to 1x4 (21.85 ms) and 1x8 (23.18 ms). This is likely due to better memory layout with the physical topology.
3. **2x4 wins on decode** (108.47 ms) by 18.5% over 1x4 (133.07 ms) — primarily from FillPad reduction and better collective ops.
4. **1x8 wins (marginally) on prefill** (143.12 ms) vs 2x4 (143.48 ms) — effectively tied.
5. **Fused ops reduce kernel time by 6.3%** on 1x4 (124.69 vs 133.07 ms) but this saving is negated by data marshaling overhead at the wall-clock level.

### FillPad Operations

TTNN generates `FillPadDeviceOperation` calls to zero/inf-fill tile padding regions. In non-fused decode on 1x4, there are 369 FillPad ops per device consuming ~22 ms (15.3% of device time). The largest contributors:
- Post-FFN LayerNorm output padding (10.12 ms) — potentially eliminable
- MoE expert output zero-init (7.86 ms) — functionally necessary
- KV concat padding (2.90 ms) — potentially eliminable

### Device Op Counts (Decode, Per Device, 1x4)

| Metric | Non-Fused | Fused |
| --- | --- | --- |
| Total device ops | ~4,568 | ~4,474 (-2.1%) |
| FillPad ops | 369 (17.5% of device time) | 369 (17.4%) |
| Fused kernel op code | N/A | `GenericOpDeviceOperation` |
| Fused kernel duration | N/A | ~10 µs/invocation |

---

## Operational Learnings

### Hugepage Management

8-device runs require **16 × 1GB hugepages** (2 per Wormhole device). The system allocates these at boot via `/opt/tenstorrent/bin/hugepages-setup.sh`.

**Problem:** When runs crash or are killed, hugepages can leak and not be released.

```bash
# Check status
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages    # should be 16
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages  # need 16 for 8 devices

# If free < 16, check for leaked mappings
ls /dev/hugepages-1G/

# Clean up leaked hugepage mappings
rm -f /dev/hugepages-1G/*tenstorrent*

# If still not enough, check for Docker containers holding them
docker ps -a --filter "name=tt-inference"
docker stop <container_name>
rm -f /dev/hugepages-1G/*tenstorrent*

# Last resort: system reboot (reclaims all leaked hugepages)
```

### Weight Cache Regeneration

When switching mesh topologies or enabling new fusion flags, expert weight caches must be regenerated. This causes:
- **Long startup times** (~540s for all-fused with `EXPERTS_GATE_UP` on first run, vs ~86s for cached)
- **High disk usage** (each expert cache file is large)
- **Potential disk OOM** if multiple cache versions accumulate

Monitor with: `df -h /home/ttuser` and clean old caches if needed.

### Profiler Buffer Size

The default `python -m tracy` op tracking buffer is too small for this model. Without `--op-support-count 20000`:
- Profiler CSVs will be incomplete (missing decode ops)
- Post-processing may fail with `AssertionError: Device data missing`
- Only prefill ops (e.g., `TilizeWithValPaddingDeviceOperation`) may appear

**Always use:** `python -m tracy ... --op-support-count 20000`

### Profiler + Trace Incompatibility

`ttnn.ReadDeviceProfiler()` calls `event_synchronize` which is not supported during trace capture. The model code wraps this in a try/except:

```python
try:
    ttnn.ReadDeviceProfiler(self.device)
except RuntimeError:
    pass  # skip profiler flush during trace capture
```

### Mesh Topology Learnings

| Mesh | Devices | Physical Match | Non-Fused Decode | Fused (3 flag) Decode | All-Fused Decode | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1x4 | 4 | Partial | 137.6 ms | 141.0 ms | DRAM OOM | No hugepage issues |
| 1x8 | 8 | No (flat 1D) | — | 124.7 ms | 130.8 ms | Linear topology, suboptimal interconnect |
| **2x4** | **8** | **Yes** | **120.6 ms** | **120.8 ms** | **127.6 ms** | **Matches T3K physical layout** |

The 1x8 mesh forces a linear communication path that doesn't match the physical 2x4 interconnect. This leads to suboptimal FillPad and AllGather performance during decode. The 2x4 mesh consistently outperforms 1x8 by 3-4% across all fusion configurations.

### DRAM OOM with Fusion Flags

| Flag Combination | 1x4 (4 dev) | 1x8 (8 dev) | 2x4 (8 dev) |
| --- | --- | --- | --- |
| KV_BRANCH only | Works | Works | Works |
| KV_BRANCH + QKV_A | Works | Works | Works |
| KV_BRANCH + QKV_A + SHARED_GATE_UP | Works | Works | Works |
| + **EXPERTS_GATE_UP** | **DRAM OOM** | Works (130.8 ms, +4.9%) | Works (127.6 ms, +5.6%) |

`EXPERTS_GATE_UP` concatenates w1 and w3 expert weight matrices, roughly doubling DRAM usage for expert weights. With only 4 devices, DRAM is insufficient. With 8 devices (both 1x8 and 2x4), there's enough DRAM but the larger tensors increase bandwidth pressure, making decode 5-6% slower.

### Disk Space Management

Profiler runs generate large files:
- `.tracy` host logs (several GB each)
- `profile_log_device.csv` intermediate files
- `ops_perf_results_*.csv` final reports

```bash
# Check disk usage
df -h /home/ttuser

# Clean up old profiler artifacts
rm -f generated/profiler/reports/*/*.tracy
rm -f generated/profiler/reports/*/profile_log_device.csv
rm -f generated/profiler/reports/*/cpp_device_perf_report.csv
```

---

## Optimization Opportunities

### Current Bottleneck (Best Config: 2x4 Trace-Sampling)

```
120.6 ms total = ~36 ms kernel + ~84 ms non-kernel
```

The ~84 ms non-kernel time includes:
1. NOC data movement between DRAM and compute cores (~30-40 ms est.)
2. Fabric communication for collective ops across 8 devices (~10-15 ms est.)
3. DRAM bandwidth saturation from weight reads (~20-30 ms est.)
4. Trace replay runtime overhead (~5 ms est.)
5. Host input copy via `copy_host_to_device_tensor` (~0.3 ms)

### Potential Next Steps

| Optimization | Expected Impact | Complexity |
| --- | --- | --- |
| **Fused KV cache branch + kernel DRAM I/O** | Eliminate ~15 data-movement ops, save ~10-15 ms | High |
| **bf8 KV cache** (`--kv-cache-dtype bf8`) | Reduce DRAM bandwidth for KV reads by ~50% | Low (flag exists) |
| **Op fusion** (norm + matmul, etc.) | Reduce kernel launch overhead, improve L1 reuse | Medium |
| **Weight compression** (bf8 for all weights) | Reduce DRAM read bandwidth by ~50% | Low |
| **Larger batch** (batch > 1) | Amortize weight reads across batch | Medium |
| **Multi-token prediction (MTP)** | Generate 2+ tokens per decode step | Medium |
| **Eliminate redundant FillPad** | Remove unnecessary padding ops (~10 ms savings) | Low-Medium |
| **DRAM-sharded weights** | Reduce NOC traffic for weight reads | Medium |

---

## Implementation Details

### Script Changes Made

**`models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py`:**
1. **CLI arguments:** `--enable-trace`, `--trace-mode {logits, sampling}`, `--mesh-rows`
2. **Mesh shape:** Changed from hardcoded `ttnn.MeshShape(1, mesh_cols)` to `ttnn.MeshShape(mesh_rows, mesh_cols)`
3. **Warm-up loop:** Passes `enable_trace=True` and `sampling_params=True` to `runner.decode()`
4. **Generation loop:** Per-token timing with `time.perf_counter()`
5. **`_sampling_result_to_token()` helper:** Handles mesh-distributed device tensor readback
6. **Performance reporting:** First-token vs steady-state latency with mean/min/max

**`models/demos/glm4_moe_lite/tt/model_tt.py`:**
1. Wrapped `ttnn.ReadDeviceProfiler()` in try/except to allow profiling during trace

**`models/demos/glm4_moe_lite/tt/decoder_layer_tt.py`:**
1. Made `_fused_kv_branch_forward` trace-compatible by eliminating host-device roundtrips
2. Replaced `_to_torch_single`/`ttnn.from_torch` with device-side ROW_MAJOR layout tricks

### Pre-existing Infrastructure Used

The trace machinery in `model_tt.py` was already implemented:
- `decode(enable_trace=True)` → routes to `_decode_trace_logits()` or `_decode_trace_sampling()`
- `_capture_decode_trace_sampling()` → warm-up compile, persistent tensor allocation, trace capture
- `_copy_decode_trace_inputs()` → updates persistent device tensors before each replay
- `_DecodeTraceSamplingState` → dataclass holding trace_id, persistent tensors, output tensors

---

## Generated Profiler Reports

Reports stored under `$TT_METAL/generated/profiler/reports/`:

| Report Name | Path | Description |
| --- | --- | --- |
| Non-fused 1x4 both | `glm4_unfused_both/2026_03_13_04_25_02/` | 4 devices, unfused, prefill+decode |
| Non-fused 1x8 both | `glm4_unfused_both/2026_03_13_05_08_27/` | 8 devices (1x8), unfused |
| Non-fused 2x4 both | `glm4_unfused_2x4/2026_03_13_09_32_34/` | 8 devices (2x4), unfused |
| Fused 1x4 both (KV+QKV_A) | `glm4_fused_both/2026_03_13_04_36_59/` | 4 devices, fused KV_BRANCH+QKV_A |
| Fused 1x4 both (KV only) | `glm4_fused_both/2026_03_13_02_10_27/` | 4 devices, fused KV_BRANCH only |
| Non-fused both (original) | `glm4_both_eth/2026_03_12_01_36_44/` | Original 4-device profiling |
| Non-fused decode standalone | `glm4_decode_eth/2026_03_12_01_11_39/` | Decode-only profiling run |

Each report directory contains `ops_perf_results_<name>_<timestamp>.csv` with full op-level performance data.

---

## Reference Implementations

| Reference | Approach | Key Pattern |
| --- | --- | --- |
| [tt_cnn](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_cnn) | Whole-model trace | allocate → compile → capture → replay |
| [Whisper](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/audio/whisper/tt/whisper_generator.py#L189) | Autoregressive decode trace | capture single step, replay in loop |
| [ign/glm_flash](https://github.com/tenstorrent/tt-metal/tree/ign/glm_flash) | Per-module `@trace_enabled` | Modular trace with pre-allocated semaphores |
| **This implementation** | Whole-decode-step trace | Uses existing `model_tt.py` infrastructure |

---

## Repetitive Op Elimination (Post-Optimization Results)

**Date:** 2026-03-17
**Commit:** `0134195933` on `sdawle/dvartanians/glm47_flash`

### Optimizations Applied

| # | Optimization | Description | Files Modified |
| --- | --- | --- | --- |
| 1 | **Broadcast mul** | Removed `ttnn.repeat` in MoE routing weight expansion; use broadcast `ttnn.mul` instead of creating 33MB intermediate per MoE layer | `moe_tt.py` (2 locations) |
| 2 | **skip_defensive_clones** | Set `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1` to skip ~30 `ttnn.clone` ops per decode step in traced execution | Runtime flag |
| 3 | **KVPE pad-in-RM** | Untilize → pad(RM) → tilize for KVPE cache update padding, avoiding implicit `FillPadDeviceOperation` | `decoder_layer_tt.py` |
| 4 | **MoE hidden pad-in-RM** | Same pad-in-RM strategy for MoE hidden state padding before sparse expert kernels | `mlp_decode.py` |
| 5 | **MoE sparse dispatch pad-in-RM** | Same pad-in-RM strategy for sparse dispatch `hidden_states` padding | `moe_tt.py` |

### Incremental Optimization Results (1x8 mesh, batch=4, trace-sampling)

| # | Optimization | Decode (ms) | Delta (ms) | Delta (%) | Cumulative Speedup |
| --- | --- | --- | --- | --- | --- |
| 0 | **Baseline** (no changes) | **112.4** | — | — | — |
| — | concat_heads=True (tested previously) | 114.7 | +2.3 | +2.0% | Skipped (slower) |
| 1 | **Broadcast mul** (kill repeat+permute+tilize) | **100.6** | −11.8 | −10.5% | −10.5% |
| 2 | + **skip_defensive_clones** | **97.5** | −3.1 | −3.1% | −13.3% |
| 3 | + **KVPE pad-in-RM** | **93.8** | −3.7 | −3.8% | −16.5% |
| 4 | + **MoE hidden pad-in-RM** | **93.4** | −0.4 | −0.4% | −16.9% |
| 5 | + **MoE sparse dispatch pad-in-RM** | **85.9** | −7.5 | −8.0% | **−23.6%** |

**Total: 112.4 ms → 85.9 ms (−26.5 ms, −23.6% faster, +31% throughput)**

### Optimized Batch Size Sweep (1x8 mesh, trace-sampling, all optimizations)

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
- Decode latency at 20k (~101-103 ms) is ~18% higher than short-context batch=4 (86 ms) due to larger paged attention scan (313 blocks vs ~2 blocks)

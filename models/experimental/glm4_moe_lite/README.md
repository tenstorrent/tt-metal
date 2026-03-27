# GLM-4.7-Flash on Tenstorrent: Commands, Configuration & Performance

**Model:** zai-org/GLM-4.7-Flash (47 layers, MLA attention, MoE, 4.7B params)
**Hardware (T3K):** 8 Wormhole devices; tested with mesh shapes 1x4 (4 devices), 1x8 (8 devices), and 2x4 (8 devices, matching T3K physical topology)
**Hardware (Galaxy):** 32 Wormhole B0 devices; 4x8 mesh
**Dispatch:** `DispatchCoreType.ETH` (all 64 Tensix cores per device available for compute)

**Current best decode latency (Galaxy, batch=1):** ~74.8 ms @ ISL=128, ~75.4 ms @ ISL=512, ~77.3 ms @ ISL=1024 (~13.4 tok/s)
**Current best aggregate TPS (Galaxy, batch=32):** ~367 tok/s @ ISL=128, ~358 tok/s @ ISL=512, ~350 tok/s @ ISL=1024

## Directory Structure

```
models/experimental/glm4_moe_lite/
├── tt/                        # Core model implementation
│   ├── model_tt.py            #   Top-level runner (prefill, decode, trace)
│   ├── decoder_layer_tt.py    #   Decoder layer (attention + MLP/MoE)
│   ├── attention_decode.py    #   Decode attention (Q proj, FlashMLA, output)
│   ├── mlp_decode.py          #   Shared MLP + MoE forwarding
│   ├── moe_tt.py              #   MoE (sparse, dense, packed)
│   ├── layer_weights.py       #   Weight conversion (torch → TT)
│   ├── config.py              #   Hyperparameters
│   ├── runtime_config.py      #   Env-var feature flags
│   ├── weights.py             #   Weight loading / caching
│   └── ...                    #   linear_helpers, embedding, trace, refs, vLLM
├── fused_ops/                 # Custom device kernels
│   ├── kv_cache_branch/       #   DKV + RMSNorm + RoPE fused kernel
│   └── pre_sdpa/              #   Pre-SDPA fused kernel
├── scripts/                   # Run & sweep scripts
│   ├── debug_run_full_tt_greedy.py   # Single-run debug / benchmark
│   └── run_sweep_isl_batch.py        # ISL × batch sweep
└── tests/                     # PCC & integration tests (17 files)
```

> `experiments/` (sweep results, plots, profiler scripts) is git-ignored — local only.

---

## Table of Contents

1. [Quick Start](#quick-start)
   - [Greedy Debug Script (single run)](#greedy-debug-script-single-run)
   - [Batch & ISL Sweep](#batch--isl-sweep)
2. [Script & CLI Options](#script--cli-options)
3. [Environment Variables](#environment-variables)
   - [Required](#required)
   - [Feature Toggles](#feature-toggles)
   - [Performance Tuning](#performance-tuning)
   - [Data Type Overrides](#data-type-overrides)
   - [Debug / Profiling](#debug--profiling)

---

## Quick Start
```bash
# After you have TT_METAL built (./build_metal.sh)
# And python env created (./create_venv.sh)

cd tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

### Greedy Debug Script (single run)

```bash
cd $TT_METAL_HOME && \
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
python models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Summarize" \
  --simulate-context-len 128 \
  --min-cache-tokens 256 \
  --max-new-tokens 32 \
  --batch-size 1 \
  --mesh-rows 4 --mesh-cols 8 \
  --kv-cache-dtype bf8 \
  --phase both \
  --enable-trace --trace-mode sampling
```

### Batch & ISL Sweep

```bash
cd $TT_METAL_HOME && \
mkdir -p models/experimental/glm4_moe_lite/experiments/g1_multilink_4_ring_isl_sweep && \
GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 \
GLM4_MOE_LITE_FUSE_QKV_A=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 \
GLM4_MOE_LITE_BATCHED_PREFILL=1 \
GLM4_MOE_LITE_DECODE_L1_ACT=1 \
GLM4_MOE_LITE_EP_L1=1 \
GLM4_MOE_LITE_CCL_NUM_LINKS=4 \
GLM4_MOE_LITE_CCL_TOPOLOGY=ring \
python models/experimental/glm4_moe_lite/scripts/run_sweep_isl_batch.py \
  --out-dir models/experimental/glm4_moe_lite/experiments/g1_multilink_4_ring_isl_sweep \
  --timeout 1200 \
  --isl 128 512 1024 2048 4096 8192 16384 32768 65536 131072 \
  --batch 1
```

> **Note:** The sweep script (`run_sweep_isl_batch.py`) already sets `EXPERTS_TT_DTYPE=bf4`, `TP=1`, `FUSE_MLP_MOE_REDUCE=1`, and `SKIP_TYPECAST=1` internally — no need to pass them on the command line.

---

## Script & CLI Options

**Script:** `scripts/debug_run_full_tt_greedy.py` (relative to this directory; from repo root: `models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py`)

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

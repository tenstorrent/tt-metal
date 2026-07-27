# GLM-4.7-Flash on Tenstorrent: Commands, Configuration & Performance

**Model:** zai-org/GLM-4.7-Flash (47 layers, MLA attention, MoE, 4.7B params)
**Hardware (T3K):** 8 Wormhole devices; tested with mesh shapes 1x4 (4 devices), 1x8 (8 devices), and 2x4 (8 devices, matching T3K physical topology)
**Hardware (Galaxy):** 32 Wormhole B0 devices; 4x8 mesh
**Dispatch:** `DispatchCoreType.ETH` (all 64 Tensix cores per device available for compute)

**Current best decode latency (Galaxy, batch=1):** **51.3 ms** @ ISL=128 (19.49 tok/s), 52.5 ms @ ISL=512 (19.05 tok/s), 53.4 ms @ ISL=1024 (18.73 tok/s)
**Current best aggregate TPS (Galaxy):** **449 tok/s** @ batch=32/ISL=128, 431 tok/s @ batch=32/ISL=512, 408 tok/s @ batch=32/ISL=1024; peak **592 tok/s** @ batch=128/ISL=128

> Measured on a 32-chip WH Galaxy (4x8 mesh) with traced sampling decode and the
> default flag set below. This supersedes the earlier published 74.8 ms / 13.37 tok/s
> baseline — a **31.4% latency reduction** and **+45.8% throughput**. See
> [Performance](#performance) for the full ISL x batch matrix and the optimization history.

## Directory Structure

```
models/experimental/glm4_moe_lite/
├── tt/                        # Core model implementation
│   ├── model_tt.py            #   Top-level runner (prefill, decode, trace)
│   ├── decoder_layer_tt.py    #   Decoder layer (attention + MLP/MoE, sharded RMSNorm)
│   ├── attention_decode.py    #   Decode attention (Q proj, FlashMLA, sub-batched KV update)
│   ├── mlp_decode.py          #   Shared MLP + MoE forwarding
│   ├── moe_tt.py              #   MoE (sparse, dense, packed), router, fused epilogue
│   ├── layer0_tt.py           #   Layer-0 / page-table helpers
│   ├── layer_weights.py       #   Weight conversion (torch → TT), dtype selection
│   ├── linear_helpers.py      #   Matmul program-config helpers
│   ├── config.py              #   Hyperparameters
│   ├── runtime_config.py      #   Env-var feature flags (frozen dataclass)
│   ├── perf_defaults.py       #   Validated winning flag set (single source of truth)
│   ├── prefetcher_setup.py    #   GlobalCB DRAM-prefetcher SubDevice setup (WIP, unwired)
│   ├── weights.py             #   Weight loading / caching
│   ├── generator_vllm.py      #   vLLM integration
│   └── ...                    #   embedding, debug runtime, torch reference impls
├── fused_ops/                 # Custom device kernels — NONE currently on the hot path
│   ├── kv_cache_branch/       #   DKV + RMSNorm + RoPE (gated off: numerically incorrect)
│   ├── pre_sdpa/              #   Pre-SDPA (not referenced by the model)
│   └── q_concat_transpose/    #   Q concat + transpose (not referenced by the model)
├── scripts/                   # Run, sweep, and kernel-check scripts
│   ├── debug_run_full_tt_greedy.py   # Single-run debug / benchmark
│   ├── run_sweep_isl_batch.py        # ISL × batch sweep
│   ├── run_pre_sdpa_kernel_check.py  # Standalone pre-SDPA kernel check
│   └── run_fused_kv_branch_check.py  # Standalone fused-KV-branch kernel check
└── tests/                     # PCC & integration tests (16 files)
```

> Multi-token prediction lives in `model_tt._mtp_forward_eager`, not a separate module.
>
> `experiments/` is a **local scratch directory** for sweep output, profiler runs, and
> plots. It is gitignored in full — summarise results here in the README instead. It
> interacts badly with the root `.gitignore` (`*.log` and `*.csv` are dropped globally),
> so `git add experiments/...` will silently commit plots while omitting the CSV they
> came from.

---

## Table of Contents

1. [Quick Start](#quick-start)
   - [Greedy Debug Script (single run)](#greedy-debug-script-single-run)
   - [Batch & ISL Sweep](#batch--isl-sweep)
2. [Performance](#performance)
3. [Script & CLI Options](#script--cli-options)
4. [Environment Variables](#environment-variables)
   - [Default-On Optimizations](#default-on-optimizations)
   - [Feature Toggles](#feature-toggles)
   - [Performance Tuning](#performance-tuning)
   - [Data Type Overrides](#data-type-overrides)
   - [Known-Bad / Neutral Experiments](#known-bad--neutral-experiments)
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

**The validated winning flag set is now applied automatically** by both entry-point
scripts, so no env prefix is needed. Any flag you set explicitly still wins
(`os.environ.setdefault` semantics in the runner; per-run `env` in the sweep).

```bash
cd $TT_METAL_HOME && \
python models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Summarize" \
  --simulate-context-len 128 \
  --min-cache-tokens 256 \
  --max-new-tokens 128 \
  --batch-size 1 \
  --mesh-rows 4 --mesh-cols 8 \
  --kv-cache-dtype bf16 \
  --phase both \
  --enable-trace --trace-mode sampling
```

To A/B a single optimization, set just that flag — e.g. to measure the sharded
RMSNorm win, run the above with and without `GLM4_MOE_LITE_SHARDED_NORM=0`.

### Batch & ISL Sweep

```bash
cd $TT_METAL_HOME && \
mkdir -p models/experimental/glm4_moe_lite/experiments/my_sweep && \
python models/experimental/glm4_moe_lite/scripts/run_sweep_isl_batch.py \
  --out-dir models/experimental/glm4_moe_lite/experiments/my_sweep \
  --timeout 1200 \
  --isl 128 512 1024 2048 4096 8192 16384 32768 65536 \
  --batch 1 4 8 16 32 64 128
```

> **Note:** `run_sweep_isl_batch.py` shells out to `debug_run_full_tt_greedy.py` per
> cell. It **hard-assigns** the flag set from `tt/perf_defaults.py` in the child `env`
> (rather than defaulting it) and pins the known-bad/neutral experiments off, so an
> inherited environment cannot make a sweep CSV describe a configuration other than
> the one it claims to. It writes `sweep_results.csv`, `sweep_table.md`, and plots to
> `--out-dir`, and records per-cell OOM detail rather than aborting the sweep.

---

## Performance

Measured on 32-chip WH Galaxy (4x8 mesh), traced sampling decode, default flag set.
Reproduce with the sweep command above. The raw run that these figures come from was
captured on 2026-07-24; its plots and rendered table are in git history at commit
`30c7605a` (`experiments/sweep_isl_batch_complete_20260724/`, since removed — that
directory is local scratch and no longer tracked).

### Decode latency, ms/token (batch=1)

| ISL | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|---|---|---|---|
| ms/token | 51.3 | 52.5 | 53.4 | 53.9 | 55.0 | 57.2 | 61.6 | 70.6 | 87.9 |
| tok/s | 19.49 | 19.05 | 18.73 | 18.55 | 18.18 | 17.48 | 16.23 | 14.16 | 11.38 |

### Aggregate TPS at ISL=128 (total tokens/sec across all sequences)

| batch | 1 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| tok/s | 19.5 | 73.3 | 139.9 | 256.8 | 449.4 | 529.8 | 591.5 |

Scaling is near-linear to batch=32 (23.1x on 32x the sequences), then flattens as
the decode step goes compute-bound: 71.2 ms at bs=32 vs 216.4 ms at bs=128.

**Current capacity wall:** longer contexts do not reach the full batch range. The
unreachable cells fail in the DRAM bank allocator
(`tt_metal/impl/allocator/bank_manager.cpp:439`) during KV-cache allocation — a
`batch x ISL` KV footprint limit, not a compute limit. The reachable maximum batch
is 64 at ISL=512, 32 at ISL=1024, 16 at ISL=2048, 8 at ISL=4096, 4 at ISL=8192, and
1 at ISL>=16384. Reducing `--min-cache-tokens` or using `--kv-cache-dtype bf8`
extends the reachable region (bf8 KV has known accuracy degradation — see the dtype
table). The sweep records per-cell failure detail in its `sweep_results.csv` /
`sweep_table.md` output rather than aborting, so re-running it regenerates the full
matrix.

### Optimization history (batch=1 decode)

Each step below was individually validated on-device and coherence-gated
(real-prompt greedy output checked for correctness), newest last:

| Change | Effect |
|---|---|
| BF8 dense weights + production flags default-on | 71.8 → 66.8 ms (−7%) |
| Sub-batch paged KV-cache update | bs=1 neutral; fixes bs>16 L1 CB overflow, unblocks bs≥32 |
| Drop redundant MoE layout ops (reduce dispatch) | 68.9 → 65.8 ms @ ISL512 (−4.5%) |
| Width-sharded multi-core RMSNorm | 65.8 → 63.1 ms @ ISL512 (−4.1%) |
| Fused collective epilogue + buffered MoE all-reduce + fused down routing scale | 63.1 → 60.4 ms @ ISL512 (−4.3%) |
| Fused router (`topk_router_gpt`) default-on + decode/router refinements | → 52.5 ms @ ISL512, **51.3 ms @ ISL128** |

Versus the earlier published baseline (74.8 ms / 13.37 tok/s): **−31.4% latency,
+45.8% throughput**.

### Next steps

The largest remaining decode bucket is matmul (~39.5% of step time at ~20% M=1
bandwidth efficiency, per pre-optimization profiling). The GlobalCB DRAM weight
prefetcher is the planned attack — `tt/prefetcher_setup.py` has the validated
SubDevice + GlobalCB foundation, but it is **not wired into the model** and has
zero runtime effect today. The per-op integration spec is kept as a local (untracked)
working document, `GLOBALCB_PREFETCH_PORT_PLAN.md`, in this directory; it is also in
git history at commit `b43bc892`. **Re-profile the current 51.3 ms configuration before
acting on those op shares** — they were captured on the pre-optimization stack.

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
| `--mesh-cols` | `1` | Number of columns in mesh shape. T3K has 8 devices; use `--mesh-cols 4` (4 dev) or `--mesh-rows 2 --mesh-cols 4` (8 dev, physical topology). Galaxy: `--mesh-rows 4 --mesh-cols 8`. |
| `--device-ids` | `auto` | Comma-separated physical device IDs or `auto` |
| `--kv-cache-dtype` | `bf16` | KV cache data type: `bf16` (correctness) or `bf8` (memory/perf) |
| `--block-size` | `64` | KV cache block size |
| `--min-cache-tokens` | `128` | Minimum tokens to allocate in KV cache |
| `--phase` | `both` | Phase to run: `prefill`, `decode`, or `both` |
| `--enable-trace` | `false` | Enable traced decode execution (captures trace on first call, replays on subsequent) |
| `--trace-mode` | `logits` | Trace mode: `logits` (returns full logits to host) or `sampling` (on-device greedy top-1) |

---

## Environment Variables

**`tt/perf_defaults.py` is the single source of truth for the winning flag set.**
All three entry points apply it, so a served model and a benchmark run land on the
same configuration:

| Entry point | How it applies the set | Overridable from the environment? |
| --- | --- | --- |
| `scripts/debug_run_full_tt_greedy.py` | `apply_perf_defaults()` at import | Yes (`setdefault`) |
| `tt/generator_vllm.py` (vLLM) | `apply_perf_defaults(enable_moe=True, pin_off_experiments=True)` at import | Yes (`setdefault`) |
| `scripts/run_sweep_isl_batch.py` | hard `env.update(...)` per child run | **No** — deliberately, so an inherited environment cannot silently change what a sweep CSV describes |

The set is split into `WINNING_DEFAULTS` (12 vars whose *code* default is off or
conservative — these are what actually change behaviour), `CODE_DEFAULT_ON` (8 vars
already on in code, restated so the effective config is visible in one place and a
future code-default change cannot move the benchmarked configuration), and
`PINNED_OFF` (4 known-bad/neutral experiments). Defaults below are **code** defaults;
the "script" column says whether an entry point sets the flag for you.

> **Ordering:** `apply_perf_defaults()` must run *before* `Glm4RuntimeConfig.from_env()`
> and before any weight conversion — some modules snapshot their knobs into
> module-level constants at import time (e.g. `decoder_layer_tt._SHARDED_NORM`), and
> the dtype knobs are read at weight-load time. Both entry points therefore call it
> above their own model imports.

> **Library users:** if you construct `Glm4MoeLiteDenseOnlyTT` directly rather than
> going through one of the entry points, call
> `apply_perf_defaults(enable_moe=True)` yourself first. On bare code defaults you
> lose the entire optimization stack — and `ENABLE_MOE` is off, which **skips the
> routed experts entirely**.

### Default-On Optimizations

Already on in code — listed so you know what to turn **off** when bisecting.

| Variable | Code default | Description |
| --- | --- | --- |
| `GLM4_MOE_LITE_FUSED_COLLECTIVE_EPILOGUE` | **On** | Fuse the final routed expert reduction with shared-expert + residual adds into one `fast_reduce_nc` epilogue. Gated to 4x8 mesh, `TP=0`, sparse reduce dispatch, tokens≤32; **falls back safely** otherwise. Set `=0` to disable. |
| `GLM4_MOE_LITE_BUFFERED_MOE_ALL_REDUCE` | **On** | Replace the MoE gather+reduce with two buffered, semaphore-driven per-axis `all_reduce_async` passes. Bit-exact vs the safe path across 12 CCL configs. Set `=0` to disable. |
| `GLM4_MOE_LITE_FUSE_DOWN_ROUTING_SCALE` | **On** | Fold the per-token top-k routing weights into the sparse expert down-projection via a `sparse_matmul` `post_scale` width-broadcast epilogue, removing the standalone multiply. Self-gates to `num_blocks==1`; off in all-to-all mode. |
| `GLM4_MOE_LITE_SHARDED_NORM` | **On** | Width-shard the two decode RMSNorms across 8 cores (hidden=2048 = 64 tiles, otherwise single-core). Decode only; prefill unchanged. Set `=0` for the single-core path. |
| `GLM4_MOE_LITE_NORM_L1` | **On** | Keep norm intermediates in L1. |
| `GLM4_MOE_LITE_ROUTER_L1` | **On** | Keep MoE router intermediates in L1 (decode, tokens≤32). |
| `GLM4_MOE_LITE_EXPLICIT_PROG_CFG` | **On** | Explicit matmul program configs for one-tile, non-batched matmuls. Validated 58.1 → 54.2 ms/token on Galaxy B1. |
| `GLM4_MOE_LITE_KV_UPDATE_MAX_USERS=N` | `16` | Sub-batch size for the paged KV-cache write. `paged_update_cache` puts one sequence per core and its per-core L1 CBs overflow the 1499136 B limit just past batch 16 (a 32-user call is ~0.6% over). Only the cheap KV write is sub-batched; attention/MoE/collectives still run at full batch. |
| `GLM4_MOE_LITE_FORCE_OUTER_PAD=1` | Off | Restore the outer token pad/slice round-trip in `moe_mlp_forward` (removed in reduce mode — the sparse path re-pads internally). Debug/bisect only. |

### Feature Toggles

| Variable | Default | Script | Description |
| --- | --- | --- | --- |
| `GLM4_MOE_LITE_FUSED_ROUTER=1` | Off | **set** | Fused router device op (`topk_router_gpt`) replacing the multi-op routing sequence. Part of the winning set; the only winning flag whose code default is off. |
| `GLM4_MOE_LITE_ENABLE_MOE=1` | Off | **set** | Enable MoE layers. **Required for correctness** — with this off the routed experts are skipped entirely and only the shared expert runs. The debug runner hard-sets it; the vLLM path defaults it on. |
| `GLM4_MOE_LITE_NUM_LAYERS=N` | All (47) | — | Run only N layers (requires `DEBUG_ALLOW_PARTIAL_LAYERS=1`) |
| `GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS=1` | Off | — | Allow partial-layer runs with `NUM_LAYERS` |
| `GLM4_MOE_LITE_TP=1` | Off | pinned `0` | Tensor parallelism across mesh devices. **Known accuracy regression** — leave off; also disables the fused collective epilogue. |
| `GLM4_MOE_LITE_MTP=1` | Off | — | Multi-token prediction (MTP layer 47) |
| `GLM4_MOE_LITE_PRESERVE_TRACE=1` | Off | — | Skip trace release after prefill to avoid ~6s re-capture overhead |
| `GLM4_MOE_LITE_BATCHED_PREFILL=1` | Off | **set** | Batch the prefill across sequences |
| `GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE=N` | `0` (auto) | **set** (sweep) | Cap the prefill chunk length |

### Performance Tuning

| Variable | Default | Script | Description |
| --- | --- | --- | --- |
| `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1` | Off | **set** | Skip defensive clone operations (saves memory/time, may cause aliasing bugs) |
| `GLM4_MOE_LITE_FUSE_QKV_A=1` | Off | **set** | Fuse Q and KV_A projections into a single matmul |
| `GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1` | Off | **set** | Fuse shared MLP gate + up projections |
| `GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1` | Off | **set** | Fuse MLP + MoE reduce step (consolidates dual ReduceScatter+AllGather pairs in MoE layers) |
| `GLM4_MOE_LITE_SKIP_TYPECAST=1` | Off | **set** | Skip unnecessary bf16 typecasts in attention path (eliminates ~1,500 TypecastDeviceOperation calls per decode step) |
| `GLM4_MOE_LITE_DECODE_L1_ACT=1` | Off | **set** | Keep decode activations in L1 |
| `GLM4_MOE_LITE_EP_L1=1` | Off | **set** | Keep expert-parallel intermediates in L1 |
| `GLM4_MOE_LITE_CCL_NUM_LINKS=N` | `1` | **set** `4` | Ethernet links per CCL op |
| `GLM4_MOE_LITE_CCL_TOPOLOGY=ring` | `linear` | **set** `ring` | CCL topology |
| `GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1` | Off | — | Fuse expert gate + up projections (not in the winning set) |
| `GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1` | Off | pinned `0` | DRAM-sharded weight layout |
| `GLM4_MOE_LITE_DRAM_SHARDED_ATTN=1` | Off | pinned `0` | DRAM-sharded attention weights (requires `DRAM_SHARDED_WEIGHTS=1`) |
| `GLM4_MOE_LITE_DRAM_SHARDED_MLP=1` | On (if `DRAM_SHARDED_WEIGHTS=1`) | — | DRAM-sharded MLP weights |
| `GLM4_MOE_LITE_SHARDED_MLP=1` | Off | — | L1 WIDTH_SHARDED activations for shared MLP decode |
| `GLM4_MOE_LITE_BATCH_EXPAND=1` | Off | — | Batch expansion. Requires `KV_UPDATE_MAX_USERS >= batch`. |
| `GLM4_MOE_LITE_USE_DECODE_ROPE=1` | Off (auto-enabled with trace) | — | Use decode-specific RoPE implementation |
| `GLM4_MOE_LITE_MOE_FP32_ACC=1` | Off | — | FP32 accumulation for MoE matmuls |
| `GLM4_MOE_LITE_MLA_FP32_ACC=1` | Off | — | FP32 accumulation for FlashMLA (silently ignored unless `UNSAFE_ALLOW_FP32_MLA=1`) |
| `GLM4_MOE_LITE_CONCAT_HEADS=1` | Off | — | `ttnn.transformer.concatenate_heads` for attention output head-flattening (neutral in traced mode; not recommended) |
| `GLM4_MOE_LITE_NLP_CONCAT_HEADS=1` | Off | — | `ttnn.experimental.nlp_concat_heads` for prefill attention output path |

### Data Type Overrides

| Variable | Code default | Script | Description |
| --- | --- | --- | --- |
| `GLM4_MOE_LITE_EXPERTS_TT_DTYPE` | `bf8` | `bf8` | TT dtype for expert weights (`bf16`, `bf8`, `bf4`, `f16`, `f32`). `bf4` was tried and **reverted — accuracy regression**. |
| `GLM4_MOE_LITE_DENSE_TT_DTYPE` | `bf16` | **set `bf8`** | TT dtype for dense (non-expert) weights. **The code default is bf16**; `bf8` is a measured ~7% bs=1 decode win with coherence verified, and is what both scripts set. Set this explicitly outside the scripts. |
| `GLM4_MOE_LITE_KV_CACHE_TT_DTYPE` | (from CLI) | — | Override KV cache dtype. `bf16` is the validated setting — `bf8` has observed accuracy issues. |

### Known-Bad / Neutral Experiments

Present in the tree and gated off. These are `PINNED_OFF` in `tt/perf_defaults.py`:
explicitly forced to `0` by the sweep and defaulted to `0` on the vLLM path, so an
inherited environment cannot turn them on in a serving or benchmarking run. Do not
enable without re-validating.

| Variable | Status |
| --- | --- |
| `GLM4_MOE_LITE_FUSED_KV_BRANCH=1` | **Numerically incorrect — do not enable.** Builds and runs, but produces wrong results. The `fused_ops/kv_cache_branch/` kernel is kept as a gated experiment. |
| `GLM4_MOE_LITE_TRACE_2CQ=1` | Neutral at bs=1. Stages host inputs on CQ1; no measured win. Incompatible with MTP. |
| `GLM4_MOE_LITE_LMHEAD_SHARD=1` | **Negative at bs=1.** Auto-enabled when `TP=1`. |
| `GLM4_MOE_LITE_TP=1` | Accuracy regression (see Feature Toggles). |

### Debug / Profiling

| Variable | Default | Description |
| --- | --- | --- |
| `TT_METAL_DEVICE_PROFILER=1` | Off | Enable device profiler (used by `tt_metal_profiler`) |
| `GLM4_MOE_LITE_PROFILE=1` | Off | Enable per-op Python-level profiling |
| `GLM4_MOE_LITE_PROFILE_LAYER=N` | All | Profile only layer N |
| `GLM4_MOE_LITE_PROFILE_PRINT_EVERY=N` | (default) | Print profile every N steps |
| `GLM4_MOE_LITE_PROFILER_READ_INTERVAL=N` | `0` (off) | Device-profiler read interval, in decode steps |
| `GLM4_MOE_LITE_MOE_ROUTER_IMPL=cpu` | `tt` | Use CPU reference for MoE routing (debug) |
| `GLM4_MOE_LITE_MOE_EXPERTS_IMPL` | `sparse` | Expert dispatch: `sparse` (reduce), `dense`, `packed` |
| `GLM4_MOE_LITE_MLA_SCALE_MODE=kvpe` | `qk` | MLA scaling mode (`qk` matches HF, `kvpe` is experimental) |
| `GLM4_MOE_LITE_DECODE_EMBED_ONLY=1` | Off | Skip all decoder layers, return after embedding (debug) |
| `GLM4_MOE_LITE_DEBUG_LOGITS_SANITY=1` | Off | Run logits sanity checks |
| `GLM4_MOE_LITE_DEBUG_PAGE_TABLE_BOUNDARY=1` | Off | Debug page table boundary conditions |
| `GLM4_MOE_LITE_SYNC_AFTER_KV_UPDATE=1` | Off | Force device sync after KV cache update |
| `GLM4_MOE_LITE_LAYER_IDENTITY=1` | Off | Make each layer an identity function (debug) |
| `GLM4_MOE_LITE_SKIP_KV_UPDATE=1` | Off | Skip KV cache update entirely (debug) |
| `GLM4_MOE_LITE_DISABLE_MLP=1` | Off | Disable MLP/MoE FFN (debug) |
| `GLM4_MOE_LITE_DISABLE_FLASH_MLA_DECODE=1` | Off | Disable FlashMLA for decode (debug) |

> **Note:** `TT_METAL_GTEST_ETH_DISPATCH=1` is **not** required for these two scripts —
> `debug_run_full_tt_greedy.py` requests `DispatchCoreConfig(DispatchCoreType.ETH)`
> programmatically. Set it only if you drive the model from a harness that does not.

---

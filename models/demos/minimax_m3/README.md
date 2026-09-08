# MiniMax-M3 — Prefill bring-up

TTNN implementation of **MiniMax-M3** prefill inference for Tenstorrent Blackhole.
Target: one Blackhole Galaxy (8×4 mesh) running **SP=8 × TP=4 + EP=32**.
Config: [`configs/MiniMax-M3/config.json`](configs/MiniMax-M3/config.json).

## Architecture

| | |
|---|---|
| Decoder layers | 60 (hybrid schedule; layers **0–2 dense**, **3–59 sparse/MoE**) |
| Hidden / MoE intermediate | 6144 / 3072 (dense MLP intermediate 12288) |
| Attention | GQA: 64 q / 4 kv heads, head_dim 128, **partial RoPE** (rotary_dim 64), θ=5e6, QK-norm |
| Sparse attention (MSA, layers 3–59) | block_size 128, top-16 blocks, 4 index heads, index_dim 128, forced-local block |
| MoE (layers 3–59) | 128 experts / top-4 + 1 always-on shared expert |
| Activation | clamped **swigluoai** (gpt-oss): α=1.702, clamp limit=7.0 |
| Vocab | 200064 |

## Deployment path (Galaxy, 8×4)

- **Sequence-parallel prefill** (SP=8) over the mesh rows, **tensor-parallel** (TP=4) over the columns, **expert-parallel** (EP=32) MoE.
- **Dense attention:** `ring_joint` SDPA (first-chunk + block-cyclic SP-sharded KV-cache read).
- **Sparse attention (MSA):** `indexer_score_msa` → top-k blocks → `sparse_sdpa_msa`, token-level causal mask, per-device causality via mesh-coord `cluster_axis`. On-device indexed RoPE (whole-cache block-cyclic cos/sin built once).
- **MoE:** DeepSeek EP dispatch/combine + the fused `unified_routed_expert_ffn` kernel with M3's clamped swigluoai activation (`RoutedExpertActivation.SwiGluOai`).
- **KV cache:** SP-sharded, block-cyclic; chunked prefill reads the accumulated prefix on-device.

All ttnn C++ ops and fabric mesh descriptors this model uses are upstreamed and consumed from `main`; this directory is Python only.

## Status

Verified on a Blackhole Galaxy against the torch golden KV-cache, per-layer, **race-free** (3 runs bit-identical):

| Run | min PCC across 60 layers (K / V / index_k) |
|---|---|
| 5k one-shot | 0.96289 / 0.87884 / 0.97573 |
| 10k chunked (2×5120, cache-read path) | 0.96380 / 0.88037 / 0.97607 |
| 55k chunked (11×5120, cache-read path) | 0.96738 / 0.88542 / 0.97836 |

Decode is not part of this bring-up.

## Run

MiniMax-M3 is a `trust_remote_code` model: install a `transformers` new enough to carry the `minimax_m3_vl` modeling code, and download the checkpoint (safetensors weights + `config.json` + tokenizer) from the official MiniMax-M3 release (HuggingFace / GitHub) into a local dir that `HF_MODEL` points at. A tilized weight cache (`M3_WEIGHTS_FROM_CACHE=1`) avoids re-conversion after the first load.

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/path/to/MiniMax-M3
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
export EXPERT_DTYPE=bf4 M3_WEIGHTS_FROM_CACHE=1   # bf4 experts (default); EXPERT_DTYPE=bf8 trades ~20% more memory for higher PCC

# Per-layer KV-cache PCC vs golden (one-shot 5120):
PREFILL_CHUNKED=0 PREFILL_TRACE_DIR=/path/to/golden/longbook_5120 \
  python3 models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py

# Chunked (2 chunks -> exercises the cache-read path):
PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=5120 PREFILL_TRACE_DIR=/path/to/golden/longbook_10240 \
  python3 models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py
```

Module-level PCC tests (vs torch reference / HF) live under [`tests/unit/`](tests/unit/); golden generation is in [`scripts/`](scripts/).

### Environment variables

Every variable the M3 prefill code and its harnesses read. The production runner
(`models/demos/common/prefill/`) has its own `PREFILL_*` set, documented there; the ones the M3 adapter /
runtime consume are listed here.

**Model / weights** (read by `tt/`, apply to every harness and the production runner)

| Variable | Default | Effect |
|---|---|---|
| `HF_MODEL` | unset (required) | MiniMax-M3 checkpoint dir (safetensors + `config.json` + tokenizer); also the tilized-cache root unless `TT_CACHE_PATH` is set. `PREFILL_HF_MODEL` overrides it in the production runner. |
| `TT_CACHE_PATH` | `$HF_MODEL` | Root for the tilized weight cache `tensor_cache_bfp8_MeshShape([sp, tp])`. Point it at a dir you own when the checkpoint dir is read-only. |
| `EXPERT_DTYPE` | `bf4` | MoE routed-expert weight dtype: `bf4` or `bf8` (~20% more memory, higher PCC). |
| `M3_WEIGHTS_FROM_CACHE` | unset | `1` = load every tilized weight from the cache and skip the bf16 source read (also automatic when the cache is complete for the requested layers). |
| `M3_FORCE_LOAD_WEIGHTS` | unset | `1` = read the bf16 source even if the cache is complete (cache populate / refresh). |
| `M3_LOAD_NLAYERS` | unset (all) | Read only the safetensors shards holding layers `[M3_LOAD_LAYER_START, +N)` plus embed/norm/lm_head. Set by the harnesses from `PREFILL_NUM_LAYERS` / `PROFILE_NUM_LAYERS` / `NLAYERS`. |
| `M3_LOAD_LAYER_START` | `0` | First global layer of the `M3_LOAD_NLAYERS` window (a pipeline rank's adapter sets its `first_layer_idx`). |
| `M3_SHARDED_RESIDUAL` | `1` | Residual-stream layout (`tt/residual.py`): `1` = `emb/tp`-sharded per TP column (attention and the MLPs close with a reduce-scatter only); `0` = full-emb replicated (the layout older baselines were measured on, kept for bisects). |
| `M3_SHARDED_RESIDUAL_NORM` | `gather_first` | Only with a sharded residual: `gather_first` = all-gather the residual shard, then one single-pass norm (measured fastest); `distributed` = 3-op distributed RMSNorm on the shard, then all-gather (the DeepSeek/Kimi/GLM shape, kept for A/Bs). |
| `M3_EMBED_SHARD_VOCAB` | `1` | Embedding table sharding (`tt/parallel_embedding.py`): `1` = 2D vocab + hidden; `0` = 1D hidden only. Each layout has its own cache entry. |
| `M3_INDEX_CACHE_BF16` | unset | `1` = cache the MSA `index_k` in bf16 instead of the K/V cache dtype (bf8), keeping the indexer's hard top-16 block selection stable across chunks. |
| `M3_PROFILE_ZONES` | `0` | `1` = emit the Tracy zone signposts (`utils/profiler_utils.py`); read at import, the profiler harness sets it. |
| `M3_PROFILE_LEVEL` | `2` | Zone detail: `1` attn vs mlp, `2` every block that costs real time, `3` everything incl. norms and sub-splits. |
| `M3_PROFILE_HOST_ZONES` | `1` | `0` = skip the host-side Tracy zones (cosmetic; the signposts are what the parser reads). |

**Fabric / collectives** (harness-side; the production runner uses `PREFILL_FABRIC_MODE`, default `1d` for SP<=8)

| Variable | Default | Effect |
|---|---|---|
| `TT_MESH_GRAPH_DESC_PATH` | script-picked | Mesh graph descriptor. `FABRIC_1D` runs on `single_bh_galaxy_mesh_graph_descriptor.textproto`; ring / torus fabrics need `single_bh_galaxy_torus_xy_graph_descriptor.textproto` (the scripts pick it). |
| `M3_FABRIC` | `FABRIC_1D` | `ttnn.FabricConfig` name for `tests/galaxy_prefill_kv_pcc.py` / `scripts/run_prefill_perf.sh`. The MSA `high_bw_all_gather` rings by itself on a ring/torus fabric (measurements in PR #55668). |
| `M3_CCL_TOPOLOGY` | `Linear` | `ttnn.Topology` for the legacy CCLs (`all_gather_async`, `reduce_scatter_minimal_async`) in both harnesses. `Ring` needs a ring/torus fabric. |
| `PROFILE_FABRIC` / `FABRIC` | `1d` | Zone-profiler fabric: `1d`, `1d_ring`, `2d`, `2d_torus_xy` (`tests/perf/profile_prefill.py` / `scripts/run_prefill_profile.sh`). |

**KV-PCC / perf harness** (`tests/galaxy_prefill_kv_pcc.py`, driven by `scripts/run_prefill_perf.sh`)

| Variable | Default | Effect |
|---|---|---|
| `PREFILL_TRACE_DIR` | unset (required) | Golden trace dir (`metadata.json` with `token_ids`, plus `kv_cache/` for the PCC check). |
| `PREFILL_CHUNKED` | `0` | `1` = chunked prefill (exercises the cache-read path); `0` = one-shot. |
| `PREFILL_CHUNK_SIZE` | `5120` | Tokens per chunk (>= 2048 so the first MSA chunk has 16 blocks). |
| `PREFILL_NUM_LAYERS` | all 60 | Build / run only the first N layers (sets `M3_LOAD_NLAYERS`). |
| `PREFILL_TPS_ITERS` | `1` | Timed whole-sequence repetitions. |
| `PREFILL_SKIP_PCC` | unset | `1` = perf only, skip the per-layer KV PCC (synthetic traces carry no golden). |
| `PREFILL_EXPECTED_TPS` / `PREFILL_PERF_MARGIN` | unset / `0.05` | Perf gate: assert whole-sequence tok/s within `EXPECTED +/- MARGIN`. |
| `PREFILL_REQUIRE_HIGH_POWER` | `0` | `1` = skip unless the galaxy is at its high-power TDP (perf CI only). |
| `PREFILL_STANDALONE_CHUNKED_PCC` | `0.88` | Per-layer KV PCC floor (also read by `tt/runners/prefill_kv_validation.py`). |
| `PREFILL_STANDALONE_CHUNKED_RECORD_ONLY` | `0` | `1` = record PCCs without asserting (`prefill_kv_validation.py`). |
| `GOLDEN_DIR`, `SRC_TRACE`, `LOGDIR`, `PERF_WORKDIR` | see script header | `run_prefill_perf.sh` trace synthesis and logging paths. |

**Zone profiler** (`tests/perf/profile_prefill.py`, driven by `scripts/run_prefill_profile.sh`): `PROFILE_CHUNK`,
`PROFILE_CACHE`, `PROFILE_NUM_LAYERS`, `PROFILE_LAYER_IDS`, `PROFILE_READ_EVERY`, `PROFILE_READ_IN_CHUNK`,
`PROFILE_SKIP_PREFIX`, `PROFILE_STAGES`, `PROFILE_STAGE`, `PROFILE_FABRIC`, `PROFILE_DRY_RUN` and the script's
short aliases (`CHUNK`, `CACHE`, `LAYERS`, `LAYER_IDS`, `LEVEL`, `STAGES`, `STAGE`, `FABRIC`, `NOC_TRACES`,
`SKIP_PREFIX`, `RESULTS_DIR`) are documented in the harness docstring and the script header; see
[`tests/perf/README_profiling.md`](tests/perf/README_profiling.md).

**Other harnesses and tests**

| Variable | Where | Effect |
|---|---|---|
| `NUM_GEN`, `TARGET_LEN`, `NLAYERS`, `FORCE_DENSE`, `EP_SEQ_PER_CHIP`, `NO_SP` | `tests/galaxy_generate_m3.py` | Tokens to generate, prompt length (`5120`), layers to build, all-dense attention (`1`), per-chip EP token count, disable SP (`1`). |
| `PERF_SEQ`, `PERF_LAYERS`, `PERF_EXPERTS`, `PERF_REPS`, `REAL`, `TRACE_REGION` | `tests/perf/test_model_perf.py` | Synthetic model perf test shape / repetitions / real-weights switch / trace region bytes. |
| `M3_CKPT` | `tests/unit/test_msa_layer_vs_ref.py` | Checkpoint dir for the real-weights MSA test (falls back to `HF_MODEL`; the test skips without either). |
| `RUN_INDEXER_ACCURACY` | `tests/unit/test_indexer_score_msa_accuracy.py` | `1` = run the opt-in indexer accuracy sweep. |
| `REF_ATTN_Q_CHUNK`, `REF_FFN_TOKEN_CHUNK` | `reference/model.py` | Torch reference chunking (`256` query rows, `4096` FFN tokens) to bound host memory. |
| `OFFLOAD_DIR` | `tests/golden_hf_first_token.py` | HF `accelerate` offload dir for the first-token golden (`/tmp/m3_offload`). |
| `CI` | `tests/test_factory.py` | `true` = parametrize only the largest mesh shape that fits the runner. |

Multi-galaxy pipeline-parallel prefill (2 / 4 galaxies) — running, KV-cache accuracy, and throughput/overlap measurement — is documented in [`docs/PIPELINE_PREFILL_TESTING.md`](docs/PIPELINE_PREFILL_TESTING.md).

## Layout

```
tt/attention/     dense SP (ring_joint) + sparse MSA (indexer + sparse_sdpa_msa), RoPE, KV-cache
tt/moe/   EP MoE (TtMiniMaxMoE + fused swigluoai routed expert), activation
tt/               dense_mlp, layer, model, rms_norm, topk, mlp, weight_cache, tt_prefill_runtime
reference/        torch reference model + sparse GQA prefill
scripts/          golden KV-cache generation + verification
docs/             multi-galaxy pipeline-parallel prefill running & testing
configs/MiniMax-M3/config.json    dims only (modeling code loaded from the checkpoint via HF_MODEL)
tests/unit/       module-by-module PCC tests
tests/            galaxy harnesses (prefill KV-cache PCC, first-token, smoke)
```

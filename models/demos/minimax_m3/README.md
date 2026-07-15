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

## Layout

```
tt/attention/     dense SP (ring_joint) + sparse MSA (indexer + sparse_sdpa_msa), RoPE, KV-cache
tt/moe/   EP MoE (TtMiniMaxMoE + fused swigluoai routed expert), activation
tt/               dense_mlp, layer, model, rms_norm, topk, mlp, weight_cache, tt_prefill_runtime
reference/        torch reference model + sparse GQA prefill
scripts/          golden KV-cache generation + verification
configs/MiniMax-M3/config.json    dims only (modeling code loaded from the checkpoint via HF_MODEL)
tests/unit/       module-by-module PCC tests
tests/            galaxy harnesses (prefill KV-cache PCC, first-token, smoke)
```

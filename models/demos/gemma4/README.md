<!-- template-version: 4 -->

# Gemma-4

Gemma 4 is the next-generation open-weights model family from Google, extending the Gemma line with mixed sliding-window/global attention, partial RoPE on global layers, per-layer-input embeddings on the smaller variants, and a sparse mixture-of-experts block on the larger ones. This directory implements inference for five checkpoints — E2B and E4B (dense, with per-layer-input embeddings), 12B, 26B-A4B, and 31B — running on TT-NN with tensor parallelism across Tenstorrent meshes, plus the vision tower, the 12B audio embedder and image/video/audio prefill for the served multimodal variants.

## Variants

| Variant | HuggingFace card | Modalities served |
|---|---|---|
| E2B | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | text |
| E4B | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | text |
| 12B | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | text, image, video, audio |
| 26B-A4B | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | text, image, video |
| 31B | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | text, image, video |

## Served Performance

Release-flow benchmarks (tt-inference-server `--workflow release` on the tt-shield runners, vLLM serving, 128 output tokens, prefix caching off per request). Single-user TTFT at 128 / 2048 / 8192 / 32768 input tokens, single-user decode at 128/128, and the full-concurrency 128/128 point. The 31B and 12B numbers are the measured blocks in the [benchmark reference](https://github.com/tenstorrent/tt-inference-server/blob/main/reference_config/benchmarking/benchmark_targets/model_performance_reference.json); every run linked below passed its release gates.

| Variant | System | Layout | TTFT 128 / 2K / 8K / 32K (ms) | Tokens/s/user (1 user) | Max users | TTFT / tokens/s/user at max users | Source (tt-shield run ids) |
|---|---|---|---|---:|---:|---|---|
| 31B | Blackhole Galaxy | DP=4 x TP=8 | 118 / 547 / 2351 / 10728 | 25.4 | 128 | 2286 ms / 18.7 | release 39420209538, benchmark 37769772176 |
| 31B | QB2 (P300x2) | TP=4 | 130 / 582 / 2726 / 14420 | 26.7 | 32 | 3112 ms / 20.2 | release 39427665515, benchmark 37769767325 |
| 12B | Blackhole Galaxy | DP=4 x TP=8 | 72 / 297 / 1261 / 5573 | 44.2 | 128 | 1219 ms / 30.0 | release 39427589570, benchmark 37720096956 |
| 12B | QB2 (P300x2) | TP=4 | 76 / 332 / 1403 / 6737 | 45.2 | 32 | 1689 ms / 31.6 | release 39427740462, benchmark 37769777681 |
| 26B-A4B | Blackhole Galaxy | one engine, TP=8 x EP=4 (4 lanes x 32 slots) | 450 / 477 / 1320 / 5455 | 22.1 | 128 | 2526 ms / 6.5 | release 39894754540 |
| 26B-A4B | QB2 (P300x2) | column mesh (4,1), TP=4, token-dispatch prefill | 144 / 389 / 1169 / 5467 | 46.2 | 32 | 3198 ms / 34.7 | local serving on this branch (release CI at this head pending; the fused decode set is opt-in) |

Release-flow evaluations (thinking on, temperature 1.0; MMMU-Pro thinking off, greedy; each within tolerance of the same-protocol GPU reference):

| Variant | System | GPQA Diamond | LiveCodeBench | MMMU-Pro | Terminal-Bench 2 | FLEURS (en) | tt-shield run id |
|---|---|---:|---:|---:|---:|---:|---|
| 31B | Blackhole Galaxy | 83.33 | 78.63 | 66.13 | 58.33 | n/a | 39420209538 |
| 31B | QB2 | 83.33 | 79.39 | 64.97 | 62.50 | n/a | 39427665515 |
| 12B | Blackhole Galaxy | 77.78 | 71.76 | 60.46 | 20.83 | 93.34 | 39427589570 |
| 12B | QB2 | 74.75 | 74.05 | 59.77 | 33.33 | 93.44 | 39427740462 |
| 26B-A4B | Blackhole Galaxy | 79.80 | 74.81 | 65.32 | 50.00 | n/a | 39894754540 |

## Metal Demo Performance (CI)

Batch-1 `text_demo.py` runs from the branch's e2e CI (128-token prompt, 128 generated tokens, warm decode after the one-time compile; TTFT includes the first-token prefill of the warmed server). The 26B-A4B job also runs the image demo (`vision_demo.py`): image TTFT 14.2 s, 16.1 tokens/s/user.

| Variant | System | Mesh | TTFT (ms) | Tokens/s/user | CI leg (run 39925769041) |
|---|---|---|---:|---:|---|
| E2B | N150 | 1x1 | 389 | 14.6 | wh_n150, job 120326719435 |
| E2B | P150 | 1x1 | 111 | 19.7 | bh_p150, job 120326719361 |
| E4B | P300 | 1x2 | 150 | 17.9 | bh_p300, job 120326719270 |
| 12B | P150 | 1x1 | 91 | 16.9 | bh_p150, job 120326719372 |
| 12B | QB2 (P150x4) | 1x4 | 54 | 39.0 | bh_quietbox_2, job 120326719248 |
| 26B-A4B | T3K | 1x8 | 398 | 19.9 | wh_llmbox, job 120326719319 |
| 31B | T3K | 1x8 | 186 | 16.0 | wh_llmbox, job 120326719397 |

## Serving knobs and defaults per build

`tt/mesh_orientation.py` is the source of truth: a model build records its profile (`set_decode_fusion_profile`) and
every fusion flag is read through `decode_flag` (an env var, when set, wins). The profiles:

| Build | Fused decode / prefill programs on by default |
|---|---|
| 26B-A4B on the 4-chip Blackhole meshes ((1,4) row, (4,1) column) | `GEMMA4_MOE_FUSED`, `GEMMA4_MLP_GEGLU_FUSED`, `GEMMA4_QKV_NORM_ROPE_FUSED`, `GEMMA4_NORM_FUSED=2`, `GEMMA4_LAYER_NORM_FUSED`, `GEMMA4_SOFTCAP_FUSED`, `GEMMA4_CONCAT_HEADS_FUSED`, `GEMMA4_QKV_L1_OUT`, `GEMMA4_HEAD_NORM_4D`, `GEMMA4_MLP/ATTN/ROUTER_DECODE_1D`, `GEMMA4_SMALL_ROW_EXPERTS` |
| 31B / 12B on the 4-chip Blackhole meshes | `GEMMA4_DRAM_SHARD_ROW_VIEW`, `GEMMA4_QKV_L1_OUT`, `GEMMA4_QKV_HEADS_FUSED`, `GEMMA4_CONCAT_HEADS_FUSED`, `GEMMA4_CONCAT_HEADS_SHARDED`, `GEMMA4_MLP_GEGLU_SHARDED`, `GEMMA4_NORM_SHARDED_OUT`, `GEMMA4_SOFTCAP_FUSED`, `GEMMA4_QKV_NORM_ROPE_PREFILL` (the fused per-head norm + RoPE prefill program; refused on Wormhole whatever the env) |
| Wormhole, one chip, the (1,8) line, the Galaxy lane engine | none: the composite ttnn paths (`GEMMA4_<flag>=1` opts a program in where its kernels are validated) |

The other knob families, with the code defaults (unset = the code's own choice):

| Family | Knob = default | Meaning |
|---|---|---|
| Decode-trace guard | `GEMMA4_TRACE_ALLOC_GUARD=1`, `_ARM=alloc`, `_PERIOD=32`, `_MAX=8`, `_MIN_INTERVAL_S=30` | re-capture the decode trace when a device allocation appears after a capture (rate-limited) |
| Prefill trace / warmup | `GEMMA4_TRACE_SHORT_PREFILL_MAX=128`, `GEMMA4_TRACE_PREFILL_SEQ_LENS`, `GEMMA4_WARMUP_PREFILL_BATCHES`, `GEMMA4_WARMUP_PREFILL_ISLS` (unset = the model's ladder) | which prefill shapes are traced / warmed at boot |
| Prefill SDPA | `GEMMA4_PREFILL_SDPA_PROGCFG` (unset = auto: fixed configs for >= 8 Q heads per chip on Blackhole, the op defaults below that; `1` fixed, `0` op defaults), `_WIDE=1`, `_QCHUNK` / `_KCHUNK` / `_GRID` (explicit chunks and grid, clamped to the device grid) | the prefill SDPA program config; every grid derives from `compute_with_storage_grid_size` |
| Prefill matmul slabs | `GEMMA4_PREFILL_L1_HEADROOM_KB=64` | the per-core L1 headroom the row-slab rule leaves (`tt/dram_sharded.py`) |
| KV cache / pools | `GEMMA4_KV_BFP8` (catalog default bfp8 for the dense 4-chip builds), `GEMMA4_BOUNDED_SLIDING_KV_CACHE` (unset = the plugin's choice), `GEMMA4_SPEC_RING_HEADROOM_BLOCKS` (unset = sized from the verify width), `GEMMA4_PACKED_PREFILL=0`, `GEMMA4_LONG_CONTEXT_POLICY` (unset = the per-mesh cutover table) | paged-KV dtype, the bounded sliding ring, the packed multi-prompt prefill, the long-context cutovers |
| Lane engine (Galaxy) | `GEMMA4_LANE_PREFILL_PROMPTS=1`, `GEMMA4_LANE_PREFILL_CAPABILITY=1`, `GEMMA4_LANE_CCL_ASYNC=0`, `GEMMA4_LANE_CCL_SYNC=0`, `GEMMA4_LANE_PREFILL_TABLE_UPLOAD=1` | the k-prompts-per-lane gathered prefill, the capability the plugin reads, the lane collectives |
| Collectives | `GEMMA4_CCL_TOPOLOGY` (unset = auto), `GEMMA4_CCL_ASYNC=0`, `GEMMA4_CCL_PERSISTENT_BUF=1`, `GEMMA4_CCL_NUM_WORKERS=2`, `GEMMA4_CCL_NUM_BUFFERS=2`, `GEMMA4_CCL_NUM_LINKS` (unset = the mesh's) | the CCL manager's topology, async mode and buffers |
| dFlash / MTP speculation | `GEMMA4_DFLASH_VERIFY=5`, `_SERVE_HORIZON=2048`, `_CTX_CAP=2048`, `_WIDTH_SET=1`, `_WARMUP_DECODE=1`, `_PACKED=1`, `_SDPA=1`, `_CTX_CACHE=1`, `_DECODER_REUSE=0`, `_MAX_SPEC_ISL=0`, `GEMMA4_DFLASH_DRAFTER` (unset = resolved from the HF cache) | the speculative serving contract's widths, horizon and drafter |
| Accuracy gates (tests) | `GEMMA4_MIN_TOP1` / `GEMMA4_MIN_TOP5` (unset = reference minus the band; `0` = a report-only row), `GEMMA4_TOKACC_TAG`, `GEMMA4_PROSE_REPORT_ONLY`, `GEMMA4_PROSE_TAG`, `GEMMA4_PROSE_LENGTH=2048`, `GEMMA4_HF_REF_CACHE` (unset = `TT_CACHE_PATH/hf_reference`), `GEMMA4_TEST_PROMPT_TOKENS`, `GEMMA4_TEST_LAST_GATE`, `GEMMA4_TEST_GRID` (test-only cap of the compute grid) | the token-accuracy and prose gates and their report-only rows |

## Code Support Matrix

What the code in this directory supports, independent of what CI exercises.

Legend: 🟢 fully supported · 🟡 supported with limitations · 🔴 not supported · — not applicable

| Variant | N150 (1x1) | N300 (1x2) | T3K (1x8) | P150 (1x1) | QB2 / P150x4 (1x4) | LoudBox / P150x8 (1x8) | Blackhole Galaxy (8x4) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| E2B     | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | — |
| E4B     | 🟢 | 🟡 | 🟢 | 🟢 | 🟢 | 🟢 | — |
| 12B     | 🟡 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |
| 26B-A4B | 🔴 | 🔴 | 🟡 | 🔴 | 🟢 | 🟢 | 🟢 |
| 31B     | 🔴 | 🔴 | 🟢 | 🔴 | 🟢 | 🟢 | 🟢 |

Notes on the matrix:

- E4B on N300 runs locally; its CI entry is disabled for runner availability (`tests/pipeline_reorg/models_{unit,e2e}_tests.yaml`).
- 12B on N150 is short-context only (the long-context KV does not fit). On a single P150, E2B and E4B run the full 256k context unbounded with 4096-token prefill chunks; 12B reaches 256k with bounded sliding attention and 4096-token chunks above 32k (unbounded KV runs out of DRAM).
- Wormhole T3K carries 12 GB per ASIC (Blackhole P150: 32 GB): 12B and 31B reach 256k with bounded sliding attention and 2048-token prefill chunks above 8k to 32k; 26B-A4B on T3K is short-context only. 26B-A4B and 31B do not fit N150, N300 or a single P150 at any input length.
- QB2 (TP=4, 128 GB) and LoudBox (TP=8, 256 GB) are the long-context serving targets for 12B, 26B-A4B and 31B: `text_demo_v2.py` covers 4k to 256k with the per-device bounded-sliding and chunking policy in `tt/generator_trace.py`; the 26B-A4B 4-chip build opens the column mesh (4,1) with token-dispatch prefill by default.
- Blackhole Galaxy: 12B and 31B serve as four data-parallel TP=8 engines folded into in-process lanes; 26B-A4B serves as one engine over all 32 chips (TP=8 x EP=4 with four lanes of 32 slots). Long-context coherence on the Galaxy is checked with needle retrieval at 256k (single user and 4 x 11 users at 230k).

## Prerequisites

- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) with submodules.
- TT-Metalium / TT-NN installed: see [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
- Model-specific dependencies (tt-metal uses [`uv`](https://docs.astral.sh/uv/) for Python package management — install via `uv pip`, not plain `pip`):
  ```
  uv pip install -r models/demos/gemma4/requirements.txt
  ```
- HuggingFace cache populated with `HF_HOME` set, plus `HF_HUB_OFFLINE=1` to skip network access.

## How to Run

### E2B on N150 (1×1)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-E2B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-E2B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x1"
```

### E4B on N150 (1×1)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-E4B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-E4B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x1"
```

### E4B on N300 (1×2)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-E4B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-E4B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x2"
```

### 26B-A4B on T3K (1×8)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-26B-A4B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-26B-A4B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x8"
```

### 31B on T3K (1×8)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-31B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-31B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x8"
```

### P150 (1×1) — E2B / E4B / 12B (full ISL through 256k)

Use **`text_demo_v2.py`** for long-context / perf (logs TTFT **and** decode tok/s). `text_demo.py` remains the CI short-demo path.

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-12B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-12B-it \
       MESH_DEVICE=P150
# Short demo (CI path)
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x1" -s --timeout 1500
# Full HF context — 12B auto-bounds + chunk=4096 above 32k; E2B/E4B stay unbounded
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-256k" -s --timeout 3600
```

E2B/E4B: unbounded multi-chunk through 256k. 12B: bounded sliding + chunked prefill through 256k (required for DRAM).

### 12B / 26B-A4B / 31B on QB2 (P150x4) or LoudBox (P150x8)

Set `MESH_DEVICE` to the board mesh. Prefer **`text_demo_v2.py`** for long-context (TTFT + decode tok/s). Short CI demo and batch-32 stay on `text_demo.py`. Long-context rows are **not** gated by `--max-prefill`:

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-31B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-31B-it \
       MESH_DEVICE=P150x8   # or P150x4 / P300x2 for QB2

pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x8"
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-4k or long-context-32k or long-context-128k" -s --timeout 1800
pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_32 -k "prefill_128 and 1x8" -v
```

For 12B on QB2, use `HF_MODEL=google/gemma-4-12B-it` and `MESH_DEVICE=P150x4` (filter `-k "1x4"`).

Keep defaults for quality: `GEMMA4_HOST_SAMPLE=1` (default). Prefill chunk follows `GEMMA4_LONG_CONTEXT_POLICY` (usually 4096; **31B/26B-A4B @ ≥128k on QB2/LB → 2048 + bounded**). Avoid `GEMMA4_DEMO_SINGLE_CHUNK=1` on long ISL (known “la la / lapped” collapse). Optional overrides: `GEMMA4_BOUNDED_SLIDING`, `GEMMA4_GEN_PREFILL_CHUNK`, `GEMMA4_MAX_SEQ_LEN`, `GEMMA4_MAX_NEW_TOKENS`.

### Short / batch demos (`text_demo_v2.py` batch-1)

```bash
HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x8 \
  pytest models/demos/gemma4/demo/text_demo_v2.py -k "batch-1" -sv
```

### Single-layer smoke test

```bash
HF_MODEL=<path-or-id> pytest models/demos/gemma4/demo/text_demo.py::test_demo_single_layer
```

## Details

- **Entry points:**
  - `models/demos/gemma4/demo/text_demo_v2.py` — **preferred for long-context / perf**: Generator path; reports TTFT **and** decode tok/s (batch-1 / batch-32 / long-context). Same `GEMMA4_LONG_CONTEXT_POLICY` as v1.
  - `models/demos/gemma4/demo/text_demo.py` — CI short demo + batched prefill (`test_demo_batch_prefill` / `test_demo_batch_32`). Long-context Generator path works but currently logs TTFT only.
- **Batch size:** batch 1 by default; batched tests support up to batch-32 (override with `GEMMA4_BATCH_DEMO_SIZE`). On P150x8, batched-prefill microbatch is capped at ≤4 to avoid hangs.
- **Sequence length:** short demo buckets are 128 / 4096; long-context tests cover 4k–256k with per-(model, device) bounded-sliding / chunked-prefill cutovers (`GEMMA4_LONG_CONTEXT_POLICY` in `tt/generator_trace.py`). On **P150**, E2B/E4B/12B support full ISL **256k** (12B via bounded+chunked). On QB2/LB, coherence target for 12B/31B is 4k–128k. Batched prefill uses a **128k** virtual-token ceiling (`GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN`) with chunking above that.
- **Prefill trace:** enabled for MoE models on ISL buckets up to 4096 when `padded_batch × kernel < 32k`. Above 4k ISL or at/above 32k batched virtual tokens, prefill trace is disabled automatically (no perf gain, OOM risk).
- **Architecture:**
  - Mixed attention pattern: `sliding_attention` and `full_attention` layers interleaved per `hf_config.layer_types`.
  - Partial RoPE (factor 0.25) on global layers, full RoPE on sliding-window layers.
  - Per-layer-input embeddings on E2B/E4B; disabled on the MoE variants.
  - Optional `K=V` tying on global layers and KV-sharing across layer groups.
  - Sparse MoE block on 26B-A4B and 31B; dense MLP on E2B/E4B.
- **Pre/post-processing:** tokenization via the upstream HF tokenizer on host; logit softcapping (`final_logit_softcapping=30.0`) applied on device.

## Notes

- Weight cache is created on first run under `$TT_CACHE_PATH/tensor_cache_<dtype>/`. Subsequent runs reuse it; finetuned weights or a new dtype need a fresh cache. The cache directory must be writable (CI mounts of `/mnt/MLPerf` are often read-only — set `mlperf-read-only=false` or point `TT_CACHE_PATH` at a writable path for first-use).

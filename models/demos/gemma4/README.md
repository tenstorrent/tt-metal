<!-- template-version: 4 -->

# Gemma-4

Gemma 4 is the next-generation open-weights model family from Google, extending the Gemma line with mixed sliding-window/global attention, partial RoPE on global layers, per-layer-input embeddings on the smaller variants, and a sparse mixture-of-experts block on the larger ones. This directory implements text-only inference for five checkpoints — E2B and E4B (dense, with per-layer-input embeddings), 12B, 26B-A4B, and 31B — running on TT-NN with tensor parallelism across Tenstorrent meshes.

## Variants

| Variant | HuggingFace card |
|---|---|
| E2B | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) |
| E4B | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| 12B | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| 26B-A4B | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| 31B | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |

## Tested Configurations

The configurations below are exercised by CI. Performance numbers are from the linked CI run.

| Variant | System | Mesh | Tokens/s | Tokens/s/user | TTFT (ms) [^ttft] | CI job |
|---|---|---|---:|---:|---:|---|
| E2B     | N150 | 1×1 | 12.24 | 12.24 | 38714.7 | [log](https://github.com/tenstorrent/tt-metal/actions/runs/25099500256/job/73545762105) |
| E4B     | N150 | 1×1 |  7.95 |  7.95 | 36832.1 | [log](https://github.com/tenstorrent/tt-metal/actions/runs/25099500256/job/73545762123) |
| 26B-A4B | T3K  | 1×8 | 11.68 | 11.68 | 64186.5 | [log](https://github.com/tenstorrent/tt-metal/actions/runs/25099500256/job/73545762080) |
| 31B     | T3K  | 1×8 |  9.48 |  9.48 | 44772.1 | [log](https://github.com/tenstorrent/tt-metal/actions/runs/25099500256/job/73545762120) |

[^ttft]: TTFT is currently measured cold-start, including one-time device program compile. A steady-state TTFT will be added once the demo splits compile from inference timing.

E4B on N300 (1×2) is currently disabled in CI due to reduced N300 runner availability — see the Code Support Matrix below.

## Code Support Matrix

What the *code* in this directory supports, independent of what CI exercises.

Legend: 🟢 fully supported · 🟡 supported with known issues / limitations · 🔴 not supported · — not applicable

| Variant | N150 (1×1) | N300 (1×2) | T3K (1×8) | QB2 / P150x4 (1×4) | Galaxy (4×8) |
|---|:-:|:-:|:-:|:-:|:-:|
| E2B     | 🟢 | 🟢 | 🟢 | — | 🔴 |
| E4B     | 🟢 | 🟡 [^e4b-n300] | 🟢 | — | 🔴 |
| 12B     | 🔴 | 🔴 | — | 🟢 [^gemma4-12b] | 🔴 |
| 26B-A4B | 🔴 | 🔴 | 🟢 | — | 🔴 |
| 31B     | 🔴 | 🔴 | 🟢 | — | 🔴 |

[^e4b-n300]: E4B on N300 is exercised by the test suite locally but the CI entry is commented out due to runner availability. See `tests/pipeline_reorg/models_{unit,e2e}_tests.yaml`.
[^gemma4-12b]: 12B support uses `models/demos/gemma4/demo/text_demo_v2.py`, including batch-32 decode and long-context runs up to 256k tokens.

The 12B, 26B-A4B, and 31B variants are too large to fit on a single Wormhole device or N300. The 26B-A4B and 31B variants require T3K (TP=8). Galaxy (4×8) support has not yet been wired up.

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

### 12B on QB2 / P150x4 (1×4)

`text_demo_v2.py` supports batch-1 latency, batch-32 inference, and long-context runs up to 256k tokens for the 12B checkpoint.

```bash
HF_MODEL=google/gemma-4-12B-it pytest models/demos/gemma4/demo/text_demo_v2.py -k "batch-1"
```

For long-context 128k and 256k runs:

```bash
HF_MODEL=google/gemma-4-12B-it pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-128k"
HF_MODEL=google/gemma-4-12B-it pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-256k"
```

For batch-32 inference:

```bash
HF_MODEL=google/gemma-4-12B-it pytest models/demos/gemma4/demo/text_demo_v2.py -k "batch-32"
```

For a single-layer smoke test on any single device:

```bash
HF_MODEL=<path-or-id> pytest models/demos/gemma4/demo/text_demo.py::test_demo_single_layer
```

## Details

- **Entry points:**
  - `models/demos/gemma4/demo/text_demo.py` — single-user prefill + decode with on-device decode trace; batched prefill via `test_demo_batch_prefill` / `test_demo_batch_32` (marker `gemma4_batched_prefill`).
  - `models/demos/gemma4/demo/text_demo_v2.py` — batch and long-context runs (batch-32 inference, long-context up to 256k — see that file for limits).
- **Batch size:** `text_demo.py` defaults to batch 1; batched tests support batch-32 (override with `GEMMA4_BATCH_DEMO_SIZE`). `text_demo_v2.py` targets higher batch / long-context scenarios.
- **Sequence length:** up to 4096 in the standard demo; batched prefill in `text_demo.py` uses a **128k** virtual-token ceiling (`GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN`) with chunking above that. `text_demo_v2.py` supports longer contexts (up to 256k).
- **Prefill trace:** enabled for MoE models on ISL buckets up to 4096 when `padded_batch × kernel < 32k`. Above 4k ISL or at/above 32k batched virtual tokens, prefill trace is disabled automatically (no perf gain, OOM risk).
- **Architecture:**
  - Mixed attention pattern: `sliding_attention` and `full_attention` layers interleaved per `hf_config.layer_types`.
  - Partial RoPE (factor 0.25) on global layers, full RoPE on sliding-window layers.
  - Per-layer-input embeddings on E2B/E4B; disabled on the MoE variants.
  - Optional `K=V` tying on global layers and KV-sharing across layer groups.
  - Sparse MoE block on 26B-A4B and 31B; dense MLP on E2B/E4B.
- **Pre/post-processing:** tokenization via the upstream HF tokenizer on host; logit softcapping (`final_logit_softcapping=30.0`) applied on device.

## Notes

- Weight cache is created on first run under `$TT_CACHE_PATH/tensor_cache_<dtype>/`. Subsequent runs reuse it; finetuned weights or a new dtype need a fresh cache.

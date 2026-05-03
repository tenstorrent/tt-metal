<!-- template-version: 4 -->

# Gemma-4

Gemma 4 is the next-generation open-weights model family from Google, extending the Gemma line with mixed sliding-window/global attention, partial RoPE on global layers, per-layer-input embeddings on the smaller variants, and a sparse mixture-of-experts block on the larger ones. This directory implements text-only inference for four checkpoints — E2B and E4B (dense, with per-layer-input embeddings), and 26B-A4B and 31B (sparse MoE) — running on TT-NN with tensor parallelism across Wormhole meshes.

## Variants

| Variant | HuggingFace card |
|---|---|
| E2B | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) |
| E4B | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
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

| Variant | N150 (1×1) | N300 (1×2) | T3K (1×8) | Galaxy (4×8) |
|---|:-:|:-:|:-:|:-:|
| E2B     | 🟢 | 🟢 | 🟢 | 🔴 |
| E4B     | 🟢 | 🟡 [^e4b-n300] | 🟢 | 🔴 |
| 26B-A4B | 🔴 | 🔴 | 🟢 | 🔴 |
| 31B     | 🔴 | 🔴 | 🟢 | 🔴 |

[^e4b-n300]: E4B on N300 is exercised by the test suite locally but the CI entry is commented out due to runner availability. See `tests/pipeline_reorg/models_{unit,e2e}_tests.yaml`.

The 26B-A4B and 31B variants are too large to fit on a single Wormhole device or N300; they require T3K (TP=8). Galaxy (4×8) support has not yet been wired up.

## Prerequisites

- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) with submodules.
- TT-Metalium / TT-NN installed: see [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
- Model-specific dependencies (tt-metal uses [`uv`](https://docs.astral.sh/uv/) for Python package management — install via `uv pip`, not plain `pip`):
  ```
  uv pip install -r models/demos/gemma4/requirements.txt
  ```
- HuggingFace cache populated with `HF_HOME` set, plus `HF_HUB_OFFLINE=1` to skip network access.

Gemma4 demos default to the instruct checkpoint `google/gemma-4-26B-A4B-it`. Override it with `--model-path`, `HF_MODEL`, or `GEMMA4_MODEL_PATH`; override only the tokenizer with `--tokenizer-path` or `GEMMA4_TOKENIZER_PATH`. First run creates TT tensor caches under `$TT_CACHE_PATH/tensor_cache_<dtype>/`.

## How to Run

### User-facing 26B-A4B instruct demo on 1x8

Run hardware commands from the built TT-Metal checkout so dispatch kernels match the installed libraries:

```bash
cd /proj_sw/user_dev/moconnor/tt-metal
TT_METAL_CACHE=/tmp/tt-metal-cache-gemma4-it-smoke \
TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-26B-A4B-it \
HF_HOME=/path/to/huggingface \
HF_HUB_OFFLINE=1 \
HF_MODEL=google/gemma-4-26B-A4B-it \
/path/to/tt-metal/python_env/bin/python -u \
  /path/to/local/tt-metal/models/demos/gemma4/demo/instruct_demo.py \
  --prompt "Explain in two sentences why paged attention helps LLM serving." \
  --max-new-tokens 64 \
  --max-seq-len 512 \
  --mesh-rows 1 \
  --mesh-cols 8 \
  --trace-region-size 50000000
```

`instruct_demo.py` applies the tokenizer chat template when available. With the current local Transformers package, Gemma4 tokenizers load through the `extra_special_tokens={}` workaround and do not expose `chat_template`; the fallback mirrors the upstream Gemma4 [`chat_template.jinja`](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja) turn/channel tokens:

```text
<bos><|turn>user
...<turn|>
<|turn>model
<|channel>thought
<channel|>
```

Use `--base-completion` only for raw base-completion prompts. `--allow-cpu-sampling-fallback` is a debug-only escape hatch for meshes without `models.common.sampling`; TP meshes use device-side decode sampling.

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

For a single-layer smoke test on any single device:

```bash
HF_MODEL=<path-or-id> pytest models/demos/gemma4/demo/text_demo.py::test_demo_single_layer
```

## Details

- **Entry points:** `models/demos/gemma4/demo/instruct_demo.py` for CLI use and `models/demos/gemma4/demo/text_demo.py` for pytest/local harnesses.
- **Batch size:** 1 (single-user demo).
- **Sequence length:** up to 4096 tokens in the demo; the model itself supports the upstream context window.
- **Sampling:** first token after prefill is currently sampled on host because prefill returns gathered logits; decode uses `models.common.sampling` device-side sampling on TP meshes. Greedy, temperature, top-p, top-k, and seed are CLI-configurable.
- **vLLM:** `models/demos/gemma4/tt/generator_vllm.py` implements a minimum batch=1, `tt_data_parallel=1` vLLM adapter with paged attention metadata and per-layer KV-cache shapes. Continuous batching and prefix caching are not supported yet because Gemma4 mixes sliding/global cache geometry.
- **Weight dtype probes:** `GEMMA4_PRECISION_PROFILE=bf16` is the default. `GEMMA4_PRECISION_PROFILE=mixed_bfp8` keeps embeddings, norms, router auxiliaries, KV cache, RoPE, and LM head in BF16 while placing attention, shared-MLP, and expert projection weights in BFP8. Per-tensor overrides such as `GEMMA4_ATTENTION_QKV_WEIGHT_DTYPE`, `GEMMA4_ATTENTION_O_PROJ_WEIGHT_DTYPE`, `GEMMA4_SHARED_MLP_{GATE,UP,DOWN}_WEIGHT_DTYPE`, `GEMMA4_EXPERT_{GATE,UP,DOWN}_WEIGHT_DTYPE`, and `GEMMA4_LM_HEAD_WEIGHT_DTYPE` accept `bf16`, `bfp8`/`bfloat8_b`, and `bfp4`/`bfloat4_b`. Legacy group overrides (`GEMMA4_ATTENTION_WEIGHT_DTYPE`, `GEMMA4_SHARED_MLP_WEIGHT_DTYPE`, `GEMMA4_EXPERT_WEIGHT_DTYPE`) still work. Lower precision is opt-in profiling/bringup territory until full decode quality and latency are validated.
- **Architecture:**
  - Mixed attention pattern: `sliding_attention` and `full_attention` layers interleaved per `hf_config.layer_types`.
  - Partial RoPE (factor 0.25) on global layers, full RoPE on sliding-window layers.
  - Per-layer-input embeddings on E2B/E4B; disabled on the MoE variants.
  - Optional `K=V` tying on global layers and KV-sharing across layer groups.
  - Sparse MoE block on 26B-A4B and 31B; dense MLP on E2B/E4B.
- **Pre/post-processing:** tokenization via the upstream HF tokenizer on host; logit softcapping (`final_logit_softcapping=30.0`) applied on device.

## Notes

- Weight cache is created on first run under `$TT_CACHE_PATH/tensor_cache_<dtype>/`. Subsequent runs reuse it; finetuned weights or a new dtype need a fresh cache.

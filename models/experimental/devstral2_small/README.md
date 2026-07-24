# Devstral Small 2

## Platforms

P150 and Blackhole QuietBox.

## Introduction

This folder contains an experimental Tenstorrent (`ttnn`) port of **Mistral [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)** (Mistral3 multimodal): Pixtral-class vision tower, multimodal projector, and Ministral3 text stack. PCC tests compare subgraphs and full vision+projector, Text prefill and decode paths against Hugging Face references.

**Maximum context length**: **256K** on BH-QB, **49K** on P150 (prompt + generation combined) for the TT demos and on-device stack.

**Traced decode** is enabled for TT generation; **1540×1540** multimodal images are supported via `--vision-square-pixels 1540`.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Python packages (install into your tt-metal environment):

  ```sh
  pip install -r models/experimental/devstral2_small/requirements.txt
  ```

  Pins `mistral-common>=1.8.6` and installs the latest `transformers` from GitHub (required for Devstral/Mistral3).

- Model environment (from repo root):

  ```sh
  export HF_MODEL=mistralai/Devstral-Small-2-24B-Instruct-2512
  export TT_CACHE_PATH=/tmp/devstral_tt_cache/mistralai--Devstral-Small-2-24B-Instruct-2512
  ```


## How to run (PCC tests)

From repo root:

**All PCC tests** (building blocks + pipeline; long run, loads Devstral weights):

```sh
pytest models/experimental/devstral2_small/tests/ -k pcc
```

**Pipeline / composed models** (`tests/pipeline_tests/`):

```sh
pytest models/experimental/devstral2_small/tests/pipeline_tests/ -k pcc
```

Single file, e.g. attention only:

```sh
pytest models/experimental/devstral2_small/tests/test_ministralattn.py -k pcc
```

**Accuracy — Text Decoder Logits PCC**

Compares full decode logits (40 text layers + final norm + LM head) against HF for 32 decode steps. Default PCC threshold: ≥ 0.99 (`MINISTRAL3_DECODER_STACK_PCC`).

```sh
pytest models/experimental/devstral2_small/tests/pipeline_tests/test_ministral3_decoder_layer.py
```

**Accuracy — Text Prefill Logits PCC**

Compares full-depth prefill last-token logits (all text layers + final norm + LM head) against HF over the default sequence-length sweep (128 → 8k tokens) from the Tale of Two Cities corpus at `models/tt_transformers/tests/tale-of-two-cities.txt.bz2`. Default PCC threshold: ≥ 0.97 (`MINISTRAL3_PREFILL_LOGITS_PCC`).

```sh
pytest models/experimental/devstral2_small/tests/pipeline_tests/test_ministral3_prefill_logits.py
```

**Device profiling — Tracy CSV**

Runs the one-layer prefill+decode logits test (`n_layers=1`) under Tracy:

```sh
python -m tracy -p -r -v -m pytest models/experimental/devstral2_small/tests/test_ministral3_one_layer_prefill_decode_logits.py
```

**Performance — ISL Sweep**

Sweeps the input sequence length through the full vision-text pipeline. Text is sourced from the Tale of Two Cities corpus at `models/tt_transformers/tests/tale-of-two-cities.txt.bz2` (~192k text tokens); one image is always included; output is fixed at 200 tokens per sweep point. The default Devstral sweep is 4k → 256k tokens.

For ISL points above the corpus length (e.g. 256k = 262144 tokens), the sweep **tiles (repeats) the corpus** so prefill fills the KV cache for perf workload.

```sh
pytest models/experimental/devstral2_small/tests/pipeline_tests/test_ministral3_text_isl_sweep.py
```


## Demos

Run from repo root on Blackhole QuietBox or P150.

### Execution

Run from repo root. The only platform difference is the device mesh: **BH-QB uses `--mesh-width 4`** (1×4) and **P150 uses `--mesh-width 1`** (1×1). `--mesh-width` is now validated against the visible device count and fails early if the requested mesh is unavailable.

| Demo | Script | Description |
|------|--------|-------------|
| Image + text | `demo/tt_image_demo.py` | One-shot image Q&A on TT (vision → projector → text LM). Default prompt: `resource/sample.jpeg`. |
| Text LM | `demo/tt_text_demo.py` | Text-only TT prefill/decode + LM head. Default Fibonacci prompt; override with `--prompt`. |
| Interactive agent | `demo/tt_demo_agent.py` | Multi-turn coding REPL on TT; `/image PATH [question…]` for multimodal. |

**Image + text**

```sh
# Blackhole QuietBox (1x4)
python3 -m models.experimental.devstral2_small.demo.tt_image_demo --backend tt --image models/experimental/devstral2_small/resource/sample.jpeg --vision-square-pixels 1540 --mesh-width 4 --max-new-tokens 100

# P150 (1x1)
python3 -m models.experimental.devstral2_small.demo.tt_image_demo --backend tt --image models/experimental/devstral2_small/resource/sample.jpeg --vision-square-pixels 1540 --mesh-width 1 --max-new-tokens 100
```

**Text LM**

```sh
# Blackhole QuietBox (1x4)
python models/experimental/devstral2_small/demo/tt_text_demo.py --mesh-width 4

# P150 (1x1)
python models/experimental/devstral2_small/demo/tt_text_demo.py --mesh-width 1
```

**Interactive agent**

```sh
# Blackhole QuietBox (1x4)
python models/experimental/devstral2_small/demo/tt_demo_agent.py --vision-square-pixels 1540 --mesh-width 4

# P150 (1x1)
python models/experimental/devstral2_small/demo/tt_demo_agent.py --vision-square-pixels 1540 --mesh-width 1
```

### Performance

| Demo | System | Mesh | New tokens | Steady-state throughput| End-to-end throughput| TTFT (ms) |
|:-----|:-------|:-----|----------:|------:|----------:|----------:|
| Image + text | BH-QB | 1x4 | 100 | 29.4 | 20.8 | 1250 |
| Image + text | P150 | 1x1 | 100 | 5 | 4.2 | 3121 |
| Text LM | BH-QB | 1x4 | 200 | 30.3 | 28.4 | 279 |
| Text LM | P150 | 1x1 | 200 | 5.1 | 5 | 528 |

**Text Prefill Logits PCC**

Results from `test_ministral3_prefill_logits.py` on BH QB-2 (`P150x4`) using the Tale of Two Cities corpus from `models/tt_transformers/tests/tale-of-two-cities.txt.bz2`. This compares the full text model path (40 layers + final norm + LM head) against HF. PCC threshold: >= 0.97.

| Seq len | PCC |
|--------:|----:|
| 128 | 0.993696 |
| 256 | 0.996526 |
| 512 | 0.996700 |
| 1024 | 0.993400 |
| 4096 | 0.986712 |
| 8192 | 0.986108 |
| 32k, 64k, 128k, 256k | TBD |

Note: larger context lengths take longer to verify because the HuggingFace reference runs a full 24B forward pass over the entire sequence.

**Decode Logits PCC — 32 Generation Steps**

Results from `test_ministral3_decoder_layer.py` on BH QB-2 (`P150x4`). This compares the full text model path (40 layers + final norm + LM head) against HF over 32 decode steps. Per-step PCC threshold: >= 0.98.

| Step | PCC |
|-----:|----:|
|0  | 	0.996556|
|1  | 	0.984737|
|2  | 	0.996072|
|3  | 	0.993899|
|4  | 	0.997844|
|5  | 	0.997391|
|6  | 	0.998462|
|7  | 	0.997282|
|8  | 	0.997596|
|9  |	0.996876|
|10 |    0.998097|
|11 |    0.998316|
|12 |	0.998172|
|13 |	0.998323|
|14 |	0.997844|
|15 |	0.998249|
|16 |	0.997792|
|17 |	0.998143|
|18 |	0.998171|
|19 |	0.998485|
|20 |	0.998413|
|21 |	0.998412|
|22 |	0.998131|
|23 |	0.997850|
|24 |	0.997991|
|25 |	0.997961|
|26 |	0.998267|
|27 |	0.998253|
|28 |	0.998562|
|29 |	0.998707|
|30 |	0.997398|
|31 |	0.998379|

**ISL (Context Window) Sweep**

| Config | Batch | ISL (max_seq_len) | Prefill tokens | decode_t/s/u (tok/s) | decode_t/s (tok/s) |
|:-------|------:|------------------:|---------------:|---------------------:|-------------------:|
| b1_isl4k | 1 | 4k (4096) | 4096 | 29.94 | 29.94 |
| b1_isl8k | 1 | 8k (8192) | 8192 | 29.90 | 29.90 |
| b1_isl16k | 1 | 16k (16384) | 16384 | 29.69 | 29.69 |
| b1_isl32k | 1 | 32k (32768) | 32768 | 28.61 | 28.61 |
| b1_isl64k | 1 | 64k (65536) | 65536 | 26.62 | 26.62 |
| b1_isl128k | 1 | 128k (131072) | 131072 | 22.91 | 22.91 |
| b1_isl256k | 1 | 256k (262144) | 262144 | 18.77 | 18.77 |


## Resources

| Path | Purpose |
|------|---------|
| **`demo/`** | Runnable demos: `tt_image_demo.py` (TT multimodal), `tt_text_demo.py` (TT text LM), `tt_demo_agent.py` (interactive TT agent with `/image`). |
| **`resource/`** | Static assets only (e.g. `sample.jpeg` for image demos). No Python modules. |
| **`devstral_utils/`** | Shared helpers imported by demos, tests, and TT modules (re-exported from `devstral_utils/__init__.py`). |
| **`tt/`** | TT layer implementations (Ministral3 + Pixtral building blocks). |
| **`tt/pipeline/`** | Composed models: vision → projector → text LM. |
| **`tests/`** | Per-op PCC tests (attention, MLP, norms, Pixtral blocks, decoder layer, …). |
| **`tests/pipeline_tests/`** | Composed-model PCC: vision tower, projector, text stack, and projected image features. |

**`devstral_utils/` modules**

- `multimodal_demo_helpers.py` — mesh open/close, TT prefill padding, LM head helpers, traced decode buffers, multimodal scatter prefill.
- `fp8_dequantize_compat.py` — HF FP8 scalar-scale shim for Devstral checkpoints across `transformers` versions.
- `pixtral_seq_chunk.py` — vision matmul sequence chunk sizing (`pixtral_vision_seq_chunk_len`, `pixtral_effective_mm_seq_len`, pad/trim helpers; env `PIXTRAL_VISION_MM_SEQ_CHUNK*`).
- `dram_sharded_matmul.py` — DRAM-sharded decode matmul helpers (width-sharded L1 linear, weight build, core grids).

## Details

**Layout**

- **`tt/pipeline/`** — top-level TT models wired like HF `Mistral3Model`:
  - `tt_devstral2_small_model.py` — `TtDevstral2SmallModel` (vision + projector + `TtMinistral3Model`).
  - `tt_ministral3_model.py` — text stack (embed → decoder layers → RMSNorm; optional on-device RoPE).
  - `tt_pixtral_vision_model.py` — Pixtral vision tower.
  - `tt_multimodal_projector.py` — patch merger + projector into text hidden size.
- **`tt/`** — subgraph building blocks, e.g. `tt_ministral3_decoder_layer`, `tt_ministralattn`, `tt_ministralmlp`, `tt_ministralrmsnorm`, `tt_ministral_rotary_emb`, `tt_pixtralattn`, `tt_pixtralmlp`, `tt_pixtral_transformer`, `tt_patchmerger`, `tt_rmsnorm`, …
- **`demo/`** — prompts and default generation settings (`DEFAULT_GENERATE_KWARGS` in `tt_text_demo.py`, `MODEL_LOADING_MESSAGES` in `tt_image_demo.py`).
- **`devstral_utils/`** — cross-cutting demo/test utilities (see table above); demos import from here rather than duplicating mesh/prefill/trace logic.

**Model and limits**

- Default HF weights: `mistralai/Devstral-Small-2-24B-Instruct-2512` (`DEFAULT_MODEL_ID` in `devstral_utils`; override with `HF_MODEL` or `--model-id`).

- **Maximum context length:** **BH-QB: 256K tokens**, **P150: 52K tokens** (prompt + generation). TT demos size `ModelArgs.max_seq_len` to at least the platform cap (BH-QB **256000**, P150 **52000**), rounded to a 512 multiple for SDPA decode. Longer sessions need a smaller `--max-new-tokens` budget or a higher `max_seq_len` in code.

**PCC tests**

- Submodule tests under `tests/` validate individual TT ops against HF on Devstral weights (often layer 0, partial safetensors, or full `ModelArgs.load_state_dict()` where noted).
- Pipeline tests under `tests/pipeline_tests/` validate the vision tower, multimodal projector, text stack, and projected image features.

## Optimizations

The following optimizations were applied across the vision tower, projector, and text stack to improve throughput and memory use on Tenstorrent devices:

- **L1 placement** — Moved tensors that previously lived in DRAM into L1, including an L1 interleaved memory configuration where it improves bandwidth and locality.
- **L1 width sharding (matmul)** — Applied width-sharded L1 memory layouts to matmuls across the vision tower and text stack.
- **L1 block sharding (matmul and binary ops)** — Applied block-sharded L1 memory layouts to matmuls and binary operators across the vision tower and text stack, keeping the same sharding across ops to avoid `sharded_to_interleaved` and `interleaved_to_sharded` copies.
- **Kernel precision** — Converted HiFi2 kernels to LoFi where accuracy allows, reducing compute cost.
- **Reduced-precision weights** — Converted bfloat16 tensors to bfloat8 where supported to cut memory footprint and data movement.
- **Fused QKV projection** — Combined query, key, and value projections into a single fused matmul instead of three separate passes.
- **Vision MLP (FF1/FF3)** — Applied FF1/FF3 projection optimizations on the vision feed-forward path and fused W1 + W3 matmul input weights so the gate and up projections share one input read.
- **Fused SwiGLU** — Vision MLP uses a single multiply with SiLU fused on the activation (`input_tensor_a_activations=[SILU]`) instead of separate SiLU and multiply ops, matching patterns used on other Ministral/Qwen-VL ports.
- **Parallelism** — Increased core count on key ops for better device parallelism.
- **DRAM sharing** — Applied DRAM sharing optimizations on decode and sharded matmul paths to reuse buffers across steps.
- **Tensor lifetime** — Deallocated intermediate activations and gather buffers as soon as each stage finishes (QKV split, SDPA output, MLP gate/up paths, patch merger tiles, decode buffers) to lower peak device memory on long vision runs and multi-step decode.
- **Pure TTNN path** — Migrated remaining PyTorch operations to TTNN so inference stays on-device end to end.
- **Fused rotary embedding** — Replaced rotate-half style RoPE with a fused rotate-embedding implementation.
- **In-place residuals** — Attention and MLP blocks add residuals in place (`output_tensor=residual`) so the same op count avoids extra allocation and memory traffic.
- **Long-sequence vision matmul** — Restored pad → batched matmul → trim for long vision sequences instead of chunking and concatenating along the sequence axis.
- **Scaled dot-product attention (SDPA)** — Program config is chosen from sequence length: below 2048 tokens, 128×128 chunks (aligned with Mistral vision, PCC-safe); at 2048 and above, chunks scale up to 256×256 with matmul batch count. `PIXTRAL_SDPA_Q_CHUNK` and `PIXTRAL_SDPA_K_CHUNK` environment variables can override chunk sizes. The SDPA grid uses the device `max_grid_size`, and SDPA runs with an explicit DRAM memory config. Redundant `to_memory_config` after head concatenation was removed.
- **Async collectives (multi-chip)** — Replaced synchronous `all_gather` with `all_gather_async` on mesh paths (vision attention output gather, text MLP/decoder collectives where applicable). Tuned `chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`, and `num_links` per cluster axis (e.g. higher sync chunking and four workers per link on vision attention gather; two links and smaller chunks on text prefill/decode) for better fabric overlap and throughput.
- **Matmul batching and tiles** — On the 1024-token PCC path, matmul tile size increases from 512 to 1024 with batch fusion enabled. For 1540×1540 vision inputs (~12k tokens), the number of batches per matmul is roughly halved compared to the prior chunk-concat approach.


## CI support

Devstral is wired into Blackhole hardware CI via `tests/pipeline_reorg/blackhole_demo_tests.yaml` (model name `devstral-small-2-24b`, workflow `(Blackhole) Demo tests`). On BH QuietBox (`P150x4`, `MESH_DEVICE=P150x4`), CI runs:

- `pytest models/experimental/devstral2_small/tests/pipeline_tests/` — composed-model PCC, decode logits PCC, text prefill logits PCC (128–8k), and submodule pipeline tests
- `tt_image_demo.py` — TT multimodal (1540×1540 image, 100 new tokens)
- `tt_text_demo.py` — TT text LM

**Omitted from CI (long runtime):** Prefill logits PCC above **8k** (32k, 64k, 128k, 256k).

## Open Items

- Prefill logits PCC verification at context lengths >=32k (32k, 64k, 128k, 256k).

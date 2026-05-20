# Devstral Small 2 (experimental)

## Platforms

Blackhole only.

## Introduction

This folder contains an experimental Tenstorrent (`ttnn`) port of **Mistral [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)** (Mistral3 multimodal): Pixtral-class vision tower, multimodal projector, and Ministral3 text stack. PCC tests compare subgraphs and full vision+projector paths against Hugging Face references.

**Maximum context length:** 4K tokens (prompt + generation combined) for the TT demos and on-device stack.

**Traced decode** is enabled for TT generation; **1540×1540** multimodal images are supported via `--vision-square-pixels 1540`.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Python packages (install into your tt-metal environment):

  ```sh
  pip install git+https://github.com/huggingface/transformers
  pip install mistral-common
  ```

## How to run (PCC tests)

From repo root:

**All PCC tests** (building blocks + pipeline; long run, loads Devstral weights):

```sh
pytest models/experimental/devstarl2_small/tests/ -k pcc
```

**Pipeline / composed models** (`tests/pipeline_tests/`):

```sh
pytest models/experimental/devstarl2_small/tests/pipeline_tests/ -k pcc
```

Single file, e.g. attention only:

```sh
pytest models/experimental/devstarl2_small/tests/test_ministralattn.py -k pcc
```

## Demos

Run from repo root on BH-QB or P150.

### Execution

| Demo | Script | Description | Command |
|------|--------|-------------|---------|
| Image + text | `demo/tt_image_demo.py` | One-shot image Q&A on TT (vision → projector → text LM). Default prompt: `resource/sample.jpeg`. | `python3 -m models.experimental.devstarl2_small.demo.tt_image_demo --backend tt --image models/experimental/devstarl2_small/resource/sample.jpeg --vision-square-pixels 1540 --mesh-width 4 --max-new-tokens 100` |
| Text LM | `demo/tt_text_demo.py` | Text-only TT prefill/decode + LM head. Default Fibonacci prompt; override with `--prompt`. | `python models/experimental/devstarl2_small/demo/tt_text_demo.py --mesh-width 4` |
| Interactive agent | `demo/tt_demo_agent.py` | Multi-turn coding REPL on TT; `/image PATH [question…]` for multimodal. | `python models/experimental/devstarl2_small/demo/tt_demo_agent.py --vision-square-pixels 1540` |

### Performance

| Demo | System | Mesh | New tokens | t/s/u | t/s (e2e) | TTFT (ms) |
|:-----|:-------|:-----|----------:|------:|----------:|----------:|
| Image + text | BH-QB | 1x4 | 100 | 15.8 | 14.9 | 875 |
| Image + text | P150 | 1x4 | 100 | 2.4 | 2.3 | 6585 |
| Text LM | BH-QB | 1x4 | 200 | 17.5 | 17.1 | 198 |
| Text LM | P150 | 1x4 | 200 | 2.8 | 2.8 | 533 |

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
- **Maximum context length: 4K tokens** (prompt + generation). TT demos size `ModelArgs.max_seq_len` to at least **4096** (rounded to a 512 multiple for SDPA decode). Longer sessions need a smaller `--max-new-tokens` budget or a higher `max_seq_len` in code.

**PCC tests**

- Submodule tests under `tests/` validate individual TT ops against HF on Devstral weights (often layer 0, partial safetensors, or full `ModelArgs.load_state_dict()` where noted).
- Pipeline tests under `tests/pipeline_tests/` validate the vision tower, multimodal projector, text stack, and projected image features.

## Optimizations

The following optimizations were applied across the vision tower, projector, and text stack to improve throughput and memory use on Tenstorrent devices:

- **L1 placement** — Moved tensors that previously lived in DRAM into L1, including an L1 interleaved memory configuration where it improves bandwidth and locality.
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

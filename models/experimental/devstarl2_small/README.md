# Devstral Small 2 (experimental)

## Platforms

Blackhole only.

## Introduction

This folder contains an experimental Tenstorrent (`ttnn`) port of **Mistral [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)** (Mistral3 multimodal): Pixtral-class vision tower, multimodal projector, and Ministral3 text stack. PCC tests compare subgraphs and full vision+projector paths against Hugging Face references.

**Maximum context length:** 4K tokens (prompt + generation combined) for the TT demos and on-device stack.

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

Run from repo root on BH-QB (P150×4, `--mesh-width 4`).

### Execution

| Demo | Script | Description | Command |
|------|--------|-------------|---------|
| Image + text | `demo/tt_image_demo.py` | One-shot image Q&A on TT (vision → projector → text LM). Default prompt: `resource/sample.jpeg`. | `python3 -m models.experimental.devstarl2_small.demo.tt_image_demo --backend tt --image models/experimental/devstarl2_small/resource/sample.jpeg --vision-square-pixels 1540 --mesh-width 4 --max-new-tokens 100` |
| Text LM | `demo/tt_text_demo.py` | Text-only TT prefill/decode + LM head. Default Fibonacci prompt; override with `--prompt`. | `python models/experimental/devstarl2_small/demo/tt_text_demo.py --mesh-width 4` |
| Interactive agent | `demo/tt_demo_agent.py` | Multi-turn coding REPL on TT; `/image PATH [question…]` for multimodal. | `python models/experimental/devstarl2_small/demo/tt_demo_agent.py --vision-square-pixels 1540 ` |

### Performance

Representative BH-QB (P150×4) runs; each demo prints a traced-decode timing table to stdout after generation.

| Demo | System | Mesh | New tokens | t/s/u | t/s (e2e) | TTFT (ms) |
|:-----|:-------|:-----|----------:|------:|----------:|----------:|
| Image + text | BH-QB | 1x4 | 100 | 15.8 | 14.9 | 875 |
| Text LM | BH-QB | 1x4 | 200 | 17.5 | 17.1 | 198 |

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
- `pixtral_seq_chunk.py` — vision matmul sequence chunk sizing (`pixtral_vision_seq_chunk_len`; env `PIXTRAL_VISION_MM_SEQ_CHUNK*`).
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

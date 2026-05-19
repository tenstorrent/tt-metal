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

Tests are marked `models_performance_bare_metal` and expect a configured Blackhole mesh (`MESH_DEVICE` etc., following repo conventions for bare-metal pytest).

Examples (from repo root):

```sh
pytest models/experimental/devstarl2_small/tests/pipeline_tests/test_devstral2_small.py::test_devstral2_small_projected_image_features_pcc
```

```sh
pytest models/experimental/devstarl2_small/tests/pipeline_tests/test_ministral3_model.py::test_ministral3_model_pcc_devstral_weights
```

```sh
pytest models/experimental/devstarl2_small/tests/pipeline_tests/test_pixtral_vision_model.py::test_pixtral_vision_model_pcc_devstral_weights
```

Run a whole file (longer):

```sh
pytest models/experimental/devstarl2_small/tests/test_ministralattn.py
```

## Demos

### Image + text (`tt_image_demo.py`)

TT backend (example mesh / vision sizing):

```sh
python3 -m models.experimental.devstarl2_small.demo.tt_image_demo \
  --backend tt \
  --image models/experimental/devstarl2_small/resource/sample.jpeg \
  --vision-square-pixels 1540 \
  --mesh-width 4 \
  --max-new-tokens 100
```

### Text LM on TT (`tt_text_demo.py`)

```sh
python models/experimental/devstarl2_small/demo/tt_text_demo.py
```

### Agent on TT (`tt_demo_agent.py`)

```sh
python models/experimental/devstarl2_small/demo/tt_demo_agent.py --vision-square-pixels 1540
```

Sample multimodal prompt (attach the resource image, then ask a question):

```text
/image ./models/experimental/devstarl2_small/resource/sample.jpeg What is in this image?
```

Sample text prompt (coding task):

```text
Can you implement in Python a method to compute the fibonnaci sequence at the `n`th element with `n` a parameter passed to the function ? You should start the sequence from 1, previous values are invalid.
Then run the Python code for the function for n=5 and give the answer.
```

## Resources

| Path | Purpose |
|------|---------|
| **`demo/`** | Runnable demos: `tt_image_demo.py` (HF or TT multimodal), `tt_text_demo.py` (TT text LM), `tt_demo_agent.py` (interactive TT agent with `/image`). |
| **`resource/`** | Static assets only (e.g. `sample.jpeg` for image demos). No Python modules. |
| **`devstral_utils/`** | Shared helpers imported by demos, tests, and TT modules (re-exported from `devstral_utils/__init__.py`). |
| **`tt/`** | TT layer implementations (Ministral3 + Pixtral building blocks). |
| **`tt/pipeline/`** | Composed models: vision → projector → text LM. |
| **`tests/`** | Per-op PCC tests (attention, MLP, norms, Pixtral blocks, decoder layer, …). |
| **`tests/pipeline_tests/`** | End-to-end PCC for `TtDevstral2SmallModel`, `TtMinistral3Model`, vision tower, and projector. |

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
- Pipeline tests under `tests/pipeline_tests/` validate composed vision, text, and full multimodal paths.

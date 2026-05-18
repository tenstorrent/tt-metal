# Devstral Small 2 (experimental)

## Platforms

Blackhole only.

## Introduction

This folder contains an experimental Tenstorrent (`ttnn`) port of **Mistral [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)** (Mistral3 multimodal): Pixtral-class vision tower, multimodal projector, and Ministral3 text stack. PCC tests compare subgraphs and full vision+projector paths against Hugging Face references.

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
pytest models/experimental/devstarl2_small/tests/test_devstral2_small.py::test_devstral2_small_projected_image_features_pcc
```

```sh
pytest models/experimental/devstarl2_small/tests/test_ministral3_model.py::test_ministral3_model_pcc_devstral_weights
```

```sh
pytest models/experimental/devstarl2_small/tests/test_pixtral_vision_model.py::test_pixtral_vision_model_pcc_devstral_weights
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

For a PyTorch-only agent loop, see `demo_agent.py --help`.

## Resources

- **`resource/`** — static demo assets only (for example `sample.jpeg` for `tt_image_demo.py`). Put images here; do not add Python modules under this folder.
- **`devstral_utils/chat_reference.py`** — chat template constants (`SP`, `REFERENCE_MESSAGES`, `REFERENCE_TOOLS`, `REFERENCE_GENERATE_KWARGS`, …) imported by `tt_text_demo.py` and `tt_image_demo.py`.

## Details

- Entry point for the full TT multimodal backbone: `models/experimental/devstarl2_small/tt/tt_devstral2_small_model.py` (`TtDevstral2SmallModel`).
- Text stack: `tt/tt_ministral3_model.py`, decoder layers, attention, MLP, RMSNorm, rotary embedding under `tt/`.
- Vision: Pixtral tower and helpers under `tt/` (`tt_pixtral_*.py`, `tt_patchmerger.py`, etc.).
- Shared demo utilities: `devstral_utils/` (mesh open/close, prefill padding, FP8 scalar-scale shim, decode trace helpers, chat reference constants).
- Default HF model id used in demos/helpers: `mistralai/Devstral-Small-2-24B-Instruct-2512` (override with `HF_MODEL` / `--model-id` where supported).

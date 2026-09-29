# DiffusionGemma 26B-A4B (bring-up)

## Introduction

DiffusionGemma 26B-A4B-it is a discrete **text-diffusion** LLM fine-tuned from Gemma-4 26B-A4B (MoE).
Its text backbone is identical to [`models/demos/gemma4`](../../demos/gemma4) and is reused unchanged;
the net-new work is the block-autoregressive multi-canvas **generation procedure** — bidirectional
canvas attention, a three-phase KV-cache state machine, entropy-budget acceptance sampling and
self-conditioning. Platform: Blackhole **QB2 only**. The module has a traced denoise loop and a serving
adapter (`tt/serving.py`, `tt/generator_vllm.py`) that serves through the standalone vllm-tt-plugin.

Layout: `reference/` pure-torch oracle + drift guard, `tt/` on-device (ttnn) modules, `tests/` CPU +
QB2 suites. `weight_mapping.py` remaps the DiffusionGemma checkpoint (`model.decoder.*`) onto the
unmodified gemma4 loader (`model.language_model.*`).

## How to run

```sh
# CPU reference + parity tests (device tests auto-skip)
pytest models/experimental/diffusion_gemma/tests -q

# QB2 device validation (4x Blackhole)
DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 HF_MODEL=<path to gemma-4-26B-A4B-it> \
  pytest models/experimental/diffusion_gemma/tests -q -s -k 1x4
```

## Notes

- The reused gemma4 MoE backbone PCCs ~0.88 vs HF on Blackhole (recorded as xfail against the 0.99
  target). That is the known gemma4 MoE fidelity, **not** a DiffusionGemma defect.
- Parent issue: tenstorrent/tt-metal#47452.

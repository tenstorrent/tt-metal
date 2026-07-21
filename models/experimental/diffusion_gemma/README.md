# DiffusionGemma 26B-A4B (bring-up)

## Platforms
Blackhole — QB2

## Introduction
DiffusionGemma 26B-A4B-it is a discrete **text-diffusion** LLM fine-tuned from
Gemma-4 26B-A4B (MoE). Its text backbone is identical to
[`models/demos/gemma4`](../../demos/gemma4) and is reused unchanged; the net-new
work is the block-autoregressive multi-canvas **generation procedure**:
bidirectional canvas attention, a three-phase KV-cache state machine,
entropy-budget acceptance sampling, and self-conditioning.

## Status
This directory is the **foundation layer**, not yet an end-to-end demo:

- `reference/` — pure-torch oracle (sampling/entropy/Gumbel-max, denoise loop,
  self-conditioning, canvas attention mask): the PCC ground truth.
- `tests/test_*_parity.py` — guard that the reference reproduces HF
  `transformers` `diffusion_gemma` bit-for-bit (drift oracle).
- `tt/` — net-new on-device (ttnn) primitives: entropy/Gumbel-max sampling and
  the self-conditioning gated MLP.
- `weight_mapping.py` — remaps the DiffusionGemma checkpoint (`model.decoder.*`)
  onto the unmodified gemma4 loader (`model.language_model.*`).
- `tests/test_device_*` — QB2 validation: backbone logits PCC, entropy/accept
  chain, self-conditioning, and the 256K weights+KV memory budget.

The bidirectional encoder-decoder forward, the discrete-diffusion decode loop,
and text-only e2e generation are tracked as follow-ups (#47462 / #47463 /
#47464 / #47474).

## How to Run

CPU reference + parity tests (no device; device tests auto-skip):
```sh
pytest models/experimental/diffusion_gemma/tests -q
```

QB2 device validation (4× Blackhole):
```sh
DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 HF_MODEL=<path to gemma-4-26B-A4B-it> \
  pytest models/experimental/diffusion_gemma/tests -q -s -k 1x4
```

## Notes
- The reused gemma4 MoE backbone PCCs ~0.88 vs the HF reference on Blackhole
  (recorded as xfail against the 0.99 target — this is the known gemma4 MoE
  fidelity, not a DiffusionGemma defect).
- Parent issue: tenstorrent/tt-metal#47452.

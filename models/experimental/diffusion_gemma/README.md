# DiffusionGemma 26B-A4B (bring-up)

Status: current — front door for the module.
Owns: what this directory is and how to run its tests.
See also: [plan + execution contract](plan.md) · [refuted list](doc/REFUTED.md) · [agent guide](AGENTS.md)

## Introduction

DiffusionGemma 26B-A4B-it is a discrete **text-diffusion** LLM fine-tuned from Gemma-4 26B-A4B (MoE).
Its text backbone is identical to [`models/demos/gemma4`](../../demos/gemma4) and is reused unchanged;
the net-new work is the block-autoregressive multi-canvas **generation procedure** — bidirectional
canvas attention, a three-phase KV-cache state machine, entropy-budget acceptance sampling and
self-conditioning. Platform: Blackhole **QB2 only**. The module has a traced denoise loop, a serving
adapter (`tt/serving.py`, `tt/generator_vllm.py`) and GPQA-scale evals; [plan.md](plan.md) holds the
model facts and the current launch/metric contract.

Layout: `reference/` pure-torch oracle + drift guard, `tt/` on-device (ttnn) modules, `tests/` CPU +
QB2 suites, `doc/` per-area evidence ([decision fidelity](doc/decision_fidelity/README.md),
[perf](doc/optimize_perf/README.md), [serving](doc/vllm_integration/README.md),
[precision](doc/datatype_sweep/README.md)). `weight_mapping.py` remaps the DiffusionGemma checkpoint
(`model.decoder.*`) onto the unmodified gemma4 loader (`model.language_model.*`).

## How to run

```sh
# CPU reference + parity tests (device tests auto-skip)
pytest models/experimental/diffusion_gemma/tests -q

# QB2 device validation (4x Blackhole); env: see plan.md
DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 HF_MODEL=<path to gemma-4-26B-A4B-it> \
  pytest models/experimental/diffusion_gemma/tests -q -s -k 1x4
```

## Notes

- The reused gemma4 MoE backbone PCCs ~0.88 vs HF on Blackhole (recorded as xfail against the 0.99
  target). That is the known gemma4 MoE fidelity, **not** a DiffusionGemma defect.
- Parent issue: tenstorrent/tt-metal#47452.

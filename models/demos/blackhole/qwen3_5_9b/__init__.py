# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 hybrid text model for Blackhole P150 — a config-driven ttnn port of
transformers' Qwen3.5 family (the 9B, and the 27B 3.6 multimodal checkpoint's text
backbone, which runs through the exact same code).

Why all the __init__.py files: every directory here is a Python package, and Python
only treats a directory as an importable package if it contains an __init__.py. The
codebase imports modules by fully-qualified path
(``models.demos.blackhole.qwen3_5_9b.tt.model``), so each level needs one to exist.
Rather than leave them empty, each __init__.py carries a short docstring describing
what lives in that package, and the leaf packages re-export their main class so callers
can write ``from ...tt.mlp import Qwen35MLP`` instead of reaching into the module file.

Where to start reading:
  * tt/          — the model implementation (start at tt/model.py -> Qwen35Model).
  * demo/demo.py — runnable end-to-end text-generation demo; the easiest entry point.
  * tests/unit/  — per-component tests against the HF golden; the spec for each piece.

Layout:
  * tt/model.py        — Qwen35Model: embedding + N decoder layers + final norm + LM head,
                         plus the prefill/decode/generate drivers. The top-level class.
  * tt/layer.py        — Qwen35DecoderLayer: one hybrid layer (full-attention OR Gated DeltaNet).
  * tt/model_config.py — Qwen35ModelArgs: all config, driven by the HF_MODEL checkpoint.
  * tt/attention/      — Qwen35Attention: full (softmax) attention token mixer.
  * tt/gdn/            — Qwen35GatedDeltaNet: Gated DeltaNet (linear attention) token mixer.
  * tt/mlp/            — Qwen35MLP: the SwiGLU MLP.
  * tt/rms_norm.py     — Qwen35RMSNorm.
  * tt/tp_common.py / tt/weight_mapping.py — tensor-parallel sharding + HF->internal key remap.
  * demo/trace_runner.py — TracedRunner: capture/replay the model as a ttnn trace.
  * utils/             — small host-side helpers (cache paths, etc.).
"""

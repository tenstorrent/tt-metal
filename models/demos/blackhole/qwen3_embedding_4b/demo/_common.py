# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Embedding-4B demo plumbing: a thin shim over the optimized pplx-embed-4B stack.

Qwen3-Embedding-4B is the backbone pplx-embed-v1-4B was trained from (2560 hidden, 9728 FFN, 36 layers,
32 Q / 8 KV heads, head dim 128), so the whole tuned stack in ``models/demos/blackhole/pplx_embed_4b``
runs it unchanged. Setting ``HF_MODEL`` before the import is all it takes: the stack then keeps the
checkpoint's causal attention (``QWEN_SDPA_CAUSAL=1``) and labels its output with the model name. The
embedding is the last real token (the traced pipeline slices the last-token tile); appending the EOS
token is the Qwen3-Embedding recipe. Per-batch tuned defaults come from ``apply_workload_env``; the
knobs are documented next to each default in the pplx-embed-4B ``_common.py`` and in
``../pplx_embed_4b/PERF.md``.
"""

import os

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
os.environ.setdefault("HF_MODEL", MODEL_NAME)

from models.demos.blackhole.pplx_embed_4b.demo._common import (  # noqa: E402  (HF_MODEL must be set first)
    HIDDEN_DIM,
    apply_recommended_env,
    apply_workload_env,
    build_single_device_model,
    generate_synthetic_inputs,
    run_perf,
    standalone_main,
)

__all__ = [
    "HIDDEN_DIM",
    "MODEL_NAME",
    "apply_recommended_env",
    "apply_workload_env",
    "build_single_device_model",
    "generate_synthetic_inputs",
    "run_perf",
    "standalone_main",
]

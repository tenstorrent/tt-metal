# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the grpo_speedup update-method tests.

These are intentionally regular functions (not pytest fixtures) so each
test module can wrap them in a module-scoped fixture with whatever
extras it needs (e.g. yielding a specific layer or sub-module).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"
MAX_SEQ_LEN = 2048

REPO_ROOT = Path(__file__).resolve().parents[5]  # .../tt-metal


def build_completer(*, dummy_weights: bool, max_batch_size: int = 1, max_seq_len: int = MAX_SEQ_LEN):
    """Construct a fresh ``LlamaGRPOCompleter``.

    Heavy: opens a device and (when ``dummy_weights=False``) loads real
    Llama-3.2-1B-Instruct weights via HF auth. Tests should call this
    from a module-scoped fixture so the cost is paid once per file.
    """
    from ttml.common.config import DeviceConfig, load_config

    from utils.llama_completer_ttt import LlamaGRPOCompleter

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)
    return LlamaGRPOCompleter(
        device_config=device_config,
        model_source=MODEL_ID,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dummy_weights=dummy_weights,
    )


def teardown_completer(completer) -> None:
    """Drop ``completer`` and close the device. Call from a fixture's
    teardown so the device is released even if a test fails."""
    import ttml

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()


def as_update_input(t, mesh_device):
    """Build the HF-format on-device input that every ``.update()`` method
    accepts.

    Caller passes the natural HF shape (matches HF safetensors):
        * Linear weight   -- ``(out_features, in_features)``   (2D)
        * Embedding table -- ``(vocab_size,   hidden_size)``   (2D)
        * RMSNorm gamma   -- ``(dim,)``                        (1D)
        * Linear bias     -- ``(out_features,)``               (1D)

    Returns the canonical update() input: 4D ``(1, 1, ..., ...)``
    ``ttnn.Tensor``, replicated across the mesh, DRAM-interleaved,
    ``TILE_LAYOUT``, ``bfloat16``. Non-bf16 inputs are cast; non-contiguous
    inputs (e.g. fresh from ``.transpose``) are made contiguous before the
    upload.
    """
    import torch
    import ttnn

    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def to_torch_2d(t):
    """``ttnn.to_torch`` + strip leading unit dims so callers get the
    natural ``(rows, cols)`` shape back. Internal buffers in TTT are
    stored as 4D ``(1, 1, rows, cols)``; squeezing those down keeps the
    per-module snapshot inverses readable.
    """
    import ttnn

    out = ttnn.to_torch(t)
    while out.dim() > 2 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out

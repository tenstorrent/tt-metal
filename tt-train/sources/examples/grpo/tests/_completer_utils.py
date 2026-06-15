# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the grpo_speedup update-method tests.

These are intentionally regular functions (not pytest fixtures) so each
test module can wrap them in a module-scoped fixture with whatever
extras it needs (e.g. yielding a specific layer or sub-module).
"""

from __future__ import annotations

import contextlib
import gc
import os
from pathlib import Path
from typing import Any

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"
MAX_SEQ_LEN = 2048

REPO_ROOT = Path(__file__).resolve().parents[5]  # .../tt-metal


def load_device_config(device_config_rel: str = TTML_DEVICE_CONFIG_REL):
    """Read the standard ttml training yaml and parse its device section.

    Returns ``(device_config, raw)``: ``device_config`` is a
    :class:`ttml.common.config.DeviceConfig` ready to be passed to
    :func:`open_device`; ``raw`` is the parsed yaml dict, useful when a
    caller also needs ``raw["training_config"]["model_config"]`` to
    build a :class:`TransformerConfig` for the ttml completer.
    """
    from ttml.common.config import DeviceConfig, load_config

    raw = load_config(os.path.join(REPO_ROOT, device_config_rel))
    return DeviceConfig(raw), raw


def open_device(device_config) -> Any:
    """Open the ttml ``AutoContext`` device for the given config.

    Enables fabric (when multi-device) and opens the AutoContext mesh.
    Returns the ``ttnn.MeshDevice`` handle that can be passed to both
    :class:`LlamaCompleterTtt` and :class:`LlamaCompleterTtml`.

    Caller owns the lifetime: pair every ``open_device`` with a
    matching :func:`close_device` in a ``try/finally``.
    """
    import ttml

    if device_config.total_devices() > 1:
        ttml.core.distributed.enable_fabric(device_config.total_devices())
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    return autograd_ctx.get_device()


def close_device() -> None:
    """Close the ``AutoContext`` device opened by :func:`open_device`."""
    import ttml

    ttml.autograd.AutoContext.get_instance().close_device()


def build_completer(
    mesh_device: Any,
    *,
    dummy_weights: bool,
    max_batch_size: int = 1,
    max_seq_len: int = MAX_SEQ_LEN,
    model_source: str = MODEL_ID,
    instruct: bool = True,
):
    """Construct a fresh :class:`LlamaCompleterTtt` on ``mesh_device``.

    Heavy when ``dummy_weights=False``: loads real HF weights for
    ``model_source`` (default Llama-3.2-1B-Instruct). Tests should call
    this from a module-scoped fixture so the cost is paid once per file.
    """
    from utils.llama_completer_ttt import LlamaCompleterTtt

    return LlamaCompleterTtt(
        mesh_device=mesh_device,
        model_source=model_source,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        instruct=instruct,
        dummy_weights=dummy_weights,
    )


@contextlib.contextmanager
def open_completer(
    *,
    dummy_weights: bool,
    max_batch_size: int = 1,
    max_seq_len: int = MAX_SEQ_LEN,
    model_source: str = MODEL_ID,
    instruct: bool = True,
):
    """Context manager that opens the device, builds a TTT completer,
    and tears both down on exit.

    Cleanup runs in dependency order: drop the completer (freeing its
    on-device tensors), GC, then close the AutoContext mesh. Safe on
    construction failure (a partially-built completer is still dropped
    before close_device).

    Usage::

        @pytest.fixture(scope="module")
        def attn():
            with open_completer(dummy_weights=True) as c:
                yield c.model.layers[0].attention
    """
    device_config, _ = load_device_config()
    mesh_device = open_device(device_config)
    completer = None
    try:
        completer = build_completer(
            mesh_device,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            model_source=model_source,
            instruct=instruct,
        )
        yield completer
    finally:
        completer = None
        gc.collect()
        close_device()


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

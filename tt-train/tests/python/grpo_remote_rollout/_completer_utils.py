# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the grpo_speedup update-method tests."""

from __future__ import annotations

import contextlib
import functools
import gc
import os
from pathlib import Path
from typing import Any

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1b_1dev.yaml"
MAX_SEQ_LEN = 2048

# Nonzero trace region required for tt-transformers decode trace; 50 MB matches
# the tt-transformers demo default.
_TRACE_REGION_SIZE = 50_000_000

REPO_ROOT = Path(__file__).resolve().parents[5]  # .../tt-metal


def load_device_config(device_config_rel: str = TTML_DEVICE_CONFIG_REL):
    """Parse the ttml training yaml; return ``(DeviceConfig, raw_dict)``."""
    from ttml.common.config import DeviceConfig, load_config

    raw = load_config(os.path.join(REPO_ROOT, device_config_rel))
    return DeviceConfig(raw), raw


def open_device(device_config) -> Any:
    """Open the ttml ``AutoContext`` mesh (enabling fabric when multi-device);
    return the ``ttnn.MeshDevice``. Pair with :func:`close_device`."""
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


class _TttCompleter:
    """Adapter over :class:`TttGenerationWorker` that forwards its attributes
    and adds a lazily-loaded ``.tokenizer`` so dummy-weight tests stay
    HF-auth-free (only tests that touch ``.tokenizer`` pay for the download)."""

    def __init__(self, worker: Any, model_source: str) -> None:
        self._worker = worker
        self._model_source = model_source

    @functools.cached_property
    def tokenizer(self) -> Any:
        # Reuse the ModelArgs tokenizer if already loaded; else load on demand.
        tok = getattr(self._worker.model_args[0], "tokenizer", None)
        if tok is not None:
            return tok
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self._model_source)

    def __getattr__(self, name: str) -> Any:
        # Forward unknown attrs to the worker; guard ``_worker`` against
        # recursion before __init__ sets it.
        if name == "_worker":
            raise AttributeError(name)
        return getattr(self._worker, name)


def build_completer(
    mesh_device: Any,
    *,
    dummy_weights: bool,
    max_batch_size: int = 1,
    max_seq_len: int = MAX_SEQ_LEN,
    model_source: str = MODEL_ID,
    instruct: bool = True,
):
    """Build a :class:`TttGenerationWorker` wrapped in :class:`_TttCompleter`.

    Heavy when ``dummy_weights=False`` (loads real HF weights); call from a
    module-scoped fixture so the cost is paid once per file.
    """
    from utils.llama_ttt_presets import (
        bf16_attn_bfp8_mlp_optimizations,
        llama_stop_and_pad,
    )
    from utils.ttt_generation_worker import TttGenerationWorker

    if dummy_weights:
        # Skip the gated HF tokenizer; stop/pad IDs are immaterial here.
        stop_token_ids: Any = ()
        pad_token_id = 0
    else:
        stop_token_ids, pad_token_id = llama_stop_and_pad(model_source)

    worker = TttGenerationWorker(
        mesh_device=mesh_device,
        model_source=model_source,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        instruct=instruct,
        optimizations=bf16_attn_bfp8_mlp_optimizations,
        stop_token_ids=stop_token_ids,
        pad_token_id=pad_token_id,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        seed=0,
        dummy_weights=dummy_weights,
    )
    return _TttCompleter(worker, model_source)


@contextlib.contextmanager
def open_completer(
    *,
    dummy_weights: bool,
    max_batch_size: int = 1,
    max_seq_len: int = MAX_SEQ_LEN,
    model_source: str = MODEL_ID,
    instruct: bool = True,
):
    """Open the device, build a TTT completer, and tear both down on exit.

    Cleanup order matters: drop the completer (freeing its on-device tensors),
    GC, then close the mesh.
    """
    import ttnn

    device_config, _ = load_device_config()
    # HARDWARE-SPECIFIC: open via ttnn (not ttml AutoContext) with ETH dispatch
    # so tt-transformers keeps the full 8x8 tensix grid (WORKER dispatch leaves
    # only 8x7 on a 2-harvested N300), plus the trace region the decode needs.
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*device_config.mesh_shape),
        trace_region_size=_TRACE_REGION_SIZE,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.device.DispatchCoreType.ETH),
    )
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
        ttnn.close_mesh_device(mesh_device)


def as_update_input(t, mesh_device):
    """Upload a natural-HF-shape tensor as the canonical ``.update()`` input:
    4D ``bfloat16`` ``ttnn.Tensor``, mesh-replicated, DRAM-interleaved,
    ``TILE_LAYOUT``. Casts non-bf16 and makes non-contiguous inputs contiguous.
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
    """``ttnn.to_torch`` + strip leading unit dims to the natural ``(rows,
    cols)`` shape (TTT buffers are stored 4D as ``(1, 1, rows, cols)``)."""
    import ttnn

    out = ttnn.to_torch(t)
    while out.dim() > 2 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out

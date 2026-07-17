# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Thin executors for ``Qwen25_72B`` delegating to shared ``EagerLLMExecutor`` /
``TracedLLMExecutor`` (paged KV, page tables, compile).
"""

from __future__ import annotations

import ttnn
from models.common.models.executor import EagerLLMExecutor, TracedLLMExecutor
from models.common.models.qwen25_72b.model import Qwen25_72B, _slice_last_token_tile


def _iter_qwen25_72b_executor_named_modules(model: Qwen25_72B):
    yield ("embed", model.embed)
    yield ("rope_setup", model.rope_setup)
    for i, layer in enumerate(model.layers):
        yield (f"layer{i}", layer)
    yield ("norm", model.norm)
    yield ("lm_head", model.lm_head)


class EagerQwen25_72BExecutor:
    """Delegates to ``EagerLLMExecutor``; attaches ``model_args`` on the model for the engine."""

    def __init__(self, model: Qwen25_72B, mesh_device: ttnn.MeshDevice):
        self._engine = EagerLLMExecutor(model, mesh_device, iter_named_modules=_iter_qwen25_72b_executor_named_modules)

    @property
    def model(self) -> Qwen25_72B:
        return self._engine.model

    @property
    def mesh_device(self) -> ttnn.MeshDevice:
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def compile_prefill(self, **kwargs):
        return self._engine.compile_prefill(**kwargs)

    def compile_decode(self, **kwargs):
        return self._engine.compile_decode(**kwargs)

    def prefill_forward(self, *args, **kwargs):
        return self._engine.prefill_forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return self._engine.decode_forward(*args, **kwargs)

    def cleanup(self) -> None:
        return self._engine.cleanup()


class TracedQwen25_72BExecutor:
    """Traced path; same surface as ``EagerQwen25_72BExecutor``."""

    def __init__(self, model: Qwen25_72B, mesh_device: ttnn.MeshDevice):
        self._engine = TracedLLMExecutor(model, mesh_device, iter_named_modules=_iter_qwen25_72b_executor_named_modules)

    @property
    def model(self) -> Qwen25_72B:
        return self._engine.model

    @property
    def mesh_device(self) -> ttnn.MeshDevice:
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def trace_id_prefill(self):
        return self._engine.trace_id_prefill

    @property
    def trace_ids_decode(self):
        return self._engine.trace_ids_decode

    @property
    def already_warmed_up_prefill(self):
        return self._engine.already_warmed_up_prefill

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def compile_prefill(self, **kwargs):
        return self._engine.compile_prefill(**kwargs)

    def compile_decode(self, **kwargs):
        return self._engine.compile_decode(**kwargs)

    def prefill_forward(self, *args, **kwargs):
        return self._engine.prefill_forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return self._engine.decode_forward(*args, **kwargs)

    def cleanup(self) -> None:
        return self._engine.cleanup()


def run_prefill(model: Qwen25_72B, token_ids_tt: ttnn.Tensor, *, start_pos: int = 0) -> ttnn.Tensor:
    """Prefill chunk without paged executor (tests): ``token_ids_tt`` shape ``[1,1,1,S]``, ``S % 128 == 0``."""
    return model.prefill_from_token_ids(token_ids_tt, start_pos=start_pos)


def run_decode(model: Qwen25_72B, token_id_tt: ttnn.Tensor, *, current_pos: int) -> ttnn.Tensor:
    """Single-token decode without page table (tests)."""
    return model.decode_from_token_ids(token_id_tt, current_pos=current_pos)


def run_lm_head(model: Qwen25_72B, hidden_tt: ttnn.Tensor) -> ttnn.Tensor:
    """Last-tile hidden slice so width-sharded LM matmul M matches ``LMHead1D`` program config."""
    if len(hidden_tt.shape) == 4 and hidden_tt.shape[2] > 32:
        old = hidden_tt
        hidden_tt = _slice_last_token_tile(old, hidden_tt.shape[2] - 1)
        ttnn.deallocate(old)
    return model.lm_logits(hidden_tt)

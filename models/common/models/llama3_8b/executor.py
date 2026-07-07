# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1-8B executor wrappers."""

from __future__ import annotations

import ttnn
from models.common.models.executor import EagerLLMExecutor, TracedLLMExecutor
from models.common.models.llama3_8b.model import Llama3Transformer1D


class EagerLlamaExecutor:
    """Thin wrapper: passes Llama model to EagerLLMExecutor.

    All actual logic lives in the engine. This class exists to:
    1. Provide a model-specific type for type hints
    2. Preserve the existing API for demos and tests
    """

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        # Attach model_args to model so engine can access it via model.model_args
        if model_args is not None:
            model.model_args = model_args
        self._engine = EagerLLMExecutor(model, mesh_device, iter_named_modules=_iter_llama_executor_named_modules)

    @property
    def model(self):
        return self._engine.model

    @property
    def mesh_device(self):
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def mode(self):
        return self._engine.mode

    @mode.setter
    def mode(self, value):
        self._engine.mode = value

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def _assert_kv_cache_identity(self, kv_cache):
        return self._engine._assert_kv_cache_identity(kv_cache)

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table):
        return self._engine.prepare_decode_inputs_host(tokens, current_pos, page_table)

    def prepare_decode_inputs_device(self, tokens, current_pos, page_table):
        return self._engine.prepare_decode_inputs_device(tokens, current_pos, page_table)

    def compile_prefill(
        self,
        *,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
    ):
        return self._engine.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens,
        start_pos,
        page_table,
        kv_cache=None,
        sampling_params=None,
    ):
        return self._engine.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
    ):
        return self._engine.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
        )

    def _prefill_single_user(self, tokens, page_table, user_id, last_token_idx, num_cached_tokens=0):
        return self._engine._prefill_single_user(tokens, page_table, user_id, last_token_idx, num_cached_tokens)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache=None,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
    ):
        return self._engine.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    def cleanup(self):
        return self._engine.cleanup()


class TracedLlamaExecutor:
    """Thin wrapper: passes Llama model to TracedLLMExecutor.

    All actual logic lives in the engine. This class exists to:
    1. Provide a model-specific type for type hints
    2. Preserve the existing API for demos and tests
    """

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        # Attach model_args to model so engine can access it via model.model_args
        if model_args is not None:
            model.model_args = model_args
        self._engine = TracedLLMExecutor(model, mesh_device, iter_named_modules=_iter_llama_executor_named_modules)

    @property
    def model(self):
        return self._engine.model

    @property
    def mesh_device(self):
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def mode(self):
        return self._engine.mode

    @mode.setter
    def mode(self, value):
        self._engine.mode = value

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

    def warmup_model_prefill(self, seq_lens, make_tokens, make_page_table):
        return self._engine.warmup_model_prefill(seq_lens, make_tokens, make_page_table)

    def compile_prefill(
        self,
        *,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
    ):
        return self._engine.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens,
        start_pos,
        page_table,
        kv_cache=None,
        sampling_params=None,
    ):
        return self._engine.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
    ):
        return self._engine.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
        )

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache=None,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
    ):
        return self._engine.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    def cleanup(self):
        return self._engine.cleanup()


def _iter_llama_executor_named_modules(model):
    """Yield named submodules that declare executor input contracts."""
    if not hasattr(model, "layers"):
        return

    for i, layer in enumerate(model.layers):
        for suffix, submodule in [
            ("attn_norm", getattr(layer, "attention_norm", None)),
            ("attention", getattr(layer, "attention", None)),
            ("ff_norm", getattr(layer, "ff_norm", None)),
            ("mlp", getattr(layer, "feed_forward", None)),
        ]:
            if submodule is not None:
                yield f"layer[{i}].{suffix}", submodule

    if hasattr(model, "norm"):
        yield "final_norm", model.norm
    if hasattr(model, "lm_head"):
        yield "lm_head", model.lm_head

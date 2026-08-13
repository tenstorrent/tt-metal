# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Thin vLLM adapter for Mistral-Small-24B-Instruct-2501.

The full-model generator owns execution and the split Sampling1D trace.  This
module only translates the vLLM interface and binds the caller-owned paged KV
cache; it deliberately has no adapter-local sampler or token-feedback loop.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.generator import (
    MistralSmall24BGenerator,
    build_generator,
)
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.model import PAGED_BLOCK_SIZE

MODEL_DIR = Path(__file__).resolve().parents[1]
MAX_CONTEXT_LEN = 32_768
MAX_BATCH_SIZE = 32
HOST_SAMPLING_COMPAT_ENV = "MISTRAL_SMALL_24B_VLLM_HOST_SAMPLING_COMPAT"
REDUCED_LAYERS_ENV = "MISTRAL_SMALL_24B_VLLM_REDUCED_LAYERS"


def _host_sampling_compat_enabled() -> bool:
    return os.getenv(HOST_SAMPLING_COMPAT_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _sampling_value(sampling_params: Any, name: str, default: float | int, active_batch: int):
    value = getattr(sampling_params, name, None)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        return value
    values = list(value)[:active_batch]
    if len(values) != active_batch:
        raise ValueError(f"sampling parameter {name} has {len(values)} rows for batch {active_batch}")
    return values


def _canonical_sampling_args(sampling_params: Any, active_batch: int) -> dict[str, Any]:
    top_k = _sampling_value(sampling_params, "top_k", 1, active_batch)
    top_p = _sampling_value(sampling_params, "top_p", 1.0, active_batch)
    temperature = _sampling_value(sampling_params, "temperature", 1.0, active_batch)

    def _rows(value):
        return list(value) if isinstance(value, list) else [value] * active_batch

    k_rows = [32 if int(value) < 1 else int(value) for value in _rows(top_k)]
    p_rows = [float(value) for value in _rows(top_p)]
    temp_rows = [float(value) for value in _rows(temperature)]
    for row, temp in enumerate(temp_rows):
        if temp <= 0.0:
            # Sampling1D's canonical deterministic encoding avoids passing a
            # zero temperature into the stochastic kernel.
            k_rows[row], p_rows[row], temp_rows[row] = 1, 0.0, 1.0
    return {
        "top_k": k_rows,
        "top_p": p_rows,
        "temperature": temp_rows,
    }


class TTMistralSmall24BForCausalLM:
    """vLLM translation layer over :class:`MistralSmall24BGenerator`."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_device_penalties": False,
        "supports_device_seeds": False,
    }

    def __init__(self, generator: MistralSmall24BGenerator):
        self.generator = generator
        self.model = generator.model
        self.mesh_device = generator.mesh_device
        self.max_batch_size = generator.batch
        self._vllm_kv_cache = None
        self.already_warmed_up_prefill = False

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs: Any,
    ) -> int:
        del model_name, kwargs
        if num_devices // tt_data_parallel != 4:
            raise ValueError("Mistral Small 24B serving requires one four-device TP mesh")
        if tt_data_parallel != 1:
            raise ValueError("Mistral Small 24B vLLM data parallelism is not implemented")
        if max_model_len is not None and not 1 <= int(max_model_len) <= MAX_CONTEXT_LEN:
            raise ValueError(f"max_model_len must be in [1, {MAX_CONTEXT_LEN}]")
        if max_num_seqs is not None and not 1 <= int(max_num_seqs) <= MAX_BATCH_SIZE:
            raise ValueError(f"max_num_seqs must be in [1, {MAX_BATCH_SIZE}]")
        return MAX_CONTEXT_LEN

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=None,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs: Any,
    ):
        if optimizations is not None:
            raise ValueError(
                "adapter-local optimization presets are disabled; the selected datatype policy is mandatory"
            )
        max_seq_len = int(max_seq_len or hf_config.max_position_embeddings)
        cls.get_max_tokens_all_users(
            model_name=getattr(hf_config, "_name_or_path", ""),
            num_devices=mesh_device.get_num_devices(),
            tt_data_parallel=tt_data_parallel,
            max_model_len=max_seq_len,
            max_num_seqs=max_batch_size,
        )
        # The worker adds one scheduler-reservation block per request to the
        # 32K pooled token budget.  Match that exact caller-owned allocation.
        num_blocks = math.ceil(max_seq_len / PAGED_BLOCK_SIZE) + int(max_batch_size)
        snapshot_path = getattr(hf_config, "_name_or_path", None)
        if snapshot_path is not None and not (Path(snapshot_path) / "config.json").is_file():
            snapshot_path = None
        generator = build_generator(
            MODEL_DIR,
            mesh_device,
            snapshot_path=snapshot_path,
            max_batch_size=int(max_batch_size),
            max_seq_len=max_seq_len,
            num_blocks=num_blocks,
            pooled_kv_cache=True,
            override_num_layers=(int(os.environ[REDUCED_LAYERS_ENV]) if REDUCED_LAYERS_ENV in os.environ else None),
            **kwargs,
        )
        return cls(generator)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        del dtype
        shape = tuple(int(value) for value in kv_cache_shape)
        if len(shape) != 4 or shape[2] != PAGED_BLOCK_SIZE or shape[3] != self.model.head_dim:
            raise ValueError(f"unexpected vLLM KV-cache shape {shape}")
        if int(num_layers) < self.model.num_layers:
            raise ValueError(f"vLLM requested {num_layers} cache layers; model has {self.model.num_layers}")
        cache = self.model.allocate_kv_cache(
            num_blocks=shape[0],
            dtype=self.model.config.kv_cache_dtype,
        )
        self._vllm_kv_cache = cache
        return cache

    def _require_vllm_cache(self, kv_cache):
        if kv_cache is None or kv_cache is not self._vllm_kv_cache:
            raise ValueError("serving must pass through the exact vLLM-owned KV cache")
        return kv_cache

    def prefill_forward(
        self,
        *,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        sampling_params=None,
        **kwargs: Any,
    ):
        empty_slots = kwargs.pop("empty_slots", None)
        del kwargs
        cache = self._require_vllm_cache(kv_cache)
        lengths = [int(value) for value in prompt_lens]
        self.generator.prepare_for_prefill()
        if empty_slots is not None:
            self.generator.note_prefilled_slots(empty_slots)
        if sampling_params is None:
            if not _host_sampling_compat_enabled():
                raise RuntimeError(
                    f"host sampling is disabled; set {HOST_SAMPLING_COMPAT_ENV}=1 only for compatibility tests"
                )
            return self.generator.prefill_forward(
                tokens,
                page_table=page_table,
                kv_cache=cache,
                prompt_lens=lengths,
            )
        sampled = self.generator.prefill_forward_device_sample(
            tokens,
            page_table=page_table,
            kv_cache=cache,
            prompt_lens=lengths,
            **_canonical_sampling_args(sampling_params, tokens.shape[0]),
        )
        host_sampled = (
            self.generator._sampled_tokens_to_torch(sampled)[: tokens.shape[0]].reshape(-1, 1).to(torch.int32)
        )
        return host_sampled

    def decode_forward(
        self,
        *,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=False,
        sampling_params=None,
        reset_batch=None,
        slot_remap=None,
        **kwargs: Any,
    ):
        del kwargs
        cache = self._require_vllm_cache(kv_cache)
        if sampling_params is None:
            if not _host_sampling_compat_enabled():
                raise RuntimeError(
                    f"host sampling is disabled; set {HOST_SAMPLING_COMPAT_ENV}=1 only for compatibility tests"
                )
            logits = self.generator.decode_forward(
                tokens,
                start_pos,
                page_table=page_table,
                kv_cache=cache,
                sampling_mode="host",
                enable_trace=False,
                reset_batch=reset_batch,
                slot_remap=slot_remap,
            )
            # vLLM's host sampler consumes [batch, sequence, vocabulary].
            return logits.unsqueeze(1) if logits.ndim == 2 else logits
        tt_out = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=cache,
            sampling_mode="device",
            enable_trace=enable_trace,
            reset_batch=reset_batch,
            slot_remap=slot_remap,
            **_canonical_sampling_args(sampling_params, tokens.reshape(-1).numel()),
        )
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(tt_out), is_tokens=True)
        return tt_out

    def read_decode_output(self, tt_out, async_read=False):
        # Shared compatibility tests can explicitly route unsupported sampling
        # features through vLLM's host sampler.  That path already returns a
        # torch tensor; only sampled device tokens need a TTNN transfer.
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        host = ttnn.from_device(tt_out, blocking=not async_read)
        if not async_read:
            return host
        return host, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if not is_tokens:
            raise ValueError("the production adapter formats sampled tokens only")
        shards = ttnn.get_device_tensors(tt_out)
        host = ttnn.to_torch(shards[0] if shards else tt_out)
        return host.reshape(-1)[: self.max_batch_size].to(torch.int32).reshape(-1, 1)

    def warmup_model_prefill(self, *, kv_cache, can_sample_on_device, enable_trace, **kwargs: Any):
        del enable_trace, kwargs
        if self.already_warmed_up_prefill:
            return
        if not can_sample_on_device:
            raise ValueError("production warmup requires on-device prefill sampling")
        self.already_warmed_up_prefill = True
        tokens = torch.zeros((1, 32), dtype=torch.long)
        page_table = torch.zeros((1, 1), dtype=torch.int32)
        params = type("WarmupSampling", (), {"top_k": [1], "top_p": [0.0], "temperature": [0.0]})()
        self.prefill_forward(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[32],
            sampling_params=params,
        )

    def warmup_model_decode(
        self,
        *,
        kv_cache,
        max_batch_size,
        num_blocks,
        can_sample_on_device,
        enable_trace,
        **kwargs: Any,
    ):
        del kwargs
        if not can_sample_on_device:
            raise ValueError("production warmup requires on-device decode sampling")
        if not enable_trace:
            # Trace capture performs its own eager compile/warmup before the
            # capture, preserving the generator's single canonical path.
            return
        batch = int(max_batch_size)
        page_table = torch.zeros((batch, int(num_blocks)), dtype=torch.int32)
        page_table[:, 0] = torch.arange(batch, dtype=torch.int32)
        params = type(
            "WarmupSampling",
            (),
            {"top_k": [1] * batch, "top_p": [0.0] * batch, "temperature": [0.0] * batch},
        )()
        sampling = _canonical_sampling_args(params, batch)
        self.generator.prepare_decode_trace(
            torch.zeros((batch, 1), dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            page_table=page_table,
            kv_cache=kv_cache,
            **sampling,
        )
        # Prefill warmup ran before decode capture and used these same
        # vLLM-owned buffers. Clear both warmup phases so the first real
        # request never updates an already-quantized BFP8 cache tile.
        self.generator.model.reset_kv_cache(kv_cache)
        self.generator._trace_page_table_snapshot = None


__all__ = ["TTMistralSmall24BForCausalLM"]

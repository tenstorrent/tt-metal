# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Thin vLLM bridge for the TP4 Falcon3-7B full-model generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.generator import Falcon3Generator, _resolve_snapshot
from models.autoports.tiiuae_falcon3_7b_base.tt.model import (
    DEFAULT_PAGE_BLOCK_SIZE,
    DEFAULT_PREFILL_CHUNK_SIZE,
    MAX_CONTEXT,
    NUM_LAYERS,
    Falcon3Model,
)


class Falcon3ForCausalLM(Falcon3Generator):
    """Translate vLLM inputs to the canonical Falcon3 generator contract."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_generation_horizon": True,
    }

    def __init__(self, *args, cache_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_path = Path(cache_path)
        self._vllm_active_batch = 0

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=MAX_CONTEXT,
        tt_data_parallel=1,
        optimizations=None,
    ):
        if int(tt_data_parallel) != 1:
            raise ValueError("Falcon3 uses one fixed TP4 mesh and does not support vLLM data parallelism")
        if optimizations is not None:
            raise ValueError("Falcon3 serving always uses doc/datatype_sweep/selected_precision_config.json")
        if int(max_seq_len) != MAX_CONTEXT:
            raise ValueError(f"Falcon3 serving must advertise the context contract ({MAX_CONTEXT}), got {max_seq_len}")

        snapshot = _resolve_snapshot(getattr(hf_config, "_name_or_path", None))
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_dir = Path(__file__).resolve().parents[1]
        weight_cache_path = model_dir / "weight_cache"
        model = Falcon3Model.from_checkpoint(
            snapshot,
            mesh_device=mesh_device,
            max_batch_size=int(max_batch_size),
            max_cache_len=int(max_seq_len),
            num_layers=NUM_LAYERS,
            weight_cache_path=weight_cache_path,
        )
        return cls(
            model,
            tokenizer,
            prefill_chunk_size=DEFAULT_PREFILL_CHUNK_SIZE,
            cache_path=weight_cache_path,
        )

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate the exact vLLM-owned cache; never install a standalone cache."""
        shape = tuple(int(value) for value in kv_cache_shape)
        expected_tail = (1, DEFAULT_PAGE_BLOCK_SIZE, 256)
        if int(num_layers) != self.model.num_layers:
            raise ValueError(f"vLLM requested {num_layers} cache layers; Falcon3 has {self.model.num_layers}")
        if len(shape) != 4 or shape[1:] != expected_tail:
            raise ValueError(f"unexpected Falcon3 vLLM KV-cache shape {shape}; expected (blocks,{expected_tail})")
        if dtype != torch.bfloat16:
            raise ValueError(f"selected Falcon3 KV-cache policy is bfloat16, got {dtype}")
        return self.model.allocate_kv_cache(paged=True, num_blocks=shape[0])

    @staticmethod
    def _sampling_values(sampling_params, active_batch: int) -> dict[str, Any]:
        if sampling_params is None:
            return {"top_k": 1, "top_p": 0.0, "temperature": 0.0, "active_batch": active_batch}
        temperature = [float(value) for value in list(sampling_params.temperature)[:active_batch]]
        top_k = [int(value) for value in list(sampling_params.top_k)[:active_batch]]
        top_p = [float(value) for value in list(sampling_params.top_p)[:active_batch]]
        for slot, value in enumerate(temperature):
            if value == 0.0:
                top_k[slot] = 1
                top_p[slot] = 0.0
        return {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "active_batch": active_batch,
        }

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        sampling_params=None,
        generation_horizon=None,
        **kwargs,
    ):
        if kv_cache is None:
            raise ValueError("vLLM must own and pass the Falcon3 attention KV cache")
        if generation_horizon is not None:
            generation_horizon = int(generation_horizon)
            if generation_horizon < max(prompt_lens):
                raise ValueError("generation_horizon cannot be shorter than the prompt")
            if generation_horizon > self.model.max_cache_len:
                raise ValueError("generation_horizon exceeds the supported context")
            # Falcon3 starts with a deliberately small RoPE table. Grow it to
            # the request's declared lifetime before the first decode trace is
            # captured; replacing the shared table while a trace references it
            # would leave replay using stale positional state.
            self._release_decode_traces_before_allocating_prefill()
            self.model.ensure_rope_capacity(generation_horizon)
        active_batch = len(prompt_lens)
        if sampling_params is None:
            return super().prefill_forward(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=list(prompt_lens),
                sampling_mode="host",
            )
        self.set_sampling_params(**self._sampling_values(sampling_params, active_batch))
        self._vllm_active_batch = active_batch
        sampled = super().prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=list(prompt_lens),
            sampling_mode="device",
        )
        # vLLM's prefill completion boundary is synchronous and indexes a
        # torch token vector. Read only the sampled token ids; the unchanged
        # device tensor remains the canonical first-decode feedback source.
        return self._sampled_to_torch(sampled)[:active_batch]

    def decode_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        start_pos,
        sampling_params=None,
        reset_batch=True,
        slot_remap=None,
        enable_trace=True,
        read_from_device=False,
        **kwargs,
    ):
        if kv_cache is None:
            raise ValueError("vLLM must own and pass the Falcon3 attention KV cache")
        active_batch = int(tokens.shape[0])
        if sampling_params is None:
            self._vllm_active_batch = active_batch
            logits = super().decode_forward(
                tokens,
                start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_mode="host",
                enable_trace=False,
                active_batch=active_batch,
            )
            return logits.unsqueeze(1)
        if not enable_trace:
            raise ValueError("Falcon3 vLLM decode requires the canonical traced split-sampling path")
        if read_from_device:
            raise ValueError("Falcon3 vLLM decode uses the deferred async read boundary")
        self.set_sampling_params(**self._sampling_values(sampling_params, active_batch))

        # In steady async scheduling, vLLM deliberately supplies stale token
        # and position values. The model trace advances both persistent position
        # tensors exactly once and the sampling trace writes the emitted token
        # into the persistent token input, so those host values must not be read.
        steady = reset_batch is False and self._trace_model_id is not None
        if steady:
            if slot_remap is not None:
                identity = torch.arange(active_batch, dtype=slot_remap.dtype)
                if not torch.equal(slot_remap.reshape(-1)[:active_batch].cpu(), identity):
                    raise ValueError("Falcon3 steady async decode cannot remap live slots")
            tokens = None
            start_pos = None
            # The TT async-scheduling contract guarantees that the page table
            # is unchanged while this steady-overlap fast path is active.  Do
            # not normalize and compare its full fixed-width host tensor every
            # token; the persistent device page-table input captured at reset
            # remains authoritative.  Scheduler changes leave steady mode and
            # refresh (or recapture) the trace through the reset path below.
            page_table = None
        self._vllm_active_batch = active_batch
        return super().decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_mode="device",
            enable_trace=True,
            active_batch=active_batch,
        )

    def read_decode_output(self, tt_out, async_read=False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        if not async_read:
            return tt_out.cpu()
        # Sampling writes the same token vector to every rank.  Read one
        # replica, matching the canonical full-model token-out boundary;
        # copying the distributed tensor reads all four identical replicas.
        shards = ttnn.get_device_tensors(tt_out)
        host = (shards[0] if shards else tt_out).cpu(blocking=False)
        return host, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, tt_out, is_tokens=True):
        if not is_tokens:
            raise ValueError("Falcon3 vLLM serving never reads back full logits")
        shards = ttnn.get_device_tensors(tt_out)
        host = ttnn.to_torch(shards[0] if shards else tt_out)
        return host.reshape(-1)[: self._vllm_active_batch].to(torch.int32).view(-1, 1)

    def warmup_model_prefill(self, **kwargs):
        # Falcon3 prefill is intentionally eager and compiles logical lengths on
        # demand; the readiness requests provide coverage before measurement.
        return None

    def warmup_model_decode(self, **kwargs):
        # The first real decode initializes the split traces with its exact
        # vLLM-owned page table and request positions.
        return None


__all__ = ["Falcon3ForCausalLM"]

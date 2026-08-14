# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Thin vLLM bridge for the Qwen/Qwen3.6-27B autoport.

vLLM owns paged attention K/V.  Qwen's constant-size linear-attention state
remains model-owned.  Decode delegates to the full-model generator's canonical
split model/sampling traces; the adapter never reconstructs feedback tokens or
reads logits in the performance path.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_REVISION
from models.autoports.qwen_qwen3_6_27b.tt.generator import Qwen36Generator
from models.autoports.qwen_qwen3_6_27b.tt.model import Qwen36Model, _shard
from models.autoports.qwen_qwen3_6_27b.tt.precision_config import DEFAULT_PRECISION_CONFIG
from models.common.sampling import format_sampling_params


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = Path("/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots") / MODEL_REVISION
# The datatype-sweep capacity probe proves 2,496,512 tokens with a 4 MB trace
# region. Serving reserves 200 MB per DRAM bank; at the measured tiled-BFP8
# cost (8,704 bytes/token/device), runtime prefill and model-owned linear-state
# trace-backup evidence leaves a 2,084,000-token pool. vLLM adds one lookahead
# page per sequence, so
# report 32 pages less here.
# vLLM adds 32 lookahead pages (25,600 tokens) to this value.  The resulting
# 1,752,000-token pool leaves one 327 MiB contiguous linear-scan workspace plus
# fragmentation margin for a full-capacity serving prefill.
MAX_TOKENS_ALL_USERS = 1_726_400


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen36ForCausalLM(nn.Module, SupportsMultiModal):
    """vLLM interface translation over :class:`Qwen36Generator`."""

    decode_input_update_contract = 1
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_async_decode_overlap": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
    }

    def __init__(self, generator: Qwen36Generator | None = None, **_: Any):
        super().__init__()
        self.generator = generator
        if generator is not None:
            # vLLM owns attention K/V; never create a shadow serving cache for
            # trace capture. Only model-owned linear state is restored around
            # the stateful warm/capture executions.
            generator.trace_backup_attention_cache = False
        self.model = None if generator is None else generator.model
        self.mesh_device = None if generator is None else generator.mesh_device
        self.max_batch_size = 0 if generator is None else generator.model.batch
        self.max_seq_len = 0 if generator is None else generator.model.max_context
        self._decode_ready = False
        self._last_page_table: torch.Tensor | None = None

    @classmethod
    def get_max_tokens_all_users(
        cls, *, max_model_len: int | None = None, max_num_seqs: int | None = None, **_: Any
    ) -> int:
        del max_model_len, max_num_seqs
        # Physically derived from the datatype-sweep capacity bracket after
        # accounting for serving's larger trace reservation. This still serves
        # a 262,144-token request while constraining aggregate concurrency.
        return MAX_TOKENS_ALL_USERS

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int,
        n_layers: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **_: Any,
    ) -> "Qwen36ForCausalLM":
        del optimizations
        if tt_data_parallel != 1:
            raise NotImplementedError("Qwen3.6 autoport supports one TP4 model replica")
        if not 1 <= int(max_batch_size) <= 32:
            raise ValueError("Qwen3.6 serving max_num_seqs must be in [1, 32]")
        snapshot = Path(getattr(hf_config, "_name_or_path", "") or DEFAULT_SNAPSHOT)
        if not snapshot.is_dir():
            snapshot = DEFAULT_SNAPSHOT
        model = Qwen36Model.from_pretrained(
            mesh_device=mesh_device,
            snapshot=snapshot,
            batch=int(max_batch_size),
            max_context=int(max_seq_len),
            page_size=64,
            # vLLM allocates the real attention pool after model load.  Keep a
            # single block per layer during construction to avoid a hidden
            # standalone-cache allocation, then bind the returned vLLM cache.
            attention_cache_blocks=1,
            num_layers=n_layers,
            precision_config=DEFAULT_PRECISION_CONFIG,
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        return cls(Qwen36Generator(model=model, mesh_device=mesh_device, tokenizer=tokenizer))

    def _require_generator(self) -> Qwen36Generator:
        if self.generator is None:
            raise RuntimeError("Qwen3.6 vLLM model has not been initialized")
        return self.generator

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Return vLLM-owned paged K/V plus model-owned recurrent state."""
        del dtype
        gen = self._require_generator()
        serving_page_size = int(kv_cache_shape[2])
        if serving_page_size < 32 or serving_page_size % 32:
            raise ValueError(f"Qwen3.6 serving page size must be a tile multiple, got {serving_page_size}")
        attention_layers = sum(layer.layer_kind == "full_attention" for layer in gen.model.layers)
        if int(num_layers) != attention_layers:
            raise ValueError(f"vLLM requested {num_layers} attention layers; TT built {attention_layers}")
        if serving_page_size != gen.model.page_size:
            # vLLM hybrid-cache grouping may enlarge attention pages to match
            # the recurrent-state page. Rebind the request-boundary page table
            # to that public scheduler contract; logical positions are unchanged.
            ttnn.deallocate(gen._page_table)
            gen.model.page_size = serving_page_size
            blocks_per_request = math.ceil(gen.model.max_context / serving_page_size)
            gen.page_table_host = torch.zeros((gen.model.batch, blocks_per_request), dtype=torch.int32)
            gen._page_table = gen._upload(gen.page_table_host, dtype=ttnn.int32)
        result = []
        for layer in gen.model.layers:
            if layer.layer_kind == "linear_attention":
                result.append([layer.caches["conv"], layer.caches["recurrent"]])
                continue
            shape = tuple(int(x) for x in kv_cache_shape)
            # vLLM supplies local KV heads; the model's TP cache is sharded on
            # the global head axis, so materialize the equivalent global shape.
            global_shape = (shape[0], shape[1] * 4, shape[2], shape[3])
            result.append(
                [
                    _shard(torch.zeros(global_shape, dtype=torch.bfloat16), self.mesh_device, 1, dtype=layer.policy.cache_dtype),
                    _shard(torch.zeros(global_shape, dtype=torch.bfloat16), self.mesh_device, 1, dtype=layer.policy.cache_dtype),
                ]
            )
        gen.model.bind_kv_cache(result)
        gen.kv_cache = result
        return result

    def _full_slot_inputs(self, tokens, prompt_lens, empty_slots, page_table):
        gen = self._require_generator()
        lengths = [int(v) for v in prompt_lens]
        slots = list(range(len(lengths))) if empty_slots is None else [int(v) for v in empty_slots]
        physical = int(tokens.shape[1])
        full_tokens = torch.zeros((gen.model.batch, physical), dtype=tokens.dtype)
        full_lengths = [0] * gen.model.batch
        full_page_table = torch.zeros((gen.model.batch, page_table.shape[1]), dtype=page_table.dtype)
        for row, slot in enumerate(slots):
            full_tokens[slot] = tokens[row]
            full_lengths[slot] = lengths[row]
            full_page_table[slot] = page_table[row]
        return full_tokens, full_lengths, full_page_table, slots

    def _set_sampling(self, sampling_params):
        gen = self._require_generator()
        if sampling_params is not None:
            gen.sampling.reset_sampling_params(format_sampling_params(sampling_params, 32))

    @staticmethod
    def _format_slot_sampling(sampling_params, slots):
        """Scatter compact request-order parameters into persistent slots."""
        formatted = format_sampling_params(sampling_params, 32)
        updates = {}
        for field in fields(formatted):
            raw = getattr(sampling_params, field.name)
            values = getattr(formatted, field.name)
            # SeedManager.reset_seed consumes compact request-order seeds with
            # ``empty_slots``; TT sampler/penalty tensors consume physical-slot
            # vectors.  Scattering seed here assigns defaults to sparse slots.
            if field.name != "seed" and isinstance(raw, list) and len(raw) == len(slots) and isinstance(values, list):
                scattered = [values[-1]] * 32
                for row, slot in enumerate(slots):
                    scattered[slot] = values[row]
                updates[field.name] = scattered
        return replace(formatted, **updates)

    def _sampler_ready_prefill_logits(self, logits):
        """Translate terminal prefill rows to the canonical 32-slot sampler shape."""
        logits = ttnn.reshape(logits, (1, 1, self.max_batch_size, logits.shape[-1]))
        if self.max_batch_size < 32:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - self.max_batch_size), (0, 0)], value=0.0)
        return logits

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        sampling_params=None,
        empty_slots=None,
        start_pos=None,
        **_: Any,
    ):
        gen = self._require_generator()
        full_tokens, full_lengths, full_page_table, slots = self._full_slot_inputs(
            tokens, prompt_lens, empty_slots, page_table
        )
        # Full-attention K/V is owned and invalidated by vLLM.  Linear-attention
        # conv/recurrent state is model-owned, so a newly assigned physical slot
        # must be cleared before its replacement request is prefetched.
        gen.reset_slots(slots)
        if sampling_params is not None:
            formatted_params = self._format_slot_sampling(sampling_params, slots)
            prompt_state = torch.zeros_like(full_tokens)
            for slot, length in zip(slots, prompt_lens):
                length = int(length)
                prompt_state[slot, :length] = full_tokens[slot, :length]
            gen.sampling.apply_prefill_state(
                sampling_params=formatted_params,
                prompt_tokens=prompt_state,
                empty_slots=slots,
                replicate_seeds=False,
            )
        gen.refresh_page_table(full_page_table)
        logits = gen.prefill_forward(
            full_tokens,
            page_table=gen._page_table,
            kv_cache=kv_cache,
            prompt_lens=full_lengths,
            read_from_device=sampling_params is None,
        )
        self._last_page_table = full_page_table.clone()
        self._decode_ready = False
        if sampling_params is None:
            output = logits[slots]
        else:
            sampler_logits = self._sampler_ready_prefill_logits(logits)
            sampled, log_probs = gen.sampling.sample(sampler_logits, enable_trace=False)
            host = ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1).to(torch.int32)
            if sampler_logits is not logits:
                ttnn.deallocate(sampler_logits)
            ttnn.deallocate(logits)
            output = (host[slots], log_probs)

        # Transformers classifies Qwen3.6 under the Qwen3.5 MRoPE family, so
        # the shared runner persists a request-specific delta. Text-only input
        # has no vision-grid offset: logical token positions already used by
        # the TT generator are exact, hence the per-request delta is zero.
        return output, torch.zeros(len(prompt_lens), dtype=torch.int64)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
        rope_deltas_all_users=None,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        **_: Any,
    ):
        if not enable_trace:
            raise ValueError("Qwen3.6 vLLM decode requires tracing")
        if rope_deltas_all_users is not None and torch.as_tensor(rope_deltas_all_users).ne(0).any():
            raise NotImplementedError("Qwen3.6 text serving does not accept nonzero multimodal RoPE deltas")
        gen = self._require_generator()
        if sampling_params is None:
            # Explicit compatibility boundary for shared tests whose sampling
            # features (for example min-p) are not supported by TT sampling.
            # Performance/readiness profiles always supply sampling_params and
            # use the canonical split token-out traces below.
            if isinstance(page_table, torch.Tensor):
                gen.refresh_page_table(page_table)
                self._last_page_table = page_table.clone()
            self._decode_ready = False
            host_logits = gen.decode_forward(
                tokens.reshape(-1)[: gen.model.batch],
                start_pos.reshape(-1)[: gen.model.batch],
                page_table=gen._page_table,
                kv_cache=kv_cache,
                active_mask=start_pos.reshape(-1)[: gen.model.batch] >= 0,
            )
            return host_logits.unsqueeze(1)
        gen.sampling.apply_decode_state(
            [sampling_params],
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        start_values = start_pos.reshape(-1)[: gen.model.batch].tolist()
        active_seed_slots = [slot for slot, pos in enumerate(start_values) if int(pos) >= 0]
        formatted_params = format_sampling_params(sampling_params, 32)
        seed_values = formatted_params.seed
        remap_changed = slot_remap is not None
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap).reshape(-1)[: gen.model.batch].tolist()
            gen.remap_decode_slots(remap)
            gen.sampling.seed_manager.apply_slot_remap(remap)
        if active_seed_slots:
            gen.sampling.seed_manager.reset_seed_from_slots_if_needed(seed_values, active_seed_slots)
            gen.sampling.seed_manager.align_seed_counters_to_positions(
                seed_values, active_seed_slots, start_values
            )
        page_changed = self._last_page_table is None or not torch.equal(page_table, self._last_page_table)
        if reset_batch or not self._decode_ready or page_changed or remap_changed:
            gen.setup_token_out_decode(
                tokens.reshape(-1)[: gen.model.batch],
                start_pos.reshape(-1)[: gen.model.batch],
                page_table=page_table,
                kv_cache=kv_cache,
                active_mask=start_pos.reshape(-1)[: gen.model.batch] >= 0,
                sampling_params=sampling_params,
            )
            self._decode_ready = True
            self._last_page_table = page_table.clone()
        # Trace setup performs sampler warm/capture executions.  Write the
        # request's intended per-token device seeds afterwards so capture-side
        # RNG consumption cannot perturb the first real replay.
        gen.sampling.seed_manager.get_new_values(active_seed_slots)
        tt_out = gen.token_out_decode_step(readback=False)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(tt_out), is_tokens=True)
        return tt_out

    def read_decode_output(self, tt_out, async_read=False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        if async_read:
            host = tt_out.cpu(blocking=False)
            return host, [ttnn.record_event(self.mesh_device, 0)]
        return tt_out.cpu()

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if not is_tokens:
            if not isinstance(tt_out, torch.Tensor):
                tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
            return tt_out, None
        if isinstance(tt_out, torch.Tensor):
            return tt_out.reshape(-1)[: self.max_batch_size].to(torch.int32), None
        host = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        return host.reshape(-1)[: self.max_batch_size].to(torch.int32), None

    def warmup_model_prefill(self, *args, **kwargs):
        del args, kwargs

    def warmup_model_decode(self, *args, **kwargs):
        # Trace capture is stateful and is performed from the first real
        # scheduler batch so its page table and active slots are exact.
        del args, kwargs

    def teardown(self):
        if self.generator is not None:
            self.generator.teardown()


__all__ = ["Qwen36ForCausalLM"]

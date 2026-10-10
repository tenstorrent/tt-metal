# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation for the canonical K2 TP4 generator."""

from __future__ import annotations

import json
import logging
import os

import torch

import ttnn

from .benchmark_timing import PhaseRecorder
from .generator import K2Generator

logger = logging.getLogger(__name__)


class K2HorizonForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_chunked_prefill": True,
        "max_device_top_k": 32,
        "supports_device_penalties": False,
    }

    def compute_logits(self, hidden_states, sampling_metadata=None):
        raise NotImplementedError("The TT plugin calls prefill_forward/decode_forward directly")

    def embed_input_ids(self, input_ids):
        raise NotImplementedError("The TT generator owns embedding")

    def forward(self, input_ids, positions):
        raise NotImplementedError("The TT plugin calls prefill_forward/decode_forward directly")

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        return 524288

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= 32:
            raise ValueError("K2 supports TP4, DP1, and 1..32 sequence slots")
        if max_seq_len > 524288:
            raise ValueError("Context exceeds the K2 checkpoint contract")
        return cls(
            mesh_device,
            max_batch_size=max_batch_size,
            num_layers=n_layers or hf_config.num_hidden_layers,
        )

    def __init__(self, mesh_device, *, max_batch_size=32, num_layers=36, vllm_config=None):
        self.generator = K2Generator(mesh_device, override_num_layers=num_layers)
        self.mesh = mesh_device
        self.max_batch_size = max_batch_size
        # Requests the device sampler cannot serve (top_k outside 1..32, penalties, logit
        # controls) are routed to host sampling by the plugin; the model's own host sampler
        # keeps seeded results identical to the device path. K2_VLLM_FORCE_HOST_SAMPLING=1
        # sends every request there (shared-test mode).
        self.force_host_sampling = os.environ.get("K2_VLLM_FORCE_HOST_SAMPLING", "0") == "1"
        from .host_sampling import K2HostSampler

        self.host_sampler = K2HostSampler(max_workers=8)
        self.mode = None
        self.host_fallback_calls = 0
        self.decode_table_width = None
        self.decode_batch = max_batch_size
        self.audit_path = os.environ.get("K2_VLLM_AUDIT_PATH")
        logger.info("K2 precision policy: %s", self.generator.model.precision_config)
        logger.info("K2 forced host sampling (shared-test mode): %s", self.force_host_sampling)
        self.benchmark_phase = PhaseRecorder.from_adapter(self)

    def _audit(self, event, **fields):
        if self.audit_path:
            view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
            memory = {
                key: getattr(view, key)
                for key in (
                    "num_banks",
                    "total_bytes_per_bank",
                    "total_bytes_allocated_per_bank",
                    "largest_contiguous_bytes_free_per_bank",
                )
            }
            with open(self.audit_path, "a") as out:
                out.write(
                    json.dumps(dict(event=event, counters=dict(self.generator.counters), memory=memory, **fields))
                    + "\n"
                )

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Return vLLM-owned attention pages; never establish standalone cache."""
        pages, heads, block, width = kv_cache_shape
        model = self.generator.model
        if heads != 2 or (block, width) != (32, 128) or num_layers != model.num_layers:
            raise ValueError(f"Unexpected vLLM cache contract: {kv_cache_shape}, layers={num_layers}")
        local_shape = (pages, heads, block, width)
        return [
            [
                ttnn.zeros(
                    local_shape,
                    dtype=layer.kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
            for layer in model.layers
        ]

    @staticmethod
    def _table(table, logical_end=None):
        # The decoder reads K128 windows. Only the internal table is padded;
        # zero is a valid physical page and masked entries are never written.
        if logical_end is not None:
            # Capacity buckets select the decoder's existing short/long-context
            # attention contracts. The external cache still covers max context.
            capacity = max(4096, 1 << (max(1, logical_end) - 1).bit_length())
            table = table[:, : capacity // 32]
        table = table.int()
        padding = (-table.shape[1]) % 4
        return torch.nn.functional.pad(table, (0, padding)) if padding else table

    def _sampling(self, params, positions):
        if params is None:
            self.host_fallback_calls += 1
            if self.host_fallback_calls == 1:
                logger.info("K2 host sampling path first used (plugin returned logits request)")
            return "host"
        self.generator.configure_sampling_batch(
            top_k=params.top_k,
            top_p=params.top_p,
            temperature=params.temperature,
            seeds=params.seed,
            positions=positions,
        )
        return "device"

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        enable_trace=True,
        empty_slots=None,
    ):
        if self.benchmark_phase is not None:
            self.benchmark_phase.annotate(
                generated_entry_ns=self.benchmark_phase.entry(), generator_counters_before=dict(self.generator.counters)
            )
        starts = [0] * len(prompt_lens) if start_pos is None else list(map(int, start_pos))
        ends = list(map(int, prompt_lens))
        lengths = [end - start for start, end in zip(starts, ends)]
        chunks = torch.zeros((len(lengths), max(lengths)), dtype=tokens.dtype)
        for row, (start, end) in enumerate(zip(starts, ends)):
            chunks[row, : end - start] = tokens[row, start:end]
        mode = self._sampling(sampling_params, [end - 1 for end in ends])
        self.mode = "prefill"
        prepared_table = self._table(page_table, max(ends))
        if self.benchmark_phase is not None:
            self.benchmark_phase.annotate(
                generated_batch=len(lengths),
                chunk_lengths=lengths,
                table_capacity=prepared_table.shape[1] * 32,
                sampling_mode=mode,
                sampling_strategy="split",
                attention_branch="prefill",
            )
        out = self.generator.prefill_forward(
            chunks,
            page_table=prepared_table,
            kv_cache=kv_cache,
            prompt_lens=lengths,
            start_pos=starts,
            sampling_mode=mode,
            trace_prefill=enable_trace,
        )
        self._audit("prefill", starts=starts, lengths=lengths, sampling_mode=mode)
        if self.benchmark_phase is not None:
            self.benchmark_phase.annotate(generator_counters_after=dict(self.generator.counters))
            self.benchmark_phase.submitted()
        return out.reshape(len(lengths), 1) if mode == "device" else out

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
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
    ):
        if self.benchmark_phase is not None:
            self.benchmark_phase.annotate(
                generated_entry_ns=self.benchmark_phase.entry(), generator_counters_before=dict(self.generator.counters)
            )
        if not enable_trace:
            raise ValueError("K2 serving decode requires trace replay")
        mode = "device" if sampling_params is not None else "host"
        state = self.generator.state
        reload_inputs = reset_batch or self.mode != mode or state is None or state["trace"] is None or mode == "host"
        if reload_inputs:
            mode = self._sampling(sampling_params, list(map(int, start_pos)))
            self.decode_table_width = self._table(page_table, int(start_pos.max()) + 1).shape[1]
            active = torch.nonzero(start_pos >= 0).reshape(-1)
            count = int(active[-1]) + 1 if len(active) else 1
            # vLLM pads the wire batch to max_num_seqs. Select the generator's
            # existing power-of-two batch at scheduler resets; admission still
            # supports every slot and output retains the full wire shape.
            self.decode_batch = min(self.max_batch_size, 1 << (count - 1).bit_length())
        if self.benchmark_phase is not None:
            preparation_key = (
                self.decode_batch,
                (self.decode_batch, self.decode_table_width),
                tuple(id(t) for pair in kv_cache for t in pair),
            )
            trace_preparation = state is None or state["trace"] is None or state["key"] != preparation_key
            self.benchmark_phase.annotate(
                trace_preparation=trace_preparation,
                executed_model_passes=2 if trace_preparation else 1,
                sampling_strategy="split",
                warm_sampling_strategies=["split", "argmax"] if trace_preparation else [],
            )
            self.benchmark_phase.decode(
                positions=start_pos.tolist(),
                reload_inputs=reload_inputs,
                generated_batch=self.decode_batch,
                table_capacity=self.decode_table_width * 32,
            )
        self.mode = mode
        out = self.generator.decode_forward(
            tokens[: self.decode_batch] if reload_inputs else None,
            start_pos[: self.decode_batch] if reload_inputs else None,
            page_table=self._table(page_table[: self.decode_batch, : self.decode_table_width]),
            kv_cache=kv_cache,
            sampling_mode=mode,
            read_from_device=read_from_device,
        )
        if self.benchmark_phase is not None:
            self.benchmark_phase.annotate(generator_counters_after=dict(self.generator.counters))
            self.benchmark_phase.submitted()
        if reload_inputs:
            self._audit(
                "decode_refresh",
                positions=start_pos.tolist(),
                reset_batch=bool(reset_batch),
                table_width=self.decode_table_width,
                decode_batch=self.decode_batch,
                page_heads=page_table[:, :16].tolist(),
                sampling_mode=mode,
            )
        if read_from_device:
            if mode == "device":
                return torch.nn.functional.pad(
                    out.reshape(self.decode_batch, 1), (0, 0, 0, self.max_batch_size - self.decode_batch)
                )
            return torch.nn.functional.pad(out, (0, 0, 0, self.max_batch_size - self.decode_batch)).unsqueeze(1)
        return out

    def read_decode_output(self, tt_out, async_read=False):
        if self.mode == "device":
            host = self.generator.read_decode_output(async_read=async_read)
        else:
            host = tt_out.cpu(blocking=not async_read)
        return (host, [ttnn.record_event(self.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if is_tokens:
            return self.generator.process_decode_output_host(
                ttnn.get_device_tensors(tt_out)[0], batch_size=self.max_batch_size
            ).reshape(self.max_batch_size, 1)
        return self.generator._read_logits(tt_out).reshape(32, 1, -1)[: self.max_batch_size]

    def warmup_model_prefill(self, **kwargs):
        # Arbitrary request shapes are warmed on first use by the generator,
        # which retires live traces before admitting new program variants.
        pass

    def warmup_model_decode(self, **kwargs):
        # Capture is bound to the exact vLLM-owned cache and table on first use.
        pass

    def close(self):
        try:
            self.generator.close()
        finally:
            if self.host_sampler is not None:
                self.host_sampler.close()
            if self.benchmark_phase is not None:
                self.benchmark_phase.close()

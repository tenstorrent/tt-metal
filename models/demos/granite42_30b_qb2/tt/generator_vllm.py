# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone TT-plugin bridge to Granite's canonical full-model generator."""

import os
import secrets

import torch

import ttnn

from .generator import GraniteGenerator
from .model import GraniteModel


class GraniteForCausalLM:
    decode_input_update_contract = 1
    model_capabilities = {
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D_RING},
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "supports_chunked_prefill": True,
        "supports_device_chunked_prefill": True,
        "max_device_top_k": 32,
    }

    def __init__(self, *, vllm_config=None):
        if vllm_config is not None:
            raise RuntimeError("Initialize the TT model through initialize_vllm_model")

    # vLLM checks its generation protocol before the TT plugin selects the
    # prefill/decode execution path. The GPU entry points are not used by TT.
    def embed_input_ids(self, input_ids):
        raise RuntimeError("Use the TT prefill_forward/decode_forward interface")

    def forward(self, input_ids, positions):
        raise RuntimeError("Use the TT prefill_forward/decode_forward interface")

    def compute_logits(self, hidden_states):
        raise RuntimeError("The TT generator owns the terminal norm and LM head")

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        return 131072

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, **kwargs):
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= 16:
            raise ValueError("Granite TP4 supports one mesh with 1..16 active requests")
        self = cls()
        self.mesh = mesh_device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.model = GraniteModel(mesh_device)
        self.generator = None
        return self

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        pages, heads, block, dim = kv_cache_shape
        if (heads, block, dim) != (2, 32, 128):
            raise ValueError(f"Unsupported vLLM KV shape {kv_cache_shape}")
        # The allocator callback owns this pool; no standalone pool is created.
        cache = self.model.allocate_cache(pages)
        self.generator = GraniteGenerator(
            self.mesh,
            model=self.model,
            kv_cache=cache,
            max_seq_len=self.max_seq_len,
            batch_buckets=(1, 8, 16),
            trace_prefill=os.environ.get("GRANITE_VLLM_TRACE_PREFILL", "1") != "0",
        )
        return [cache]

    def warmup_model_prefill(self, kv_cache, **kwargs):
        self.generator._validate_cache(kv_cache[0])
        self.generator.prepare()
        self.tt_supported_decode_batch_sizes = (1, 8, 16)

    warmup_model_decode = warmup_model_prefill

    def release_persistent_capture(self):
        if self.generator is not None:
            self.generator.close()

    def _table(self, page_table, batch):
        g = self.generator
        bucket = g._bucket(batch)
        table = torch.zeros(bucket, g.pages_per_row, dtype=torch.int32)
        rows = min(batch, page_table.shape[0])
        cols = min(g.pages_per_row, page_table.shape[1])
        table[:rows, :cols] = page_table[:rows, :cols].int()
        return table

    def _sampling(self, params, *, positions=None, reload=True, reset=True):
        if params is None:
            return "host"

        def padded(values, default):
            values = list(values) if isinstance(values, (list, tuple)) else [values]
            return [default if v is None else v for v in values] + [default] * (32 - len(values))

        temp = padded(params.temperature, 1.0)
        k = padded(params.top_k, 1)
        p = padded(params.top_p, 0.0)
        for i, t in enumerate(temp):
            if t == 0:
                temp[i], k[i], p[i] = 1.0, 1, 0.0
        # Generator owns the shared sampling implementation and trace bindings.
        if reload:
            self.generator.set_sampling_params(top_k=k, top_p=p, temperature=temp, reset_seed=False)
        if reset:
            raw_seeds = list(params.seed) if isinstance(params.seed, (list, tuple)) else [params.seed]
            seeds = [secrets.randbelow(1000000) if seed is None else seed for seed in raw_seeds]
            seeds += [0] * (32 - len(seeds))
            self.generator.reset_sampling_state(seeds, positions)
        return "device"

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        intermediate_prefill_mask=None,
        **kwargs,
    ):
        prompt_lens = [int(x) for x in prompt_lens]
        starts = torch.as_tensor(start_pos).reshape(-1).tolist() if start_pos is not None else None
        starts = starts or [0] * len(prompt_lens)
        lengths = [end - start for start, end in zip(starts, prompt_lens)]
        if any(length < 0 for length in lengths):
            raise ValueError("vLLM prefill end must be at or after its start")
        local_tokens = torch.zeros(len(lengths), max(lengths), dtype=tokens.dtype)
        for row, (start, length) in enumerate(zip(starts, lengths)):
            local_tokens[row, :length] = tokens[row, start : start + length]
        positions = [end - 1 for end in prompt_lens]
        mode = self._sampling(sampling_params, positions=positions)
        out = self.generator.prefill_forward(
            local_tokens,
            page_table=self._table(page_table, len(prompt_lens)),
            kv_cache=kv_cache[0],
            prompt_lens=lengths,
            start_pos=starts,
            sampling_mode=mode,
            sample_mask=None if intermediate_prefill_mask is None else [not bool(x) for x in intermediate_prefill_mask],
        )
        return out.reshape(-1, 1) if mode == "device" else out

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        sampling_params=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        read_from_device=True,
        enable_trace=True,
        **kwargs,
    ):
        if not enable_trace:
            # Preparation still belongs to the generator; runtime always replays.
            self.generator.prepare()
        mode = "device" if sampling_params is not None else "host"
        positions = torch.as_tensor(start_pos).reshape(-1)
        if reload_sampling_params or reset_sampling_state or mode == "host":
            self._sampling(
                sampling_params, positions=positions, reload=reload_sampling_params, reset=reset_sampling_state
            )
        table = self._table(page_table, len(positions)) if reload_inputs or reload_page_table else None
        out = self.generator.decode_forward(
            tokens,
            positions,
            page_table=table,
            kv_cache=kv_cache[0],
            sampling_mode=mode,
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            read_from_device=False,
        )
        if mode == "host":
            return out[: len(positions)].reshape(len(positions), 1, -1)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(out), is_tokens=True)
        return out

    def read_decode_output(self, tt_out, async_read=False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        # Tokens are replicated. Transfer only one rank and defer host conversion.
        host = ttnn.get_device_tensors(tt_out)[0].cpu(blocking=not async_read)
        if async_read:
            return host, [ttnn.record_event(self.mesh, 0)]
        return host

    def process_decode_output_host(self, tt_out, is_tokens=True):
        return ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(-1, 1).long()

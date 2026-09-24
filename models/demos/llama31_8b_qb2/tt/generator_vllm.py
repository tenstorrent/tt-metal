# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM scheduling and sampling inputs for the TP4 Llama generator."""

from dataclasses import replace

import torch

import ttnn
from models.demos.llama31_8b_qb2.tt.generator import LlamaGenerator


class LlamaForCausalLM:
    _fabric_router_config = ttnn.FabricRouterConfig()
    _fabric_router_config.max_packet_payload_size_bytes = 8192
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "fabric_config": {
            "config": ttnn.FabricConfig.FABRIC_1D_RING,
            "router_config": _fabric_router_config,
        },
    }
    decode_input_update_contract = 1

    def __init__(self, generator):
        self.generator = generator
        self.mesh_device = generator.mesh_device
        self.max_batch_size = generator.max_batch_size

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=131072, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or max_seq_len != 131072:
            raise ValueError("Llama QB2 requires DP1 and a 131072-token context")
        if hf_config.vocab_size != 128256 or hf_config.hidden_size != 4096:
            raise ValueError("Expected the Llama-3.1-8B-Instruct checkpoint")
        torch.set_num_threads(8)
        return cls(
            LlamaGenerator(
                mesh_device,
                max_batch_size=max_batch_size,
                owns_kv_cache=False,
                record_token_history=False,
                # Single-user prefill retains its low-latency trace. Scheduler
                # admission groups rarely repeat, so compile those eagerly.
                trace_prefill_max_batch_size=1,
            )
        )

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        return 131072

    def allocate_vllm_kv_cache(self, kv_cache_shape, dtype, num_layers):
        pages, heads, block, dim = kv_cache_shape
        if (heads, block, dim) != (2, 128, 128) or num_layers != self.generator.model.num_layers:
            raise ValueError(f"Unexpected cache spec: {kv_cache_shape}, layers={num_layers}")
        cache = self.generator.model.allocate_cache(pages)
        self.generator._validate_cache(cache)
        return cache

    allocate_kv_cache = allocate_vllm_kv_cache

    def _sampling(self, params, positions, *, reset_state=True):
        if params is None:
            return
        seed = None
        if reset_state:
            seeds = params.seed if isinstance(params.seed, list) else [params.seed] * self.max_batch_size
            seed = torch.tensor([s if s is not None else int(torch.randint(0, 2**30, ())) for s in seeds])
            # Absolute positions preserve each request's random stream through
            # scheduler compaction; steady decode advances the seed on device.
            seed += torch.as_tensor(positions).flatten().clamp_min(0)
        self.generator.set_sampling(top_k=params.top_k, top_p=params.top_p, temperature=params.temperature, seed=seed)

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        *,
        sampling_params=None,
        start_pos=None,
        enable_trace=True,
        empty_slots=None,
        **kwargs,
    ):
        if start_pos is not None and torch.as_tensor(start_pos).ne(0).any():
            raise ValueError("Prefix caching and scheduler chunked prefill are unsupported")
        lengths = torch.as_tensor(prompt_lens).tolist()
        batch = len(lengths)
        slots = list(range(batch)) if empty_slots is None else list(empty_slots)
        table = torch.zeros(self.max_batch_size, 1024, dtype=torch.int32)
        table[slots, : page_table.shape[1]] = page_table
        positions = torch.full((self.max_batch_size,), -1, dtype=torch.int32)
        positions[slots] = torch.tensor(lengths, dtype=torch.int32).clamp_max(131071)
        if sampling_params is not None:
            updates = {}
            for name in ("top_k", "top_p", "temperature", "seed"):
                values = getattr(sampling_params, name)
                values = values if isinstance(values, list) else [values] * batch
                padded = [values[0]] * self.max_batch_size
                for slot, value in zip(slots, values):
                    padded[slot] = value
                updates[name] = padded
            sampling_params = replace(sampling_params, **updates)
        # Prefill predicts at length-1; decode consumes that token at length.
        self._sampling(sampling_params, positions - 1)
        self.generator.refresh_decode_inputs(
            torch.zeros(self.max_batch_size, dtype=torch.int32), positions, page_table=table
        )
        return self.generator.prefill_forward(
            tokens,
            page_table=table,
            kv_cache=kv_cache,
            prompt_lens=lengths,
            slots=slots,
            sample_on_device=sampling_params is not None,
        )

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        *,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        **kwargs,
    ):
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1 requires reload_inputs, not reset_batch")
        if not enable_trace:
            raise ValueError("Serving requires traced decode")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Resetting sampling state requires current tokens and positions")
        sample = sampling_params is not None
        if reload_sampling_params or reset_sampling_state:
            self._sampling(sampling_params, start_pos, reset_state=reset_sampling_state)
        output = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sample_on_device=sample,
            reset_batch=reload_inputs,
            reload_page_table=reload_page_table,
            read_from_device=False,
        )
        return self.process_decode_output_host(output, is_tokens=sample) if read_from_device else output

    def read_decode_output(self, output, async_read=False):
        return self.generator.read_decode_output(output, async_read=async_read)

    def process_decode_output_host(self, output, is_tokens=True):
        host = self.generator.process_decode_output_host(output, is_tokens=is_tokens)
        return host.reshape(-1, 1) if is_tokens else host.reshape(self.max_batch_size, 1, -1)

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        # Unseen logical lengths release live traces before compilation.
        pass

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        self.generator.prepare_traces(kv_cache=kv_cache)

    def release_persistent_capture(self):
        self.generator.release_traces()

# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation for this run's pinned text-only autoport."""

import json
import math
import os

import torch

import ttnn

from .benchmark_phases import PhaseRecorder
from .generator import Gemma4Generator
from .model import Gemma4Model


class Gemma4ForCausalLM:
    # Split readback and stale-input/page-refresh residency pass adapter checks.
    model_capabilities = {
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_device_penalties": False,
        "supports_async_decode": True,
        "supports_chunked_prefill": True,
        "supports_prefix_caching": True,
    }
    decode_input_update_contract = 1
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or max_batch_size not in (1, 8, 16, 32):
            raise ValueError("This autoport requires TP4 and a prepared batch1/8/16/32")
        obj = cls()
        obj.mesh = mesh_device
        obj.batch = max_batch_size
        obj.context = max_seq_len
        reduced = os.environ.get("GEMMA4_SERVING_TEST_LAYERS")
        indices = tuple(int(i) for i in reduced.split(",")) if reduced else None
        obj.model = Gemma4Model(mesh_device, layer_indices=indices)
        # Scheduler-owned hybrid pages use absolute logical indexing, not the
        # standalone generator's bounded physical ring addressing.
        for layer in obj.model.layers:
            layer.cache_position_modulo = None
        obj.generator = None
        obj.page_revision = 0
        obj.sampler_seeds = [0] * max_batch_size
        obj.active_decode_rows = 0
        obj.tt_supported_decode_batch_sizes = [max_batch_size]
        print(
            json.dumps(
                {
                    "event": "autoport_registration",
                    "class": f"{cls.__module__}:{cls.__name__}",
                    "module_file": os.path.realpath(__file__),
                    "device_ids": list(mesh_device.get_device_ids()),
                    "mesh_shape": str(mesh_device.shape),
                    "layers": list(obj.model.layer_indices),
                    "precision": obj.model.precision_config,
                }
            ),
            flush=True,
        )
        phase_path = os.environ.get("GEMMA4_BENCHMARK_PHASE_LOG")
        obj._benchmark_phases = (
            PhaseRecorder(
                phase_path,
                dict(
                    module_file=os.path.realpath(__file__),
                    layers=list(obj.model.layer_indices),
                    precision=obj.model.precision_config,
                    device_ids=list(mesh_device.get_device_ids()),
                    batch_size=max_batch_size,
                    context=max_seq_len,
                ),
            )
            if phase_path
            else None
        )
        return obj

    @classmethod
    def get_max_tokens_all_users(cls, *, max_model_len, max_num_seqs=1, **kwargs):
        # The pinned worker divides this synthetic budget by 32, while
        # hybrid normalization gives global attention 64-token blocks.
        # Its separate eight-window allowance covers the five live sliding groups.
        # Cached tails from completed turns share the same LRU pool as global
        # prefixes. Reserve one prior 24K-turn equivalent so growing 32K
        # conversations can retain prefixes while allocating their next turn.
        # Keep the single-request floor conservative; this is a cache budget,
        # not the advertised context or a scheduler token limit.
        target_tokens_per_user = 32768
        prior_turn_tokens_per_user = 24576
        worker_block_size, global_block_size = 32, 64
        global_blocks = max_num_seqs * math.ceil(
            (target_tokens_per_user + prior_turn_tokens_per_user) / global_block_size
        )
        budget = max(max_model_len, worker_block_size * global_blocks)
        print(
            json.dumps(
                {
                    "event": "serving_pool_budget",
                    "synthetic_token_budget": budget,
                    "target_tokens_per_user": target_tokens_per_user,
                    "prior_turn_reserve_tokens_per_user": prior_turn_tokens_per_user,
                    "global_block_size": global_block_size,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "basis": "normalized hybrid blocks plus prior-turn APC reserve; worker adds live sliding/output reserves",
                }
            ),
            flush=True,
        )
        return budget

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        config = vllm_config.model_config.hf_config
        config = getattr(config, "text_config", config)
        result = {}
        for i, kind in enumerate(config.layer_types):
            sliding = kind == "sliding_attention"
            args = dict(
                block_size=32,
                num_kv_heads=config.num_key_value_heads if sliding else config.num_global_key_value_heads,
                head_size=config.head_dim if sliding else config.global_head_dim,
                dtype=torch.bfloat16,
            )
            result[f"model.layers.{i}.self_attn"] = (
                SlidingWindowSpec(**args, sliding_window=config.sliding_window)
                if sliding
                else FullAttentionSpec(**args)
            )
        return result

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        self._memory_event("before_serving_cache")
        cache = []
        width = (self.context + 127) // 128 * 4
        maps, tables = [], []
        self.cache_page_factors = []
        self.cache_storage = {}
        allocation = []
        for index, layer in zip(self.model.layer_indices, self.model.layers):
            shape, _, tensor_idx = per_layer_specs[index]
            blocks, heads, block_size, head_dim = shape
            if (heads, head_dim) != (layer.nkv, layer.head_dim) or block_size % 32:
                raise ValueError(f"KV specification does not match layer{index}: {shape}")
            factor = block_size // 32
            physical_shape = (blocks * factor, heads, 32, head_dim)
            volume = blocks * heads * block_size * head_dim
            if tensor_idx not in self.cache_storage:
                self.cache_storage[tensor_idx] = layer.allocate_cache(num_blocks=blocks * factor)
            backing = self.cache_storage[tensor_idx]
            if any(math.prod(tensor.shape) != volume or tensor.dtype != layer.cache_dtype for tensor in backing):
                raise ValueError("Shared scheduler cache must have equal physical size and dtype")
            # Views share tile storage. The scheduler allocates disjoint block
            # IDs to hybrid groups, so their different head layouts cannot alias
            # live requests. Each scheduler block occupies the same tile count.
            views = tuple(ttnn.experimental.view(tensor, physical_shape) for tensor in backing)
            cache.append(views)
            self.cache_page_factors.append(factor)
            allocation.append(
                dict(
                    layer=index, tensor_idx=tensor_idx, scheduler_shape=list(shape), physical_shape=list(physical_shape)
                )
            )
            host = torch.zeros(self.batch, width, dtype=torch.int32)
            maps.append(host)
            tables.append(self.model.upload(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT))
        self._memory_event("after_serving_cache")
        self.generator = Gemma4Generator(
            self.mesh,
            model=self.model,
            batch_size=self.batch,
            max_seq_len=self.context,
            kv_cache=cache,
            page_table=tables,
            host_page_tables=maps,
        )
        self.kv_cache = cache
        print(
            json.dumps(
                {"event": "serving_cache_allocation", "unique_buffers": len(self.cache_storage), "layers": allocation}
            ),
            flush=True,
        )
        return cache

    @staticmethod
    def _physical_pages(source, factor, context):
        """Translate scheduler blocks to the decoder's 32-token tile pages."""
        if factor < 1:
            raise ValueError("Scheduler blocks must contain at least one tile page")
        columns = (context + 32 * factor - 1) // (32 * factor)
        if bool((source[:, columns:] != 0).any()):
            raise ValueError("Scheduler columns beyond context must be zero padding")
        source = source[:, :columns].to(torch.int32)
        offsets = torch.arange(factor, dtype=torch.int32)
        return (source.unsqueeze(-1) * factor + offsets).flatten(1)[:, : (context + 31) // 32]

    def _pages(self, page_table, page_tables_per_layer, kv_cache):
        if kv_cache is not self.kv_cache:
            raise ValueError("vLLM must pass the allocator's exact cache object")
        maps = []
        for index, target, factor in zip(self.model.layer_indices, self.generator.page_table, self.cache_page_factors):
            source = page_table if page_tables_per_layer is None else page_tables_per_layer[index]
            source = self._physical_pages(source, factor, self.context)
            host = torch.zeros(tuple(target.shape), dtype=torch.int32)
            if source.shape[1] > host.shape[1] or source.shape[0] > host.shape[0]:
                raise ValueError("Scheduler page table exceeds prepared capacity")
            host[: source.shape[0], : source.shape[1]] = source.to(torch.int32)
            maps.append(host)
        self.page_revision += 1
        self.generator.refresh_page_tables(maps, revision=self.page_revision)

    def _sampling(self, params, *, reset_seed=True, positions=None):
        def padded(value, default):
            if hasattr(value, "tolist"):
                value = value.tolist()
            values = list(value) if isinstance(value, (list, tuple)) else [value] * self.batch
            if len(values) > self.batch:
                raise ValueError("Sampling parameter rows exceed prepared batch")
            return values + [default] * (self.batch - len(values))

        if reset_seed:
            self.sampler_seeds = [
                int(value) % (2**31) if value is not None else int(torch.randint(0, 2**30, ()))
                for value in padded(params.seed, None)
            ]
        if reset_seed and positions is not None:
            offsets = padded(positions, 0)
            self.sampler_seeds = [
                (seed + max(0, int(offset))) % (2**32) for seed, offset in zip(self.sampler_seeds, offsets)
            ]
        self.generator.set_sampling(
            temperature=padded(params.temperature, 0.0),
            top_k=padded(params.top_k, 1),
            top_p=padded(params.top_p, 1.0),
            seed=self.sampler_seeds,
            reset_seed=reset_seed,
        )

    def _advance_sampler(self):
        # Match the uint32 device counter across the signed integer boundary.
        self.sampler_seeds = [(value + 1) % (2**32) for value in self.sampler_seeds]

    def _log(self, phase, device, **details):
        print(
            json.dumps(
                {
                    "event": "sampling_mode",
                    "phase": phase,
                    "mode": "device" if device else "host_compatibility",
                    "device_token_rows_checked_before_step": getattr(self, "device_token_rows_checked", 0),
                    **details,
                }
            ),
            flush=True,
        )

    def _validate_token_ids(self, token_ids):
        """Audit already-read device samples without another device transfer."""
        if token_ids.numel():
            minimum, maximum = torch.aminmax(token_ids)
            if int(minimum) < 0 or int(maximum) >= self.model.config.vocab_size:
                raise ValueError(f"Device sampled token outside vocabulary: [{int(minimum)}, {int(maximum)}]")
            self.device_token_rows_checked = getattr(self, "device_token_rows_checked", 0) + token_ids.numel()
        return token_ids

    def _memory_event(self, phase):
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        fields = (
            "num_banks",
            "total_bytes_allocated_per_bank",
            "total_bytes_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        )
        print(
            json.dumps(
                {
                    "event": "serving_memory",
                    "phase": phase,
                    "allocator": {name: int(getattr(view, name)) for name in fields},
                    "program_cache_entries": self.mesh.num_program_cache_entries(),
                    "generator_counters": dict(self.generator.counters) if self.generator is not None else None,
                }
            ),
            flush=True,
        )

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device):
        if kv_cache is not self.kv_cache:
            raise ValueError("Warmup must use served KV allocation")
        if self.generator.model_trace is None:
            self.generator.prepare_decode()
            self._memory_event("after_serving_prepare")

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device):
        self.warmup_model_prefill(kv_cache, enable_trace, can_sample_on_device)

    def prefill_forward(
        self,
        tokens,
        *,
        start_pos,
        prompt_lens,
        page_table,
        kv_cache,
        sampling_params=None,
        page_tables_per_layer=None,
        empty_slots=None,
        **kwargs,
    ):
        recorder = getattr(self, "_benchmark_phases", None)
        phase_step = None
        if recorder is not None:
            starts, ends = list(map(int, start_pos)), list(map(int, prompt_lens))
            phase_step = recorder.begin(
                "prefill",
                batch_size=self.batch,
                sampling_mode="device" if sampling_params is not None else "host_compatibility",
                starts=starts,
                ends=ends,
                active_rows=sum(e > s for s, e in zip(starts, ends)),
                token_shape=list(tokens.shape),
            )
        self._memory_event("before_serving_prefill")
        self._pages(page_table, page_tables_per_layer, kv_cache)
        device = sampling_params is not None
        self._log("prefill", device, starts=list(map(int, start_pos)), ends=list(map(int, prompt_lens)))
        if device:
            self._sampling(sampling_params, positions=prompt_lens)
        result = self.generator.prefill_forward(
            tokens,
            start_pos=start_pos,
            prompt_lens=prompt_lens,
            # Attention state follows the submitted page-table rows. The
            # plugin empty_slots indices address recurrent model state.
            empty_slots=None,
            page_table=self.generator.page_table,
            kv_cache=kv_cache,
            host_sampling=not device,
        )
        self._memory_event("after_serving_prefill")
        if device:
            self._advance_sampler()
            result = self._validate_token_ids(
                torch.tensor(result[: len(prompt_lens)], dtype=torch.int64).reshape(-1, 1)
            )
        if recorder is not None:
            recorder.complete(phase_step)
        return result

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        reload_inputs,
        reload_page_table,
        reload_sampling_params,
        reset_sampling_state,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        **kwargs,
    ):
        recorder = getattr(self, "_benchmark_phases", None)
        phase_step = None
        if recorder is not None:
            positions = recorder.decode_positions(start_pos, reload_inputs)
            phase_step = recorder.begin(
                "decode",
                batch_size=self.batch,
                sampling_mode="device" if sampling_params is not None else "host_compatibility",
                effective_positions=positions,
                positions=positions,
                active_rows=sum(p >= 0 for p in positions),
                token_shape=list(tokens.shape),
                reload_inputs=bool(reload_inputs),
                reload_page_table=bool(reload_page_table),
            )
        if "reset_batch" in kwargs:
            raise TypeError("Legacy reset_batch is incompatible with decode contract version1")
        if reload_inputs and reload_page_table:
            raise ValueError("Full input reload already includes page tables")
        if reload_inputs:
            self.active_decode_rows = int((torch.as_tensor(start_pos) >= 0).sum())
        if slot_remap is not None:
            remap = list(map(int, slot_remap))
            if len(remap) > self.batch or any(i < 0 or i >= self.batch for i in remap):
                raise ValueError("Invalid sampler slot remap")
            previous = self.sampler_seeds[:]
            self.sampler_seeds[: len(remap)] = [previous[i] for i in remap]
            seeds = torch.zeros(32, dtype=torch.uint32)
            seeds[: self.batch] = torch.tensor(self.sampler_seeds, dtype=torch.uint32)
            self.generator._copy(seeds, self.generator.sampler.seeds_tt_tensor)
        if reload_inputs or reload_page_table:
            self._pages(page_table, kwargs.get("page_tables_per_layer"), kv_cache)
        if sampling_params is not None and (reload_sampling_params or reset_sampling_state):
            if reset_sampling_state and not reload_inputs:
                raise ValueError("Sampling reset requires authoritative positions")
            self._sampling(
                sampling_params,
                reset_seed=reset_sampling_state,
                positions=(start_pos + 1) if reset_sampling_state else None,
            )
        self._log(
            "decode",
            sampling_params is not None,
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            active_rows=self.active_decode_rows,
        )
        result = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=self.generator.page_table,
            kv_cache=kv_cache,
            reload_inputs=reload_inputs,
            read_from_device=read_from_device,
            host_sampling=sampling_params is None,
        )
        if sampling_params is not None:
            self._advance_sampler()
        if read_from_device:
            if sampling_params is not None:
                result = self._validate_token_ids(torch.tensor(result, dtype=torch.int64).reshape(self.batch, 1))
            else:
                result = result.reshape(self.batch, 1, -1)
            if recorder is not None:
                recorder.complete(phase_step)
        elif recorder is not None:
            recorder.submitted_output(phase_step, result)
        return result

    def read_decode_output(self, output, async_read=False):
        result = self.generator.read_decode_output(output, async_read=async_read)
        recorder = getattr(self, "_benchmark_phases", None)
        if recorder is not None:
            recorder.bind_readback(output, result[0] if async_read else result)
        return result

    def process_decode_output_host(self, output, is_tokens=False):
        result = self.generator.process_decode_output_host(output, is_tokens=is_tokens)
        result = self._validate_token_ids(result) if is_tokens else result
        recorder = getattr(self, "_benchmark_phases", None)
        if recorder is not None:
            # Async callers already waited the existing readback event; sync
            # conversion above already materialized its existing device read.
            recorder.complete_output(output)
        return result

    def release_persistent_capture(self):
        if self.generator is not None:
            self.generator.release_persistent_capture()

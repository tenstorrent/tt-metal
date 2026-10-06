# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation; execution and sampling belong to KolibriGenerator."""

import json
import math
import os
import secrets
import time

import torch

import ttnn

from .checkpoint import CONTEXT
from .generator import CacheState, KolibriGenerator
from .model import DRAM, KolibriModel
from .precision import load_precision_config


class KolibriForCausalLM:
    # Upstream inspection protocol; the TT worker executes the low-level APIs.
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Use the TT prefill/decode interface")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("Use the TT prefill/decode interface")

    def compute_logits(self, hidden_states, **kwargs):
        raise NotImplementedError("Use the TT prefill/decode interface")

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,  # Direct adapter trace/deferred-read/stale-input probe passed.
        "supports_sample_on_device": True,
        "supports_chunked_prefill": True,
        "supports_intermediate_prefill_output_mask": True,
        "supports_prefill_sampling_origins": True,
        "supports_device_penalties": False,
        "max_device_top_k": 32,
    }
    decode_input_update_contract = 1
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=CONTEXT, tt_data_parallel=1, optimizations=None
    ):
        from .benchmark_timing import install

        install()
        if tt_data_parallel != 1 or mesh_device.get_num_devices() != 4:
            raise ValueError("Kolibri requires the authorized TP4 mesh")
        obj = cls()
        obj.mesh_device = mesh_device
        obj.batch_size, obj.capacity = max_batch_size, max_seq_len
        obj.host_compat = os.environ.get("KOLIBRI_VLLM_HOST_COMPAT", "0") == "1"
        indices = os.environ.get("KOLIBRI_VLLM_LAYERS")
        obj.model = KolibriModel(
            mesh_device,
            rope_capacity=max_seq_len,
            layer_indices=None if indices is None else json.loads(indices),
            precision_config=load_precision_config(),
        )
        # vLLM supplies absolute logical page tables, not the standalone ring.
        for layer in obj.model.layers:
            layer.sliding_cache_tokens = None
        obj.event_detail = os.environ.get("KOLIBRI_VLLM_EVENT_DETAIL", "1") == "1"
        obj.generator = None
        obj.event(
            "initialize",
            adapter=f"{cls.__module__}:{cls.__name__}",
            module_file=__file__,
            precision=obj.model.precision_config,
            layers=obj.model.layer_indices,
            max_model_len=max_seq_len,
            max_num_seqs=max_batch_size,
            host_compat=obj.host_compat,
        )
        return obj

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        return kwargs["max_model_len"]

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        cfg = vllm_config.model_config.hf_config
        block = vllm_config.cache_config.block_size
        if block != 32:
            raise ValueError("Kolibri TT cache requires --block-size 32")
        common = dict(block_size=32, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
        return {
            f"model.layers.{i}.self_attn": (
                SlidingWindowSpec(**common, sliding_window=513)
                if kind == "sliding_attention"
                else FullAttentionSpec(**common)
            )
            for i, kind in enumerate(cfg.layer_types)
        }

    def event(self, kind, **data):
        path = os.environ.get("KOLIBRI_VLLM_EVENTS")
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(dict(time=time.time(), pid=os.getpid(), event=kind, **data), default=str) + "\n")

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        unique, cache = {}, []
        for index in self.model.layer_indices:
            shape, dtype, tensor_index = per_layer_specs[index]
            if dtype != torch.bfloat16 or shape[1:] != (1, 32, 128):
                raise ValueError(f"KV spec differs from selected policy: {shape}, {dtype}")
            if tensor_index not in unique:
                unique[tensor_index] = tuple(
                    ttnn.zeros(
                        shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, memory_config=DRAM
                    )
                    for _ in range(2)
                )
            cache.append(unique[tensor_index])
        physical = math.ceil(self.capacity / 512) * 512
        pages = {i: torch.zeros(self.batch_size, physical // 32, dtype=torch.int32) for i in range(len(cache))}
        tables = {i: self.model.tensor(p, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for i, p in pages.items()}
        state = CacheState(cache, tables, pages, physical, self.batch_size)
        self.generator = KolibriGenerator(
            self.model, batch_size=self.batch_size, capacity=self.capacity, cache_state=state
        )
        self.generator.event_sink = self.event
        self.event(
            "allocate_cache", specs=per_layer_specs, unique_buffers=len(unique), owns_cache=self.generator.owns_cache
        )
        return cache

    def _prepare(self, kv_cache):
        g = self.generator
        if g is None or kv_cache is not g.state.layers:
            raise ValueError("vLLM must pass the exact allocated cache")
        if not g.prepared:
            started = time.monotonic()
            g.prepare()
            if os.environ.get("KOLIBRI_VLLM_STARTUP_CONTROL"):
                from ..tests.vllm_startup_control import run

                run(g)
            self.event(
                "prepared",
                elapsed=time.monotonic() - started,
                traces=g.traces,
                counters=dict(g.counters),
                signatures={
                    k: dict(batch=g.batch_size, capacity=g.capacity, buffers=[id(v) for v in g.state.layers])
                    for k in g.traces
                },
            )
        return g

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, **kwargs):
        self._prepare(kv_cache)

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs):
        self._prepare(kv_cache)

    def _tables(self, page_table, page_tables_per_layer, slots=None):
        g = self.generator
        tables = {}
        for i, index in enumerate(self.model.layer_indices):
            src = page_table if page_tables_per_layer is None else page_tables_per_layer[index]
            if slots is None:
                dst = torch.zeros_like(g.state.host_page_tables[i])
                dst[: src.shape[0], : src.shape[1]] = src
            else:
                dst = g.state.host_page_tables[i].clone()
                for row, slot in enumerate(slots):
                    dst[slot].zero_()
                    dst[slot, : src.shape[1]] = src[row]
            tables[i] = dst
        return tables

    def _ensure_sampling_state(self):
        if not hasattr(self, "_sampling_origins"):
            self._sampling_origins = [None] * self.generator.batch_size
            self._sampling_seeds = [None] * self.generator.batch_size

    def _sampling(self, params, slots=None, positions=None):
        if params is None:
            if not self.host_compat:
                raise RuntimeError("Host sampling requires explicit KOLIBRI_VLLM_HOST_COMPAT=1")
            self.event("explicit_host_compat")
            return False
        g = self.generator
        self._ensure_sampling_state()
        slots = list(range(g.batch_size)) if slots is None else slots
        positions = [None] * len(slots) if positions is None else torch.as_tensor(positions).flatten().tolist()
        values = dict(
            top_k=[1] * g.batch_size,
            top_p=[0.0] * g.batch_size,
            temperature=[1.0] * g.batch_size,
            seed=[0] * g.batch_size,
        )
        for key in values:
            raw = getattr(params, key)
            raw = raw if isinstance(raw, (list, tuple)) else [raw] * len(slots)
            for row, slot in enumerate(slots):
                if key == "seed":
                    if positions[row] is not None and positions[row] < 0:
                        continue
                    if self._sampling_seeds[slot] is None:
                        self._sampling_seeds[slot] = secrets.randbelow(1000000) if raw[row] is None else raw[row]
                    values[key][slot] = self._sampling_seeds[slot]
                else:
                    values[key][slot] = raw[row]
        offsets = [0] * g.batch_size
        for slot, position in zip(slots, positions):
            origin = self._sampling_origins[slot]
            if position is not None and position >= 0:
                if origin is None:
                    raise ValueError(f"Active sampling slot {slot} has no prefill origin at position {position}")
                offsets[slot] = max(0, position - origin)
        g.set_sampling(**values, seed_offset=offsets)
        return True

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        sampling_params=None,
        start_pos=None,
        page_tables_per_layer=None,
        prefill_output_mask=None,
        original_prompt_lens=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported prefill options: {kwargs.keys()}")
        g = self._prepare(kv_cache)
        slots = list(range(len(prompt_lens))) if empty_slots is None else empty_slots
        self._ensure_sampling_state()
        output_mask = [True] * len(slots) if prefill_output_mask is None else prefill_output_mask
        origins = prompt_lens if original_prompt_lens is None else original_prompt_lens
        for row, slot in enumerate(slots):
            if output_mask[row]:
                self._sampling_origins[slot] = int(origins[row]) - 1
                self._sampling_seeds[slot] = None
        sampled = sampling_params is not None
        if prefill_output_mask is None or any(prefill_output_mask):
            positions = [int(prompt_lens[row]) - 1 if output_mask[row] else -1 for row in range(len(slots))]
            sampled = self._sampling(sampling_params, slots, positions=positions)
        self.event(
            "prefill",
            lengths=prompt_lens.tolist(),
            starts=start_pos.tolist(),
            slots=slots,
            traces=g.traces,
            counters=dict(g.counters),
        )
        starts = torch.as_tensor(start_pos, dtype=torch.int64)
        lengths = torch.as_tensor(prompt_lens, dtype=torch.int64) - starts
        chunks = torch.zeros((len(lengths), int(lengths.max())), dtype=tokens.dtype)
        for row, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
            chunks[row, :length] = tokens[row, start : start + length]
        result = g.prefill_forward(
            chunks,
            page_table=self._tables(page_table, page_tables_per_layer, slots),
            kv_cache=kv_cache,
            prompt_lens=lengths,
            start_pos=start_pos,
            slots=slots,
            output_mask=prefill_output_mask,
            sample_on_device=sampled,
        )
        return result[slots].reshape(-1, 1) if sampled else result

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        page_tables_per_layer=None,
        slot_remap=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported decode options: {kwargs.keys()}")
        if not enable_trace:
            raise ValueError("Kolibri serving requires decode traces")
        g = self._prepare(kv_cache)
        self._ensure_sampling_state()
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap).flatten().tolist()
            if sorted(remap) != list(range(g.batch_size)):
                raise ValueError(f"Invalid sampling slot permutation: {remap}")
            self._sampling_origins = [self._sampling_origins[slot] for slot in remap]
            self._sampling_seeds = [self._sampling_seeds[slot] for slot in remap]
        sampled = sampling_params is not None
        if reload_sampling_params or reset_sampling_state or slot_remap is not None or not sampled:
            self._sampling(sampling_params, positions=start_pos)
        tables = self._tables(page_table, page_tables_per_layer) if reload_inputs or reload_page_table else None
        result = g.decode_forward(
            tokens.flatten() if reload_inputs else None,
            start_pos.flatten() if reload_inputs else None,
            page_table=tables,
            kv_cache=kv_cache,
            sample_on_device=sampled,
            read_from_device=False,
        )
        if self.event_detail or reload_inputs or reload_page_table:
            self.event(
                "decode",
                reload_inputs=reload_inputs,
                reload_page_table=reload_page_table,
                sampling=sampled,
                positions=start_pos.tolist(),
                traces=g.traces,
                counters=dict(g.counters),
            )
        if not read_from_device:
            return result
        return self.process_decode_output_host(self.read_decode_output(result), is_tokens=sampled)

    def read_decode_output(self, tt_out, async_read=False):
        # Sampled tokens are replicated: read exactly one rank. Compatibility
        # logits remain vocabulary-sharded and explicitly read all four ranks.
        shards = ttnn.get_device_tensors(tt_out)
        is_tokens = tt_out is self.generator.tokens
        out = [x.cpu(blocking=not async_read) for x in (shards[:1] if is_tokens else shards)]
        self.generator.counters["token_readbacks" if is_tokens else "logit_readbacks"] += 1
        return (out, [ttnn.record_event(self.mesh_device, 0)]) if async_read else out

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if isinstance(tt_out, ttnn.Tensor):
            tt_out = self.read_decode_output(tt_out)
        if is_tokens:
            return ttnn.to_torch(tt_out[0]).flatten().long().reshape(-1, 1)
        return torch.cat([ttnn.to_torch(x).float() for x in tt_out], -1)[0, 0].unsqueeze(1)

    def release_persistent_capture(self):
        """Pinned plugin shutdown hook, called while the mesh is still open."""
        g = getattr(self, "generator", None)
        if g is not None:
            g.close()

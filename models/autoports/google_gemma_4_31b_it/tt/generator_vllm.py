# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation for the pinned autoport generator."""

import json
import math
import os
import secrets
import time

import torch

import ttnn

from .generator import Gemma4Generator
from .model import Gemma4Model
from .precision import load_serving_precision


class Gemma4ForCausalLM:
    # Reduced adapter tests cover overlap, allocator page growth and cache hits
    # against synchronous controls; full-model gates remain in readiness_vllm.
    model_capabilities = {
        "supports_sample_on_device": True,
        "supports_async_decode": True,
        "supports_prefix_caching": True,
        "max_device_top_k": 32,
        "supports_device_penalties": False,
        "supports_chunked_prefill": True,
    }
    decode_input_update_contract = 1
    accepts_intermediate_prefill_mask = True
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True

    @classmethod
    def get_max_tokens_all_users(cls, *, max_model_len, max_num_seqs, **kwargs):
        # The plugin divides this budget by its 32-token base block. Hybrid
        # normalization gives full-attention blocks 64 tokens, so each base
        # budget token buys two full-attention tokens. Keep the context floor
        # for fixed-width native page-table validation; the plugin separately
        # reserves sliding groups and in-flight output blocks.
        return max(max_model_len, max_num_seqs * 32768 // 2)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        c = vllm_config.model_config.hf_config.text_config
        block = vllm_config.cache_config.block_size
        specs = {}
        for i, kind in enumerate(c.layer_types):
            sliding = kind == "sliding_attention"
            heads = c.num_key_value_heads if sliding else c.num_global_key_value_heads
            dim = c.head_dim if sliding else c.global_head_dim
            args = dict(block_size=block, num_kv_heads=heads, head_size=dim, dtype=torch.bfloat16)
            spec = SlidingWindowSpec(**args, sliding_window=c.sliding_window) if sliding else FullAttentionSpec(**args)
            specs[f"model.layers.{i}.self_attn"] = spec
        return specs

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or tuple(mesh_device.shape) != (1, 4):
            raise ValueError("This autoport requires one TP4 Blackhole mesh")
        if max_batch_size not in (1, 8, 16, 32):
            raise ValueError("Configure one of the prepared batches 1, 8, 16, 32")
        self = cls()
        self.mesh = mesh_device
        self.batch = max_batch_size
        self.context = max_seq_len
        reduced = os.environ.get("GEMMA4_VLLM_REDUCED") == "1"
        precision = load_serving_precision(
            os.environ.get("GEMMA4_EXPERIMENTAL_PRECISION"),
            os.environ.get("GEMMA4_EXPERIMENTAL_PRECISION_SHA256"),
        )
        self.model = Gemma4Model(mesh_device, layer_indices=(0, 5) if reduced else None, precision_config=precision)
        if self.model.precision_config != precision:
            raise RuntimeError("Serving precision differs from the requested pinned policy")
        # vLLM supplies logical hybrid block tables, not private cyclic storage.
        for layer in self.model.layers:
            layer.cache_position_modulo = None
        self.generator = None
        self.kv_cache = None
        self.page_revision = 0
        print("AUTOPORT_ADAPTER", __file__, cls.__name__, json.dumps(self.model.precision_config), flush=True)
        if os.environ.get("GEMMA4_BENCHMARK_PORT"):
            from .benchmark_timing import install

            self.benchmark_ledger = install(self)
        return self

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        if len(per_layer_specs) != self.model.config.num_hidden_layers:
            raise ValueError("Hybrid KV plan must cover every layer")
        per_layer_specs = [per_layer_specs[i] for i in self.model.layer_indices]
        # vLLM can unify 32-token sliding and 64-token full blocks. Translate
        # both into native 32-token TT pages without changing block ownership.
        storage = {}
        canonical = {}
        for shape, _, index in per_layer_specs:
            size = math.prod(shape)
            if index not in canonical or size > math.prod(canonical[index]):
                canonical[index] = shape
        for index, shape in canonical.items():
            storage[index] = tuple(
                ttnn.zeros(
                    shape,
                    dtype=getattr(ttnn, self.model.precision_config["kv_cache_dtype"]),
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
        cache = []
        self.block_sizes = []
        for layer, (shape, _, index) in zip(self.model.layers, per_layer_specs):
            blocks, heads, block, dim = shape
            if heads != layer.nkv or dim != layer.head_dim or block % 32:
                raise ValueError("Hybrid KV shape does not match the pinned layer")
            if math.prod(shape) != math.prod(canonical[index]):
                raise ValueError("Shared hybrid groups must have equal bytes per physical block")
            capacity = math.prod(canonical[index]) // (heads * 32 * dim)
            # This is a physical tile interpretation, not a logical reshape:
            # groups own disjoint equally-sized physical block IDs. A reshape
            # would reorder/copy tiles when head_dim changes and duplicate KV.
            cache.append(tuple(ttnn.experimental.view(t, (capacity, heads, 32, dim)) for t in storage[index]))
            self.block_sizes.append(block)
        self._emit(
            dict(
                event="cache_layout",
                canonical_shapes=canonical,
                layer_specs=per_layer_specs,
                native_shapes=[list(pair[0].shape) for pair in cache],
            )
        )
        self.kv_cache = cache
        width = math.ceil(self.context / 128) * 4
        maps = [torch.zeros(self.batch, width, dtype=torch.int32) for _ in cache]
        tables = [self.model.upload(p, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT) for p in maps]
        self.generator = Gemma4Generator(
            self.mesh,
            model=self.model,
            batch_size=self.batch,
            max_seq_len=self.context,
            kv_cache=cache,
            page_table=tables,
            host_page_tables=maps,
        )
        if os.environ.get("GEMMA4_VLLM_EVENTS") == "1":
            self.generator.event_sink = self._emit
        self._memory_event("cache_allocated", force=True)
        return cache

    def _emit(self, fields, *, force=False):
        if force or os.environ.get("GEMMA4_VLLM_EVENTS") == "1":
            print("AUTOPORT_EVENT " + json.dumps({"time": time.monotonic(), **fields}, default=str), flush=True)

    def _memory_event(self, phase, *, force=False):
        if not force and os.environ.get("GEMMA4_VLLM_EVENTS") != "1":
            return
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        self._emit(
            dict(
                event="memory",
                phase=phase,
                program_cache_entries=self.mesh.num_program_cache_entries(),
                allocated_bytes_per_chip=int(view.num_banks * view.total_bytes_allocated_per_bank),
                model_allocations=self.model.allocation_bytes,
            ),
            force=force,
        )

    def _check_cache(self, cache):
        if cache is not self.kv_cache:
            raise ValueError("Serving must retain vLLM's allocated cache object")

    def _refresh_pages(self, page_table, page_tables_per_layer=None):
        supplied = (
            [page_tables_per_layer[i] for i in self.model.layer_indices]
            if page_tables_per_layer
            else [page_table] * len(self.model.layers)
        )
        maps = []
        expanded_groups = {}
        for src, block, target in zip(supplied, self.block_sizes, self.generator.page_table):
            ratio = block // 32
            key = (id(src), ratio, tuple(target.shape))
            if key in expanded_groups:
                maps.append(expanded_groups[key])
                continue
            pages = torch.as_tensor(src, dtype=torch.int32)
            expanded = (pages[..., None] * ratio + torch.arange(ratio, dtype=torch.int32)).flatten(1)
            dest = torch.zeros(tuple(target.shape), dtype=torch.int32)
            rows = min(dest.shape[0], expanded.shape[0])
            cols = min(dest.shape[1], expanded.shape[1])
            dest[:rows, :cols] = expanded[:rows, :cols]
            maps.append(dest)
            expanded_groups[key] = dest
        self.page_revision += 1
        self.generator.refresh_page_tables(maps, revision=self.page_revision)
        self._emit(dict(event="page_refresh", revision=self.page_revision, block_sizes=self.block_sizes))

    def warmup_model_prefill(self, kv_cache, **kwargs):
        self._check_cache(kv_cache)
        self.generator.prepare_decode()

    def warmup_model_decode(self, kv_cache, **kwargs):
        self._check_cache(kv_cache)
        self.generator.prepare_decode()
        self._memory_event("warmup_complete", force=True)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        *,
        enable_trace=True,
        read_from_device=True,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        sampling_params=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        if "reset_batch" in kwargs:
            raise TypeError("This adapter requires decode reload contract version 1")
        self._check_cache(kv_cache)
        if not enable_trace:
            raise ValueError("Serving decode requires the generator-owned trace")
        if sampling_params is None and os.environ.get("GEMMA4_ALLOW_HOST_SAMPLING") != "1":
            raise NotImplementedError("Host sampling compatibility has not been enabled")
        if reload_inputs or reload_page_table:
            self._refresh_pages(page_table, page_tables_per_layer)
        if sampling_params is not None and (reload_sampling_params or reset_sampling_state):
            self._set_sampling(
                sampling_params, positions=start_pos if reset_sampling_state else None, reset_state=reset_sampling_state
            )
        output = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=self.generator.page_table,
            kv_cache=kv_cache,
            reload_inputs=reload_inputs,
            read_from_device=False,
            sample=sampling_params is not None,
        )
        if os.environ.get("GEMMA4_VLLM_EVENTS") == "1":
            self._emit(
                dict(
                    event="decode",
                    sample_on_device=sampling_params is not None,
                    reload_inputs=reload_inputs,
                    reload_page_table=reload_page_table,
                    reset_sampling_state=reset_sampling_state,
                    positions=torch.as_tensor(start_pos).tolist(),
                    captures=self.generator.counters["captures"],
                    counters=dict(self.generator.counters),
                    trace_ids=[self.generator.model_trace, self.generator.sample_trace],
                )
            )
        output = {"tensor": output, "is_tokens": sampling_params is not None}
        if not read_from_device:
            return output
        return self.process_decode_output_host(self.read_decode_output(output), is_tokens=sampling_params is not None)

    def _set_sampling(self, params, positions=None, reset_state=True):
        def padded(values, default):
            values = list(values)
            return values + [default] * (self.batch - len(values))

        # Reserve the entire context above each base seed: reset at position p
        # must equal resident seed increments, without a different wrap modulus.
        seed_span = 2**31 - 1 - self.context
        if seed_span <= 0:
            raise ValueError("Context leaves no signed-int32 sampling seed range")
        seeds = [0] * self.batch
        if reset_state:
            # Omitted seeds request fresh randomness. Parameter-only reloads
            # preserve the current stream and do not draw unused random seeds.
            seeds = padded([secrets.randbelow(seed_span) if s is None else int(s) % seed_span for s in params.seed], 0)
            if positions is not None:
                seeds = [seed + max(0, int(pos)) for seed, pos in zip(seeds, positions)]
        self.generator.set_sampling(
            temperature=padded(params.temperature, 0.0),
            top_k=padded(params.top_k, 1),
            top_p=padded(params.top_p, 1.0),
            seed=seeds,
            reset_seed=reset_state,
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        *,
        start_pos=None,
        sampling_params=None,
        page_tables_per_layer=None,
        intermediate_prefill_mask=None,
        **kwargs,
    ):
        self._check_cache(kv_cache)
        self._refresh_pages(page_table, page_tables_per_layer)
        count = tokens.shape[0]
        self._emit(
            dict(
                event="prefill_submission",
                sample_on_device=sampling_params is not None,
                lengths=[int(n) for n in prompt_lens],
                starts=None if start_pos is None else [int(n) for n in start_pos],
                empty_slots=kwargs.get("empty_slots"),
                submitted_rows=count,
            )
        )
        lengths = [int(n) for n in prompt_lens] + [0] * (self.batch - count)
        starts = ([0] * count if start_pos is None else [int(n) for n in start_pos]) + [0] * (self.batch - count)
        padded = torch.zeros(self.batch, tokens.shape[1], dtype=tokens.dtype)
        padded[:count] = tokens
        intermediate_only = intermediate_prefill_mask is not None and all(intermediate_prefill_mask)
        if sampling_params is not None:
            self._set_sampling(sampling_params)
        elif not intermediate_only and os.environ.get("GEMMA4_ALLOW_HOST_SAMPLING") != "1":
            raise ValueError("Host sampling is an explicit compatibility mode")
        result = self.generator.prefill_forward(
            padded,
            page_table=self.generator.page_table,
            kv_cache=kv_cache,
            prompt_lens=lengths,
            start_pos=starts,
            read_from_device=False,
            sample=sampling_params is not None,
        )
        self._memory_event("prefill_complete")
        if intermediate_only:
            # The plugin suppresses every output row. A broadcast placeholder
            # satisfies its logits-shaped boundary without reading device logits.
            return torch.zeros(count, 1, 1).expand(count, 1, self.model.config.vocab_size)
        if sampling_params is None:
            return self.generator._host_logits(result).reshape(self.batch, 1, -1)[:count]
        host, _ = self.process_decode_output_host(self.read_decode_output(result), is_tokens=True)
        return host[:count]

    def read_decode_output(self, output, async_read=False):
        is_tokens = output.get("is_tokens", True) if isinstance(output, dict) else True
        tensor = output["tensor"] if isinstance(output, dict) else output
        source = ttnn.get_device_tensors(tensor)[0] if is_tokens else tensor
        host = {"tensor": source.cpu(blocking=not async_read), "is_tokens": is_tokens, "read_submitted": True}
        self.generator.counters["readbacks"] += 1
        return (host, [ttnn.record_event(self.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, output, is_tokens=False):
        if isinstance(output, dict):
            if output["is_tokens"] != is_tokens:
                raise ValueError("Submission and scheduler output modes disagree")
            if not output.get("read_submitted", False):
                self.generator.counters["readbacks"] += 1
            output = output["tensor"]
        if not is_tokens:
            if os.environ.get("GEMMA4_ALLOW_HOST_SAMPLING") != "1":
                raise ValueError("Host logits require explicit compatibility mode")
            return (
                ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
                .float()
                .reshape(self.batch, 1, -1),
                None,
            )
        first = ttnn.get_device_tensors(output)[0]
        return ttnn.to_torch(first).reshape(-1)[: self.batch].to(torch.int64).reshape(-1, 1), None

    def release_persistent_capture(self):
        if hasattr(self, "benchmark_ledger"):
            self.benchmark_ledger.close()
        if self.generator is None:
            return
        for name in ("model_trace", "sample_trace", "buffered_sample_trace", "edge_trace"):
            trace = getattr(self.generator, name)
            if trace is not None:
                ttnn.release_trace(self.mesh, trace)
                self._emit(dict(event="shutdown_release", mode=name, trace_id=trace))
                setattr(self.generator, name, None)

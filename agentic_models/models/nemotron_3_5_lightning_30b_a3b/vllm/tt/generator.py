# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""External-cache driver for the TP4 Nemotron-H model, behind the vLLM adapter.

The model and sampling traces communicate through preallocated logits and token
buffers. Host logits are an explicit diagnostic/compatibility API boundary.
"""

import hashlib
import math
import os
import secrets
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import torch

import ttnn
from agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt import capacity
from agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt.model import ModelCache, NemotronModel
from agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt.wide_sampling import WideSampling
from models.common.sampling.generator import (
    SamplingGenerator,
    SamplingParams,
    _hash_request_seed_to_device_seed,
    format_sampling_params,
)

from .tt_ccl import TT_CCL

UNBOUNDED_TOP_K = int(os.environ.get("NEMOTRON_UNBOUNDED_TOP_K", "64"))


@dataclass(frozen=True)
class DecodeOutput:
    snapshot: object
    event: object
    slots: tuple
    is_tokens: bool = True


class NemotronGenerator:
    @classmethod
    def get_max_tokens_all_users(
        cls, *, model_name=None, num_devices=4, tt_data_parallel=1, max_model_len=262144, max_num_seqs=1
    ):
        """Aggregate TP4 resident-token budget, before caller output reservation."""
        return capacity.get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )

    def __init__(
        self,
        mesh_device,
        *,
        checkpoint=None,
        max_seq_len=262144,
        max_batch_size=1,
        greedy_strategy="split",
        per_layer_specs=None,
        kv_cache=None,
        optimized_terminal=True,
        precision_config=None,
        batched_prefill=False,
    ):
        self.mesh = mesh_device
        self.model = NemotronModel(
            mesh_device,
            checkpoint=checkpoint,
            optimized_terminal=optimized_terminal,
            precision_config=precision_config,
            batched_prefill=batched_prefill,
        )
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        if greedy_strategy not in ("split", "argmax"):
            raise ValueError("greedy_strategy must be split or argmax")
        self.greedy_strategy = greedy_strategy
        self.cache = self.model.allocate_cache(
            context_length=max_seq_len,
            num_slots=max_batch_size,
            per_layer_specs=per_layer_specs,
            existing_cache=kv_cache,
        )
        self.pages_per_slot = math.ceil(max_seq_len / 32)
        self.page_host = torch.arange(self.pages_per_slot * max_batch_size, dtype=torch.int32).reshape(
            max_batch_size, -1
        )
        self.layer_page_host = {
            layer.layer_idx: self.page_host % self.cache.physical_blocks[layer.layer_idx]
            for layer in self.model.layers
            if layer.kind == "attention"
        }
        self.layer_page_tables = {
            index: [self._integer(row[None]) for row in rows] for index, rows in self.layer_page_host.items()
        }
        self.page_host = next(iter(self.layer_page_host.values())).clone()
        self.page_tables = next(iter(self.layer_page_tables.values()))
        self.positions = [self._integer(torch.zeros(1, dtype=torch.int32)) for _ in range(max_batch_size)]
        self.prefill_positions = self._integer(torch.arange(max_seq_len, dtype=torch.int32).reshape(-1, 1))
        self.token_input = self._integer(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32)
        self.logits = self.model.tensor(torch.zeros(1, 1, 32, self.model.config.vocab_size), dim=3)
        sampler_args = SimpleNamespace(
            vocab_size=self.model.config.vocab_size,
            max_batch_size=max_batch_size,
            max_top_k=32,
            cluster_shape=tuple(mesh_device.shape),
            sampling_all_gather_axis=1,
        )
        # Only force-argmax uses CCL: two barrier and four all-gather
        # semaphores on the flattened mesh. Reserve both buffers before any
        # model warmup/capture; unused axes/reduce-scatter families otherwise
        # consume the L1 margin required by the inherited grouped MoE kernel.
        self.sampler_ccl = TT_CCL(mesh_device, eager=False)
        self.sampler_ccl.get_and_cycle_barrier_semaphore_handle()
        self.sampler_ccl.get_and_cycle_ag_semaphore_handles()
        self.sampling = SamplingGenerator(args=sampler_args, mesh_device=mesh_device, tt_ccl=self.sampler_ccl)
        self.sampling.tt_sampling._allow_force_argmax_sampling = greedy_strategy == "argmax"
        self.wide_sampling = WideSampling(self.sampling, mesh_device)
        self.wide_output = self._integer(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32)
        self._sampler_params = SamplingParams(temperature=1.0, top_k=1, top_p=0.0)
        self._set_sampling(self._sampler_params)
        self.zero_templates = {}
        for tensor in self.cache.tensors():
            key = (tuple(tensor.shape), tensor.dtype)
            if key not in self.zero_templates:
                self.zero_templates[key] = self.model.tensor(torch.zeros(tuple(tensor.shape)), dtype=tensor.dtype)
        self.decode_traces = {}
        self.sampling_traces = {}
        self.counters = Counter()
        self._prepared = False
        self.fixed_slots = tuple(range(max_batch_size))
        # perf/multiuser: decode traces per bucket (first B slots). The plugin condenses
        # decode rows to 0..n-1 and remaps state slots, so a bucket covers every live row.
        bucket_env = os.environ.get("NEMOTRON_DECODE_BUCKETS", "1,2,4,8,16,32")
        buckets = {int(b) for b in bucket_env.split(",") if b.strip() and int(b) <= max_batch_size}
        self.decode_buckets = tuple(sorted(buckets | {max_batch_size}))
        self.batched_decode = self.cache.mamba is not None
        # Raw output buffers are reused. The plugin snapshots each submission
        # before submitting again; preserve logical rows until that boundary.
        self._device_output_slots = {}
        self.active_host = torch.zeros(32, dtype=torch.float32)
        self.active = self.model.tensor(self.active_host.reshape(1, 32), dtype=ttnn.float32)
        self.sampler_active = self.model.tensor(self.active_host.reshape(1, 32), dtype=ttnn.float32)
        self.prefill_logit_staging = self.model.tensor(
            torch.zeros(1, 1, 32, self.model.config.vocab_size, dtype=torch.bfloat16),
            dim=3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.remap_index = self.model.tensor(torch.arange(32, dtype=torch.int32).reshape(1, 32), dtype=ttnn.uint32)
        self.slot_identity = list(range(max_batch_size))
        self.state_scratch = {}
        self.prompt_ends = [0] * max_batch_size
        # perf/multiuser prefix cache: (prefix_len, digest) -> snapshot row, LRU order
        self.snap_lru = OrderedDict()
        self.snap_lock = threading.Lock()  # the vLLM scheduler reads the index (snap_hit_len) from another thread
        self.request_seeds = [None] * 32
        self.seed_values = self._integer(torch.zeros(32, dtype=torch.int32), dtype=ttnn.uint32)
        self.seed_mask = self.model.tensor(torch.ones(1, 32), dtype=ttnn.float32)
        self.model.allocate_prefill_metadata(self.cache, layer_indices=self.layer_page_tables)

    def _event(self, kind, **data):
        if kind == "trace_capture":
            # Logged after each capture.
            print(
                f"NEMOTRON capture {data['family']} trace id={data['trace_id']} " f"seconds={data['capture_s']:.6f}",
                flush=True,
            )

    def _integer(self, value, dtype=ttnn.int32):
        return self.model.tensor(value, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _refresh(self, dst, value, counter):
        host = ttnn.from_torch(value.contiguous(), dtype=dst.dtype, layout=dst.layout)
        ttnn.copy_host_to_device_tensor(host, dst)
        self.counters[counter] += 1

    @staticmethod
    def _normalize_sampling(params):
        defaults = SamplingParams(temperature=1.0, top_k=1, top_p=0.0)
        result = SamplingParams(
            **{
                field.name: (list(value) if isinstance(value, tuple) else value)
                for field in fields(SamplingParams)
                for value in [getattr(params, field.name, getattr(defaults, field.name))]
            }
        )
        if not UNBOUNDED_TOP_K:
            return result

        # perf/multiuser: unset/oversized top_k at T>0 samples from the top-64 candidates on
        # device instead of forcing the whole batch onto host logits. Measured on real GPQA
        # reasoning logits: top-64 mass >= 0.9989 (p01), TV(exact nucleus, top-64 nucleus)
        # mean 6e-5. NEMOTRON_UNBOUNDED_TOP_K=0 restores host routing.
        def fix(k, t):
            k = int(k)
            return UNBOUNDED_TOP_K if (float(t) != 0 and (k < 1 or k > 64)) else k

        k, t = result.top_k, result.temperature
        if isinstance(k, list):
            temps = t if isinstance(t, list) else [t] * len(k)
            k = [fix(a, b) for a, b in zip(k, temps)] + [int(a) for a in k[len(temps) :]]
        elif hasattr(k, "tolist"):
            values = k.tolist()
            if isinstance(values, list):
                temps = t.tolist() if hasattr(t, "tolist") else t
                temps = temps if isinstance(temps, list) else [temps] * len(values)
                k = [fix(a, b) for a, b in zip(values, temps)]
            else:
                k = fix(values, t if not isinstance(t, list) else t[0])
        else:
            k = fix(k, t if not isinstance(t, list) else t[0])
        return replace(result, top_k=k)

    _INACTIVE_ROW = dict(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        seed=None,
        enable_log_probs=False,
        num_logprobs=0,
        log_probs_enable_temperature=False,
    )

    def _pad_rows(self, params):
        """perf/multiuser: the plugin sizes per-row lists to the decode bucket; pad to the slot capacity."""
        if params is None:
            return params
        updates = {}
        for field in fields(SamplingParams):
            value = getattr(params, field.name, None)
            if isinstance(value, (list, tuple)) and len(value) < self.max_batch_size:
                updates[field.name] = list(value) + [self._INACTIVE_ROW[field.name]] * (
                    self.max_batch_size - len(value)
                )
        if not updates:
            return params
        try:
            return replace(params, **updates)
        except TypeError:
            for name, value in updates.items():
                setattr(params, name, value)
            return params

    def _validate_sampling(self, params):
        params = self._normalize_sampling(params)
        if isinstance(params.seed, (list, tuple)) and len(params.seed) not in (self.max_batch_size, 32):
            raise ValueError("Seed lists must describe the fixed-slot capacity")
        for seed in params.seed if isinstance(params.seed, (list, tuple)) else [params.seed]:
            if seed is not None:
                int(seed)
        if torch.any(torch.as_tensor(params.enable_log_probs)):
            raise ValueError("TP4 logprobs use the explicit host-logits compatibility path")
        for name in ("presence_penalty", "frequency_penalty", "repetition_penalty"):
            expected = 1.0 if name == "repetition_penalty" else 0.0
            if torch.any(torch.as_tensor(getattr(params, name)) != expected):
                raise ValueError("Device penalties are disabled; use explicit host-logits compatibility")
        k, temperature = torch.as_tensor(params.top_k), torch.as_tensor(params.temperature)
        if torch.any((temperature != 0) & ((k < 1) | (k > 64))):
            raise ValueError("Native stochastic top_k must be in [1,64]; use host-logits compatibility otherwise")
        formatted = params
        if not isinstance(params.temperature, list):
            formatted = replace(params, temperature=[params.temperature] * self.max_batch_size)
        return format_sampling_params(formatted, 32)

    def _set_sampling(self, params):
        params = self._normalize_sampling(params)
        self.sampling.reset_sampling_params(self._validate_sampling(params))

        # The common formatter deliberately retains its <=32 contract. Wide
        # rows use the original, unclipped parameters and their own trace.
        def padded(value, inactive):
            values = list(value) if isinstance(value, (list, tuple)) else [value] * self.max_batch_size
            return values + [inactive] * (32 - len(values))

        k = padded(params.top_k, 1)
        p = padded(params.top_p, 0.0)
        temperature = padded(params.temperature, 1.0)
        self.wide_sampling.reset(top_k=k, top_p=p, temperature=temperature)
        self._use_wide_sampling = any(t != 0 and n > 32 for n, t in zip(k, temperature, strict=True))
        self._sampler_params = params

    def _merge_sampling_for_slots(self, params, slots):
        params = self._normalize_sampling(params)
        values = {}
        for field in fields(SamplingParams):
            old, new = getattr(self._sampler_params, field.name), getattr(params, field.name)
            old = list(old) if isinstance(old, (list, tuple)) else [old] * self.max_batch_size
            new = list(new) if isinstance(new, (list, tuple)) else [new] * len(slots)
            if len(new) != len(slots):
                raise ValueError("Prefill sampling parameters must describe the submitted rows")
            for slot, value in zip(slots, new):
                old[slot] = value
            values[field.name] = old
        merged = SamplingParams(**values)
        self._validate_sampling(merged)
        return merged

    def _set_sampling_for_slots(self, params, slots):
        self._set_sampling(self._merge_sampling_for_slots(params, slots))

    def _apply_seed_update(self, *, use_positions=False):
        old = ttnn.typecast(
            ttnn.to_layout(ttnn.reshape(self.sampling.tt_sampling.seeds_tt_tensor, (1, 32)), ttnn.TILE_LAYOUT),
            ttnn.float32,
        )
        new = ttnn.typecast(ttnn.to_layout(ttnn.reshape(self.seed_values, (1, 32)), ttnn.TILE_LAYOUT), ttnn.float32)
        if use_positions:
            positions = self.positions[0] if len(self.positions) == 1 else ttnn.concat(self.positions, dim=0)
            positions = ttnn.reshape(positions, (1, len(self.positions)))
            if len(self.positions) < 32:
                positions = ttnn.pad(positions, [(0, 0), (0, 32 - len(self.positions))], value=0)
            positions = ttnn.typecast(ttnn.to_layout(ttnn.reshape(positions, (1, 32)), ttnn.TILE_LAYOUT), ttnn.float32)
            new = ttnn.add(new, ttnn.maximum(positions, 0))
        chosen = ttnn.typecast(ttnn.where(self.seed_mask, new, old), ttnn.uint32)
        chosen = ttnn.reshape(ttnn.to_layout(chosen, ttnn.ROW_MAJOR_LAYOUT), (32,))
        ttnn.copy(chosen, self.sampling.tt_sampling.seeds_tt_tensor)

    def reset_sampling_state(self, seed=None, *, counters=None, slots=None, new_requests=False, use_positions=False):
        slots = list(self.fixed_slots) if slots is None else list(slots)
        values = list(seed) if isinstance(seed, (list, tuple)) else [seed] * self.max_batch_size
        if len(values) not in (self.max_batch_size, 32):
            raise ValueError("Expected one seed or a seed for every fixed slot")
        values += [None] * (32 - len(values))
        counts = [0] * len(slots) if counters is None else list(counters)
        if len(counts) != len(slots) or any(c < 0 or c > self.max_seq_len for c in counts):
            raise ValueError("Invalid sampling counters")
        seeds = torch.zeros(32, dtype=torch.int64)
        mask = torch.zeros(1, 32)
        for slot, count in zip(slots, counts):
            value = values[slot]
            if value is None:
                value = (
                    secrets.randbits(64)
                    if new_requests or self.request_seeds[slot] is None
                    else self.request_seeds[slot]
                )
            self.request_seeds[slot] = int(value)
            seeds[slot] = (_hash_request_seed_to_device_seed(int(value), 0) % 700000) + 1 + int(count)
            mask[0, slot] = 1
        self._refresh(self.seed_values, seeds, "seed_refreshes")
        self._refresh(self.seed_mask, mask, "seed_mask_refreshes")
        self._apply_seed_update(use_positions=use_positions)

    def _reload_seed_parameters(self, old, new):
        def expanded(value):
            return list(value) if isinstance(value, (list, tuple)) else [value] * self.max_batch_size

        before, after = expanded(old.seed), expanded(new.seed)
        changed = [s for s in self.fixed_slots if before[s] != after[s]]
        for slot in changed:
            if after[slot] is None:
                self.request_seeds[slot] = None
        if changed:
            self.reset_sampling_state(after, slots=changed, use_positions=True)

    def _validate_pages(self, page_table, page_tables_per_layer=None):
        if page_table is None:
            candidates = (
                page_tables_per_layer.values()
                if isinstance(page_tables_per_layer, dict)
                else page_tables_per_layer or []
            )
            page_table = next((table for table in candidates if table is not None), self.page_host)
        value = torch.as_tensor(page_table, dtype=torch.int32)
        if value.shape != self.page_host.shape:
            raise ValueError("Page table must retain the configured fixed-slot shape")
        if isinstance(page_tables_per_layer, (list, tuple)):
            if len(page_tables_per_layer) != len(self.layer_page_tables):
                raise ValueError("Wrong number of per-layer page tables")
            page_tables_per_layer = {
                index: value if table is None else table
                for index, table in zip(self.layer_page_tables, page_tables_per_layer)
            }
        if page_tables_per_layer is not None and set(page_tables_per_layer) != set(self.layer_page_tables):
            raise ValueError("Per-layer page tables must cover exactly the attention layers")
        selected_tables = {}
        for index in self.layer_page_tables:
            selected = (
                value
                if page_tables_per_layer is None or page_tables_per_layer[index] is None
                else torch.as_tensor(page_tables_per_layer[index], dtype=torch.int32)
            )
            if selected.shape != value.shape:
                raise ValueError("Per-layer page table shape changed")
            if torch.any(selected < 0) or torch.any(selected >= self.cache.physical_blocks[index]):
                raise ValueError("Page mapping exceeds allocated attention pages")
            selected_tables[index] = selected
        return value, selected_tables

    def _refresh_pages(self, page_table, page_tables_per_layer=None):
        value, selected_tables = self._validate_pages(page_table, page_tables_per_layer)
        for index, selected in selected_tables.items():
            for slot, (row, old, dst) in enumerate(
                zip(selected, self.layer_page_host[index], self.layer_page_tables[index], strict=True)
            ):
                if not torch.equal(row, old):
                    self._refresh(dst, row[None], "page_table_refreshes")
            self.layer_page_host[index] = selected.clone()
        self.page_host = value.clone()

    def _selected_pages(self, slots):
        return {index: [rows[s] for s in slots] for index, rows in self.layer_page_tables.items()}

    def _prefill_pages(self, page_table, per_layer, slots, ends, layout):
        if layout == "slots":
            return self._validate_pages(page_table, per_layer)
        if layout != "rows":
            raise ValueError("page_table_layout must be rows or slots")
        if page_table is None:
            candidates = per_layer.values() if isinstance(per_layer, dict) else per_layer or []
            page_table = next((table for table in candidates if table is not None), self.page_host[list(slots)])
        value = torch.as_tensor(page_table, dtype=torch.int32)

        def scatter(source, destination):
            source = torch.as_tensor(source, dtype=torch.int32)
            if source.ndim != 2 or source.shape[0] < len(slots) or not 0 < source.shape[1]:
                raise ValueError("Execution-row page table has invalid geometry")
            if any(math.ceil(int(end) / 32) > source.shape[1] for end in ends):
                raise ValueError("Execution-row page table does not cover the prompt")
            result = destination.clone()
            source = source[: len(slots), : self.pages_per_slot]
            for row, slot in zip(source, slots):
                result[slot, : row.numel()] = row
                result[slot, row.numel() :] = row[-1]
            return result

        common = scatter(value, self.page_host)
        if isinstance(per_layer, (list, tuple)):
            if len(per_layer) != len(self.layer_page_tables):
                raise ValueError("Wrong number of per-layer page tables")
            per_layer = dict(zip(self.layer_page_tables, per_layer))
        if per_layer is not None and set(per_layer) != set(self.layer_page_tables):
            raise ValueError("Per-layer page tables must cover exactly the attention layers")
        tables = {
            index: scatter(value if per_layer is None or per_layer[index] is None else per_layer[index], old)
            for index, old in self.layer_page_host.items()
        }
        return self._validate_pages(common, tables)

    def reset(self):
        for tensor in self.cache.tensors():
            ttnn.copy(self.zero_templates[(tuple(tensor.shape), tensor.dtype)], tensor)
        for pos in self.positions:
            self._refresh(pos, torch.full((1,), -1, dtype=torch.int32), "position_refreshes")
        self.active_host.zero_()
        self._refresh(self.active, self.active_host.reshape(1, 32), "active_refreshes")
        self.sampling.reset_output_state()
        self.prompt_ends = [0] * self.max_batch_size
        self.slot_identity = [None] * self.max_batch_size
        self.reset_sampling_state(self._sampler_params.seed, new_requests=True)
        self.counters["resets"] += 1

    def _cache(self, kv_cache):
        cache = self.cache if kv_cache is None else kv_cache
        if not isinstance(cache, ModelCache):
            raise TypeError("Pass ModelCache including attention pages and Mamba recurrent/conv slots")
        if cache.context_length != self.max_seq_len or cache.num_slots != self.max_batch_size:
            raise ValueError("Cache geometry must match the configured generator")
        return cache

    def allocate_kv_cache_per_layer(self, per_layer_specs=None):
        """Expose the setup-owned pool without allocating a second cache.

        Supply resolved specs to construction when the scheduler owns the pool
        budget; this method validates that the same plan is being bound.
        """
        if self.decode_traces:
            raise RuntimeError("Allocate and bind cache before generator preparation")
        return self.model.allocate_cache(
            context_length=self.max_seq_len,
            num_slots=self.max_batch_size,
            per_layer_specs=per_layer_specs,
            existing_cache=self.cache,
        )

    def _prepare_state_moves(self, cache):
        # Scratch is allocated before capture. Two passes preserve cycles and
        # duplicates in an absolute source-slot mapping without host state reads.
        for layer_index, states in enumerate(cache.layers):
            for slot, state in enumerate(states):
                for field, tensor in enumerate([*(state.conv or []), state.recurrent]):
                    if tensor is not None:
                        self.state_scratch[layer_index, slot, field] = ttnn.clone(tensor)
        self.position_scratch = [ttnn.clone(x) for x in self.positions]
        self.apply_slot_remap(self.fixed_slots, cache=cache)

    def _remap_vector(self, tensor):
        original_shape = tuple(tensor.shape)
        tiled = ttnn.to_layout(ttnn.reshape(tensor, (1, 32)), ttnn.TILE_LAYOUT)
        moved = ttnn.gather(tiled, 1, self.remap_index)
        if tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
            moved = ttnn.to_layout(moved, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(ttnn.reshape(moved, original_shape), tensor)

    def apply_slot_remap(self, remap, *, cache=None):
        cache = self._cache(cache)
        remap = [int(x) for x in remap]
        if len(remap) != self.max_batch_size or any(x not in self.fixed_slots for x in remap):
            raise ValueError("Remap must describe every fixed slot using valid source slots")
        for layer_index, states in enumerate(cache.layers):
            for destination, source in enumerate(remap):
                for field, tensor in enumerate([*(states[source].conv or []), states[source].recurrent]):
                    if tensor is not None:
                        ttnn.copy(tensor, self.state_scratch[layer_index, destination, field])
            for destination, state in enumerate(states):
                for field, tensor in enumerate([*(state.conv or []), state.recurrent]):
                    if tensor is not None:
                        ttnn.copy(self.state_scratch[layer_index, destination, field], tensor)
        self.model.mamba_remap(cache, remap)
        for destination, source in enumerate(remap):
            ttnn.copy(self.positions[source], self.position_scratch[destination])
        for position, scratch in zip(self.positions, self.position_scratch):
            ttnn.copy(scratch, position)
        full = remap + list(range(self.max_batch_size, 32))
        self._refresh(self.remap_index, torch.tensor([full], dtype=torch.int32), "remap_refreshes")
        for tensor in (self.token_input, self.sampling.tt_sampling.seeds_tt_tensor, self.active):
            self._remap_vector(tensor)
        self.active_host = self.active_host[full]
        updates = {}
        for field in fields(SamplingParams):
            value = getattr(self._sampler_params, field.name)
            if isinstance(value, list) and len(value) >= self.max_batch_size:
                updates[field.name] = [value[s] for s in remap] + value[self.max_batch_size :]
        self._set_sampling(replace(self._sampler_params, **updates))
        self.slot_identity = [self.slot_identity[s] for s in remap]
        self.prompt_ends = [self.prompt_ends[s] for s in remap]
        self.request_seeds = [self.request_seeds[s] for s in full]
        self.counters["slot_remaps"] += 1

    def release_request(self, slot, *, kv_cache=None):
        cache = self._cache(kv_cache)
        if slot not in self.fixed_slots:
            raise ValueError("Invalid request slot")
        for states in cache.layers:
            state = states[slot]
            for tensor in [*(state.conv or []), state.recurrent]:
                if tensor is not None:
                    ttnn.copy(self.zero_templates[(tuple(tensor.shape), tensor.dtype)], tensor)
        self.model.mamba_release(cache, slot)
        self._refresh(self.positions[slot], torch.tensor([-1], dtype=torch.int32), "position_refreshes")
        self.active_host[slot] = 0
        self._refresh(self.active, self.active_host.reshape(1, 32), "active_refreshes")
        self.sampling.reset_output_state(slots=[slot])
        self.slot_identity[slot] = None
        self.request_seeds[slot] = None
        self.prompt_ends[slot] = 0

    # ---- perf/multiuser hybrid prefix cache (Mamba state snapshots at 32-token block boundaries) ----
    @staticmethod
    def _prefix_digest(row, n):
        return hashlib.blake2b(row[:n].numpy().astype("int32").tobytes(), digest_size=16).digest()

    def _snap_lookup(self, row, start):
        """Longest saved Mamba snapshot whose token prefix (length <= start) matches this row."""
        best = None
        with self.snap_lock:
            for key in self.snap_lru:
                n, digest = key
                if n <= start and (best is None or n > best[0]) and self._prefix_digest(row, n) == digest:
                    best = key
            if best is None:
                return 0, None
            self.snap_lru.move_to_end(best)
            return best[0], self.snap_lru[best]

    def snap_hit_len(self, token_ids, n):
        """Longest saved snapshot length <= n whose token prefix matches token_ids (0 if none).

        Called by the vLLM scheduler (perf/patches plugin patch) to clip an attention prefix-cache hit to a point where
        the Mamba state can be restored, so vLLM budgets the re-prefill instead of the model doing it unbudgeted.
        """
        with self.snap_lock:
            lengths = sorted({k[0] for k in self.snap_lru if 0 < k[0] <= n})
            keys = set(self.snap_lru)
        if not lengths:
            return 0
        import numpy as np

        arr = np.asarray(token_ids[: lengths[-1]], dtype=np.int32)
        h, done, best = hashlib.blake2b(digest_size=16), 0, 0
        for m in lengths:
            h.update(arr[done:m].tobytes())
            done = m
            if (m, h.copy().digest()) in keys:
                best = m
        return best

    def _snap_save(self, cache, slot, row, n):
        key = (n, self._prefix_digest(row, n))
        with self.snap_lock:
            if key in self.snap_lru:
                self.snap_lru.move_to_end(key)
                return
            if len(self.snap_lru) < cache.snap[next(iter(cache.snap))].recurrent.shape[0]:
                index = len(self.snap_lru)
            else:
                _, index = self.snap_lru.popitem(last=False)
        self.model.mamba_snapshot(cache, slot, index)
        with self.snap_lock:
            self.snap_lru[key] = index
        self.counters["prefix_snapshots"] += 1

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        return_all_logits=False,
        return_output=True,
        slots=None,
        on_device_sampling=False,
        page_tables_per_layer=None,
        empty_slots=None,
        output_positions=None,
        page_table_layout=None,
        sampling_params=None,
        enable_trace=True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported prefill arguments: {sorted(kwargs)}")
        if not isinstance(return_output, bool):
            raise ValueError("return_output must be a bool")
        if not return_output and (
            return_all_logits or on_device_sampling or sampling_params is not None or output_positions is not None
        ):
            raise ValueError("No-output prefill cannot request logits, output positions, or sampling")
        cache = self._cache(kv_cache)
        if cache is not self.cache:
            raise RuntimeError("Bind external cache with prepare_decode before submitting prefill")
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        if tokens.ndim != 2 or not tokens.shape[0]:
            raise ValueError("Prefill tokens must be a nonempty batch of rows")
        if torch.any(tokens < 0) or torch.any(tokens >= self.model.config.vocab_size):
            raise ValueError("Token outside vocabulary")
        if page_table_layout is None:
            page_table_layout = "rows" if empty_slots is not None else "slots"
        if empty_slots is not None:
            if slots is not None:
                raise ValueError("Use either slots or empty_slots")
            slots = empty_slots
        slots = list(range(tokens.shape[0])) if slots is None else list(slots)
        starts = [0] * len(slots) if start_pos is None else torch.as_tensor(start_pos).tolist()
        prompt_lens = [int(n) for n in prompt_lens]
        if len(slots) != tokens.shape[0] or len(slots) != len(prompt_lens) or len(starts) != len(slots):
            raise ValueError("Row, length, start, and slot metadata must agree")
        if len(set(slots)) != len(slots) or any(s < 0 or s >= cache.num_slots for s in slots):
            raise ValueError("Request slots must be unique valid cache slots")
        if any(not 0 <= a < n <= min(cache.context_length, tokens.shape[1]) for a, n in zip(starts, prompt_lens)):
            raise ValueError("Invalid logical prefill span")
        if return_all_logits and (on_device_sampling or sampling_params is not None):
            raise ValueError("All-logit diagnostics and device sampling are separate modes")
        if output_positions is not None and (
            not return_all_logits
            or len(output_positions) != len(slots)
            or any(not a <= p < n for row, a, n in zip(output_positions, starts, prompt_lens) for p in row)
        ):
            raise ValueError("Output positions must lie within each logical prefill span")
        if sampling_params is not None:
            if not self._prepared:
                raise RuntimeError("Prepare generator before device-prefill sampling")
            self._merge_sampling_for_slots(sampling_params, slots)
        prepared_pages, prepared_layers = self._prefill_pages(
            page_table, page_tables_per_layer, slots, prompt_lens, page_table_layout
        )
        # Request validation is complete; only accepted submissions mutate state.
        self._refresh_pages(prepared_pages, prepared_layers)
        snapshots = cache.snap is not None and not return_all_logits
        for i, (slot, start) in enumerate(zip(slots, starts)):
            if start == 0:
                self.release_request(slot, kv_cache=cache)
            elif snapshots and self.prompt_ends[slot] != start:
                # A new (or resumed) request whose first `start` tokens hit vLLM's attention prefix cache. The Mamba
                # state cannot be rebuilt from KV pages: restore the longest saved snapshot at or before `start` and
                # re-prefill from there (rewrites the already-cached KV pages of [restored, start) with the same
                # values), else from 0.
                restored, row = self._snap_lookup(tokens[i], start)
                self.release_request(slot, kv_cache=cache)
                if row is not None:
                    self.model.mamba_restore(cache, row, slot)
                self.counters["prefix_hits"] += 1
                self.counters["prefix_hit_tokens"] += start
                self.counters["prefix_restored_tokens"] += restored
                starts[i] = restored
        for slot, end in zip(slots, prompt_lens):
            self.prompt_ends[slot] = int(end)
        plans = []
        snap_after = [None] * len(slots)
        for i, (a, n) in enumerate(zip(starts, prompt_lens)):
            cut = n // 32 * 32
            if snapshots and cut > a:
                plan = list(self.model.prefill_plan(a, cut))
                snap_after[i] = (len(plan) - 1, cut)
                if cut < n:
                    plan += list(self.model.prefill_plan(cut, n))
            else:
                plan = list(self.model.prefill_plan(a, n))
            plans.append(plan)
        finals = [None] * len(slots)
        result = None
        if return_all_logits:
            count = (
                max(map(len, output_positions))
                if output_positions is not None
                else max(n - a for a, n in zip(starts, prompt_lens))
            )
            result = torch.zeros(len(slots), count, self.model.config.vocab_size)
        for step in range(max(map(len, plans))):
            indices = [i for i, plan in enumerate(plans) if step < len(plan)]
            spans = [plans[i][step] for i in indices]
            selected_slots = [slots[i] for i in indices]
            rows = [self._integer(tokens[i, a:b][None].int(), ttnn.uint32) for i, (a, b) in zip(indices, spans)]
            metadata = self.model.refresh_prefill_metadata(
                cache=cache, slots=selected_slots, starts=[a for a, _ in spans], page_host=self.layer_page_host
            )
            need_all = return_all_logits and (
                output_positions is None
                or any(any(a <= p < b for p in output_positions[i]) for i, (a, b) in zip(indices, spans))
            )
            output = self.model.prefill(
                rows,
                cache=cache,
                page_tables=self._selected_pages(selected_slots),
                slots=selected_slots,
                starts=[a for a, _ in spans],
                prefill_metadata=metadata,
                last_only=not need_all,
            )
            for i in indices:
                if snap_after[i] is not None and snap_after[i][0] == step:
                    self._snap_save(cache, slots[i], tokens[i], snap_after[i][1])
            for i, (a, b), value in zip(indices, spans, output):
                if return_all_logits:
                    if output_positions is None:
                        result[i, a - starts[i] : b - starts[i]] = self.model.read_logits(value).reshape(
                            -1, self.model.config.vocab_size
                        )
                    else:
                        selected = [(j, p - a) for j, p in enumerate(output_positions[i]) if a <= p < b]
                        if selected:
                            host = self.model.read_logits(value).reshape(-1, self.model.config.vocab_size)
                            result[i, [j for j, _ in selected]] = host[[p for _, p in selected]]
                            del host
                elif return_output and step == len(plans[i]) - 1:
                    finals[i] = value
            del output, rows
        if not return_output:
            return None
        if return_all_logits:
            if on_device_sampling:
                raise ValueError("All-logit diagnostics and device sampling are separate modes")
            return result
        if sampling_params is not None:
            return self.sample_prefill(
                finals, slots=slots, prompt_lens=prompt_lens, sampling_params=sampling_params
            ).reshape(-1, 1)
        if on_device_sampling:
            return finals
        return torch.cat([self.model.read_logits(x).reshape(1, 1, -1) for x in finals], dim=0)

    def _model_step(self, cache, slots):
        packed = ttnn.reshape(self.token_input, (1, 32))
        if self.batched_decode and tuple(slots) == tuple(range(len(slots))):
            rows = len(slots)
            out = self.model.decode_batched(
                packed if rows == 32 else packed[:, :rows],
                cache=cache,
                page_tables=self._selected_pages(slots),
                rows=rows,
                current_positions=[self.positions[s] for s in slots],
                sampler_ready=True,
            )
        else:
            tokens = [packed[:, s : s + 1] for s in slots]
            out = self.model.decode(
                tokens,
                cache=cache,
                page_tables=self._selected_pages(slots),
                slots=slots,
                current_positions=[self.positions[s] for s in slots],
                sampler_ready=True,
            )
        ttnn.copy(out, self.logits)
        for s in slots:
            ttnn.plus_one(self.positions[s], skip_negative_entries=True)
        ttnn.copy(self.active, self.sampler_active)

    def _sample_step(self, *, wide=False):
        previous = ttnn.typecast(
            ttnn.to_layout(ttnn.reshape(self.token_input, (1, 32)), ttnn.TILE_LAYOUT), ttnn.float32
        )
        result = self.sampling._run_sampling(
            self.logits, penalties_on=False, grammar_on=False, tt_out_tok=self.token_input, count_tokens=False
        )
        if wide:
            self.wide_sampling.sample(self.logits, self.sampling.tt_sampling.seeds_tt_tensor, self.wide_output)
            narrow = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(self.token_input, (1, 32)), ttnn.TILE_LAYOUT), ttnn.float32
            )
            wider = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(self.wide_output, (1, 32)), ttnn.TILE_LAYOUT), ttnn.float32
            )
            merged = ttnn.typecast(ttnn.where(self.wide_sampling.wide_mask, wider, narrow), ttnn.uint32)
            merged = ttnn.reshape(ttnn.to_layout(merged, ttnn.ROW_MAJOR_LAYOUT), self.token_input.shape)
            ttnn.copy(merged, self.token_input)
        seeds = ttnn.to_layout(ttnn.reshape(self.sampling.tt_sampling.seeds_tt_tensor, (1, 32)), ttnn.TILE_LAYOUT)
        seeds = ttnn.add(ttnn.typecast(seeds, ttnn.float32), self.sampler_active)
        seeds = ttnn.to_layout(ttnn.typecast(seeds, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(ttnn.reshape(seeds, (32,)), self.sampling.tt_sampling.seeds_tt_tensor)
        selected = ttnn.typecast(
            ttnn.to_layout(ttnn.reshape(self.token_input, (1, 32)), ttnn.TILE_LAYOUT), ttnn.float32
        )
        selected = ttnn.typecast(ttnn.where(self.sampler_active, selected, previous), ttnn.uint32)
        selected = ttnn.reshape(ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, 32))
        ttnn.copy(selected, self.token_input)
        return result

    def prepare_decode(self, *, cache=None, slots=None):
        cache = self._cache(cache)
        slots = self.fixed_slots if slots is None else tuple(slots)
        if slots != self.fixed_slots:
            raise ValueError("Prepare the complete fixed-slot layout")
        key = (id(cache), self.fixed_slots)
        if key in self.decode_traces:
            return
        bucket_slots = [tuple(range(b)) for b in self.decode_buckets]
        if self.decode_traces:
            raise RuntimeError("New cache/slot signatures require generator-wide preparation before capture")
        if cache is not self.cache:
            self.cache = self.model.bind_cache(
                cache,
                context_length=self.max_seq_len,
                num_slots=self.max_batch_size,
                per_layer_specs=[spec.as_tuple() for spec in self.cache.attention_specs],
            )
            self.model.allocate_prefill_metadata(cache, layer_indices=self.layer_page_tables)
        self.model.prepare_prefill(
            cache=cache, page_tables=self.layer_page_tables, page_host=self.layer_page_host
        )
        self._prepare_state_moves(cache)
        self._apply_seed_update(use_positions=True)
        for bslots in bucket_slots:
            self._model_step(cache, bslots)
        first = self.logits[:, :, :1, :]
        padded = ttnn.pad(first, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
        ttnn.copy(padded, self.logits)
        del padded, first
        first = ttnn.clone(self.logits[:, :, :1, :])
        self._stage_prefill_logits([first], [0])
        del first
        for force_argmax in (False, True):
            self.sampling.tt_sampling._force_argmax_sampling = force_argmax
            self._sample_step()
        self.sampling.tt_sampling._force_argmax_sampling = False
        self._sample_step(wide=True)
        self.reset()
        ttnn.synchronize_device(self.mesh)
        self.mesh.set_program_cache_misses_allowed(False)
        try:
            for bslots in bucket_slots:
                capture_start = time.perf_counter()
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._model_step(cache, bslots)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.decode_traces[(id(cache), bslots)] = tid
                self._event(
                    "trace_capture",
                    family="decode",
                    capture_s=time.perf_counter() - capture_start,
                    trace_id=tid,
                    signature={
                        "cache_id": id(cache),
                        "slots": bslots,
                        "cache_specs": [spec.as_tuple() for spec in cache.attention_specs],
                    },
                )
            for mode, force_argmax in (("topk", False), ("argmax", True), ("wide64", False)):
                self.sampling.tt_sampling._force_argmax_sampling = force_argmax
                capture_start = time.perf_counter()
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._sample_step(wide=mode == "wide64")
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.sampling_traces[mode] = tid
                self._event(
                    "trace_capture",
                    family="sampling",
                    capture_s=time.perf_counter() - capture_start,
                    trace_id=tid,
                    signature={"mode": mode, "physical_rows": 32, "vocab_size": self.model.config.vocab_size},
                )
        finally:
            self.mesh.set_program_cache_misses_allowed(True)
        self._set_sampling(self._sampler_params)
        self.counters["captures"] += 3 + len(bucket_slots)
        self._prepared = True

    def _stage_prefill_logits(self, outputs, slots):
        for output, slot in zip(outputs, slots, strict=True):
            row = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.slice_write(
                row, self.prefill_logit_staging, [0, 0, slot, 0], [1, 1, slot + 1, output.shape[-1]], [1, 1, 1, 1]
            )
            del row
            ttnn.deallocate(output)
        tiled = ttnn.to_layout(self.prefill_logit_staging, ttnn.TILE_LAYOUT)
        ttnn.copy(tiled, self.logits)

    def sample_prefill(self, outputs, *, slots, prompt_lens, sampling_params=None, read_tokens=True):
        """Consume final device logits and sample only the admitted fixed slots.

        Input device logits are released after staging. Intermediate prefill
        chunks never call this operation or consume an RNG draw.
        """
        slots = tuple(slots)
        if not self._prepared:
            raise RuntimeError("Prepare generator before submitting device-prefill sampling")
        if len(outputs) != len(slots) or len(prompt_lens) != len(slots) or len(set(slots)) != len(slots):
            raise ValueError("Prefill logits, lengths and unique slots must agree")
        if any(s not in self.fixed_slots for s in slots):
            raise ValueError("Invalid sampling slot")
        if sampling_params is not None:
            self._set_sampling_for_slots(sampling_params, slots)
        self._stage_prefill_logits(outputs, slots)
        mask = torch.zeros(1, 32)
        for slot, length in zip(slots, prompt_lens):
            self._refresh(self.positions[slot], torch.tensor([length], dtype=torch.int32), "position_refreshes")
            self.active_host[slot] = 1
            mask[0, slot] = 1
        self._refresh(self.active, self.active_host.reshape(1, 32), "active_refreshes")
        self._refresh(self.sampler_active, mask, "sampling_mask_refreshes")
        self.reset_sampling_state(self._sampler_params.seed, counters=[int(n) - 1 for n in prompt_lens], slots=slots)
        self._sample()
        return self._read_tokens(self.max_batch_size)[list(slots)] if read_tokens else self.token_input

    def _sample(self):
        mode = (
            "wide64"
            if self._use_wide_sampling
            else "argmax"
            if self.sampling.tt_sampling.force_argmax_sampling
            else "topk"
        )
        ttnn.execute_trace(self.mesh, self.sampling_traces[mode], cq_id=0, blocking=False)
        self.counters["sampling_trace_replays"] += 1
        self.counters[f"sampling_{mode}_replays"] += 1

    def _read_tokens(self, count):
        values = ttnn.to_torch(self.token_input, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, 0))
        self.counters["token_readbacks"] += 1
        return values.reshape(4, 32)[0, :count].long()

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        slots=None,
        on_device_sampling=True,
        refresh_inputs=True,
        read_tokens=True,
        reload_inputs=None,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        sampling_params=None,
        slot_remap=None,
        page_tables_per_layer=None,
        output_token_counts=None,
        read_from_device=None,
        enable_trace=True,
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("Serving decode requires the prepared traces")
        if "reset_batch" in kwargs:
            raise TypeError("Legacy reset_batch is unsupported; use the explicit reload commands")
        if kwargs:
            raise TypeError(f"Unsupported decode arguments: {sorted(kwargs)}")
        if read_from_device is not None:
            read_tokens = bool(read_from_device)
        if reload_inputs is not None:
            refresh_inputs = bool(reload_inputs)
        if slots is not None and tuple(slots) not in [tuple(range(b)) for b in self.decode_buckets]:
            # Run the smallest captured bucket covering every requested slot.
            need = max(slots) + 1
            bucket = next(b for b in self.decode_buckets if b >= need)
        else:
            bucket = len(slots) if slots is not None else self.max_batch_size
        if refresh_inputs and reload_page_table:
            raise ValueError("Full reload and page-only reload are mutually exclusive")
        if reset_sampling_state and not refresh_inputs:
            raise ValueError("Sampler state reset requires a full input reload")
        cache = self._cache(kv_cache)
        slots = self.fixed_slots if slots is None else tuple(slots)
        key = (id(cache), tuple(range(bucket)))
        if key not in self.decode_traces:
            raise RuntimeError("Call prepare_decode before submitting requests")
        if len(set(slots)) != len(slots) or any(s not in self.fixed_slots for s in slots):
            raise ValueError("Invalid decode slots")
        if refresh_inputs:
            tokens = torch.as_tensor(tokens).flatten()
            start_pos = torch.as_tensor(start_pos).flatten()
            if tokens.numel() != len(slots) or start_pos.numel() != len(slots):
                raise ValueError("Tokens, positions and slots must agree")
            if torch.any(tokens < 0) or torch.any(tokens >= self.model.config.vocab_size):
                raise ValueError("Token outside vocabulary")
            if torch.any(start_pos < -1) or torch.any(start_pos >= self.max_seq_len):
                raise ValueError("Decode position outside configured context")
        if slot_remap is not None:
            slot_remap = [int(x) for x in slot_remap]
            if len(slot_remap) != self.max_batch_size or any(x not in self.fixed_slots for x in slot_remap):
                raise ValueError("Invalid fixed-slot remap")
        if reload_sampling_params:
            if sampling_params is None:
                raise ValueError("Parameter reload requires sampling_params")
            sampling_params = self._pad_rows(sampling_params)
            self._validate_sampling(sampling_params)
        if refresh_inputs or reload_page_table:
            self._validate_pages(page_table, page_tables_per_layer)
        if reset_sampling_state and output_token_counts is not None:
            output_token_counts = list(output_token_counts)
            if len(output_token_counts) != len(slots) or any(
                c < 0 or c > self.max_seq_len for c in output_token_counts
            ):
                raise ValueError("Invalid output-token counters")
        # Only after validation may this accepted submission move device state.
        if slot_remap is not None:
            self.apply_slot_remap(slot_remap, cache=cache)
        if reload_sampling_params:
            previous_params = self._sampler_params
            self._set_sampling(sampling_params)
        if reset_sampling_state:
            active_slots = [s for s, p in zip(slots, start_pos) if p >= 0]
            counts = [
                int(output_token_counts[i]) if output_token_counts is not None else int(p)
                for i, (s, p) in enumerate(zip(slots, start_pos))
                if p >= 0
            ]
            self.reset_sampling_state(self._sampler_params.seed, counters=counts, slots=active_slots)
        if refresh_inputs or reload_page_table:
            self._refresh_pages(page_table, page_tables_per_layer)
        if refresh_inputs:
            if len(set(slots)) != len(slots) or any(s not in self.fixed_slots for s in slots):
                raise ValueError("Invalid decode slots")
            self.active_host.zero_()
            packed = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            refreshed_positions = torch.full((self.max_batch_size,), -1, dtype=torch.int32)
            for slot, token, position in zip(
                slots, torch.as_tensor(tokens).flatten(), torch.as_tensor(start_pos).flatten(), strict=True
            ):
                packed[0, 0, 0, slot] = token
                self.active_host[slot] = position >= 0
                refreshed_positions[slot] = position
            # One authoritative write per slot also clears omitted/inactive
            # rows; do not upload -1 and immediately overwrite it for live rows.
            for slot, pos in enumerate(self.positions):
                self._refresh(pos, refreshed_positions[slot : slot + 1], "position_refreshes")
            self._refresh(self.token_input, packed, "token_refreshes")
            self._refresh(self.active, self.active_host.reshape(1, 32), "active_refreshes")
        if reload_sampling_params and not reset_sampling_state:
            self._reload_seed_parameters(previous_params, self._sampler_params)
        key = (id(cache), tuple(range(bucket)))
        if key not in self.decode_traces:
            raise RuntimeError("Call prepare_decode before submitting requests")
        ttnn.execute_trace(self.mesh, self.decode_traces[key], cq_id=0, blocking=False)
        self.counters["model_trace_replays"] += 1
        if not on_device_sampling:
            if read_from_device is False:
                self._device_output_slots[id(self.logits)] = tuple(slots)
                return self.logits
            if not read_tokens:
                return self.logits
            self.counters["host_logits_readbacks"] += 1
            logits = self.model.read_logits(self.logits).reshape(32, -1)
            return logits if tuple(slots) == tuple(range(32)) else logits[list(slots)]
        self._sample()
        if read_from_device is False:
            self._device_output_slots[id(self.token_input)] = tuple(slots)
            return self.token_input
        return self._read_tokens(self.max_batch_size)[list(slots)] if read_tokens else self.token_input

    def _snapshot_output(self, tensor, slots, *, is_tokens):
        self.counters["token_readbacks" if is_tokens else "host_logits_readbacks"] += 1
        snapshot = tensor.cpu(blocking=False)
        return DecodeOutput(snapshot, ttnn.record_event(self.mesh, 0), tuple(slots), is_tokens)

    def read_decode_output(self, handle, *, async_read=False):
        if not isinstance(handle, DecodeOutput):
            slots = self._device_output_slots.get(id(handle), self.fixed_slots)
            handle = self._snapshot_output(handle, slots, is_tokens=handle.dtype == ttnn.uint32)
        if async_read:
            return handle, [handle.event]
        ttnn.event_synchronize(handle.event)
        return self._format_decode_snapshot(handle)

    def _format_decode_snapshot(self, handle):
        """Format a completed host snapshot; the caller owns event waiting."""
        if not handle.is_tokens:
            logits = self.model.read_logits(handle.snapshot).reshape(32, 1, -1)
            return logits if handle.slots == tuple(range(32)) else logits[list(handle.slots)]
        values = ttnn.to_torch(handle.snapshot, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, 0))
        return values.reshape(4, 32)[0, list(handle.slots)].long()

    def process_decode_output_host(self, output, *, is_tokens=False, slots=None):
        slots = self.fixed_slots if slots is None else tuple(slots)
        if isinstance(output, DecodeOutput):
            # The pinned async plugin waits the returned events before this
            # callback. Do not enqueue another read or wait inside formatting.
            result = self._format_decode_snapshot(output)
            return result.reshape(-1, 1) if output.is_tokens else result
        if isinstance(output, torch.Tensor):
            return output.reshape(len(slots), 1, -1) if not is_tokens else output.reshape(-1, 1)
        if is_tokens:
            values = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, 0))
            return values.reshape(4, 32)[0, list(slots)].long().reshape(-1, 1)
        return self.model.read_logits(output).reshape(32, 1, -1)[list(slots)]

    def note_state_slots_moved(self, moves):
        # Device/host ownership already moved together on the accepted decode.
        # This notification must not apply the permutation a second time.
        if any(old not in self.fixed_slots or new not in self.fixed_slots for old, new in moves.items()):
            raise ValueError("Invalid committed slot movement")
        self.counters["slot_move_notifications"] += 1

    def close(self):
        for tid in (*self.decode_traces.values(), *self.sampling_traces.values()):
            ttnn.release_trace(self.mesh, tid)
        self.decode_traces.clear()
        self.sampling_traces.clear()
        self._prepared = False

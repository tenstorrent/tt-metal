# SPDX-License-Identifier: Apache-2.0
"""Readiness generator. Model and sampling traces share persistent device inputs."""
import json
import time
from pathlib import Path

import torch

# Keep the installed readiness ABC identity when available. Standalone serving
# ships the exact tt-model-bringup 0.1.23 contract (Apache-2.0) alongside this file.
try:
    from readiness_check.contract import Generator as GeneratorContract
except ModuleNotFoundError as exc:
    if exc.name not in {"readiness_check", "readiness_check.contract"}:
        raise
    from models.autoports.qwen_qwen3_8_27b.tt.generator_contract import Generator as GeneratorContract
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import Model
from models.common.modules.sampling.sampling_1d import Sampling1D


class Generator(GeneratorContract):
    def __init__(
        self,
        model,
        checkpoint,
        *,
        batch_size=1,
        context_length=None,
        kv_cache=None,
        page_table=None,
        sampling_strategy="topk",
        persistent_sampling_ccl=None,
        pack_prefill_terminal=True,
    ):
        self.model, self.mesh = model, model.mesh
        self.trace_events = []
        self.event_sink = None
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        generation_config = Path(checkpoint) / "generation_config.json"
        eos = (
            json.loads(generation_config.read_text()).get("eos_token_id")
            if generation_config.exists()
            else self.tokenizer.eos_token_id
        )
        self.eos_token_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
        self.batch_size = batch_size
        self.pack_prefill_terminal = pack_prefill_terminal
        if sampling_strategy not in ("topk", "argmax"):
            raise ValueError("Unknown common-sampler strategy")
        self.sampling_strategy = sampling_strategy
        self.context_length = model.config.max_position_embeddings if context_length is None else context_length
        if (
            not isinstance(self.context_length, int)
            or not 1 <= self.context_length <= model.config.max_position_embeddings
        ):
            raise ValueError("Context must lie within the HF advertised capability")
        if not 1 <= batch_size <= 32:
            raise ValueError("Supported batch range is 1..32")
        self.owns_cache = kv_cache is None
        self.kv_cache = (
            model.allocate_cache(batch_size=batch_size, context_length=self.context_length)
            if kv_cache is None
            else kv_cache
        )
        if len(self.kv_cache) != len(model.layers):
            raise ValueError("Cache must provide one state per selected layer")
        self.owns_page_table = not isinstance(page_table, ttnn.Tensor)
        if self.owns_page_table:
            self.page_table_host = (
                torch.arange(batch_size * ((self.context_length + 255) // 256) * 8, dtype=torch.int32).reshape(
                    batch_size, -1
                )
                if page_table is None
                else page_table.clone()
            )
            self.page_table = self.put(self.page_table_host, ttnn.int32)
        else:
            # Binding an existing replicated map is setup-only. Preserve its address;
            # a one-time read establishes the host-update comparison baseline.
            self.page_table = page_table
            replicas = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(page_table)]
            if not replicas or any(not torch.equal(replicas[0], t) for t in replicas[1:]):
                raise ValueError("TP4 page mapping must be replicated across devices")
            self.page_table_host = replicas[0].clone()
        if self.page_table.dtype != ttnn.int32 or self.page_table.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("Page table must be row-major int32")
        if (
            len(self.page_table.shape) != 2
            or self.page_table.shape[0] != batch_size
            or self.page_table.shape[-1] * 32 < self.context_length
        ):
            raise ValueError("Page table must cover every configured slot/context")
        if self.page_table.shape[-1] % 8:
            raise ValueError("Physical page-table rows must contain a multiple of8 int32 entries for SDPA")
        self._page_table_host_valid = True
        self.tokens = self.put(torch.zeros(1, 1, 1, batch_size, dtype=torch.int32), ttnn.uint32)
        self.positions = self.put(torch.zeros(batch_size, dtype=torch.int32), ttnn.int32)
        self.rope_indices = self.put(torch.zeros(batch_size, 1, dtype=torch.int32), ttnn.uint32)
        sampler_ccl = model.layers[0].ccl
        if persistent_sampling_ccl is None:
            persistent_sampling_ccl = "async" if batch_size == 32 else False
        self.sampling_ccl_mode = persistent_sampling_ccl or "baseline"
        if persistent_sampling_ccl:
            from .sampling_ccl import SamplingCCL

            sampler_ccl = SamplingCCL(sampler_ccl, persistent=persistent_sampling_ccl != "async")
        self.sampler = Sampling1D(
            vocab_size=model.config.vocab_size,
            mesh_device=self.mesh,
            tt_ccl=sampler_ccl,
            max_batch_size=32,
            max_top_k=32,
            allow_force_argmax=True,
            pad_to_power_of_2=True,
        )
        self.sampler.load_device_buffers()
        self.k = self.put(torch.ones(batch_size, dtype=torch.int32), ttnn.uint32)
        self.p = self.put(torch.zeros(batch_size), ttnn.bfloat16)
        self.temp = self.put(torch.ones(batch_size), ttnn.bfloat16)
        self.host_sampling_greedy = True
        self.sampling_values = (
            torch.ones(batch_size, dtype=torch.int32),
            torch.zeros(batch_size),
            torch.ones(batch_size),
        )
        self.seed_values = torch.arange(32, dtype=torch.int32) + 1234
        self.seeds = self.put(self.seed_values, ttnn.uint32)
        # Every persistent request buffer exists before the first trace capture.
        self.buckets = {}
        max_bucket = min(4096, ((self.context_length + 31) // 32) * 32)
        for physical in [1, *range(32, max_bucket + 1, 32)]:
            all_width = min(physical, 256)
            self.buckets[physical] = dict(
                tokens=self.put(torch.zeros(batch_size, physical, dtype=torch.int32), ttnn.uint32),
                positions=self.put(torch.zeros(batch_size, physical, dtype=torch.int32), ttnn.uint32),
                valid_mask=model.put(torch.zeros(batch_size, physical, 1)),
                history_indices=self.put(torch.zeros(batch_size, 3, dtype=torch.int32), ttnn.uint32),
                last_indices=self.put(torch.zeros(batch_size, 1, dtype=torch.int32), ttnn.uint32),
                all_indices=self.put(torch.zeros(batch_size, all_width, dtype=torch.int32), ttnn.uint32),
                fill_page_table=self.put(
                    torch.zeros(batch_size, ((physical + 255) // 256) * 8, dtype=torch.int32), ttnn.int32
                ),
                page_indices=self.put(
                    torch.zeros(batch_size, ((physical + 255) // 256) * 8, dtype=torch.int32), ttnn.uint32
                ),
                chunk_start=self.put(torch.zeros(1, dtype=torch.int32), ttnn.int32),
                active_mask=model.put(torch.zeros(batch_size, 1, 1)),
            )
        self.prefill_output = model.put(torch.zeros(batch_size, 1, model.config.vocab_size // 4))
        self.prepared_buckets = set()
        self._warm_layer_indices = None
        self._warming = False
        self.model_trace = self.sample_trace = None
        self.decode_traces = {}
        self.buffered_sample_trace = None
        self.output_window = 128
        self.eos_output_window = 16
        self.output_history = self.put(
            torch.zeros(self.output_window, 1, 1, batch_size, dtype=torch.int32), ttnn.uint32
        )
        self.output_index = self.put(torch.zeros(1, dtype=torch.int32), ttnn.uint32)
        self.active_mask = model.put(torch.ones(batch_size, 1, 1))
        self.logits = model.put(torch.zeros(1, 1, batch_size, model.config.vocab_size // 4))
        # Request-boundary helpers use fixed-shape indices/masks. Their programs
        # are warmed generator-wide before any decode/sampling capture.
        self.state_reset_mask = model.put(torch.zeros(batch_size, 1, 1))
        self.state_remap_indices = {}
        for state in self.kv_cache:
            for name in ("conv", "recurrent"):
                value = getattr(state, name, None)
                if value is not None and name not in self.state_remap_indices:
                    shape = tuple(value.shape)
                    indices = (
                        torch.arange(batch_size, dtype=torch.int32)
                        .reshape(batch_size, *([1] * (len(shape) - 1)))
                        .expand(shape)
                        .contiguous()
                    )
                    self.state_remap_indices[name] = model.put(indices, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
        self.parameter_remap_indices = self.put(torch.arange(batch_size, dtype=torch.int32), ttnn.uint32)
        self.seed_remap_indices = self.put(torch.arange(32, dtype=torch.int32), ttnn.uint32)
        self.prefill_saved_tokens = self.put(torch.zeros(1, 1, 1, batch_size, dtype=torch.int32), ttnn.uint32)
        self.prefill_saved_seeds = self.put(torch.zeros(32, dtype=torch.int32), ttnn.uint32)
        self.prefill_token_select = self.put(
            torch.arange(batch_size, dtype=torch.int32).reshape(1, 1, 1, batch_size), ttnn.uint32
        )
        self.prefill_seed_select = self.put(torch.arange(32, dtype=torch.int32), ttnn.uint32)
        self._resident_inputs_valid = False
        self._resident_active = None
        self.counters = dict(
            model_replays=0,
            sampling_replays=0,
            token_refreshes=0,
            position_refreshes=0,
            page_table_refreshes=0,
            readbacks=0,
            captures=0,
            trace_retirements=0,
            trace_evictions=0,
            trace_invalidations=0,
            host_copies_total=0,
        )

    def put(self, value, dtype):
        return self.model.put(value, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def refresh(self, value, target):
        host = ttnn.from_torch(
            value, dtype=target.dtype, layout=target.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, target)
        self.counters["host_copies_total"] += 1

    def refresh_page_table(self, page_table):
        if not self._page_table_host_valid or not torch.equal(page_table, self.page_table_host):
            if tuple(page_table.shape) != tuple(self.page_table_host.shape):
                raise ValueError("Page-table shape must match the configured pool")
            self.refresh(page_table, self.page_table)
            self.page_table_host = page_table.clone()
            self._page_table_host_valid = True
            self.counters["page_table_refreshes"] += 1

    def _decode(self, masked=False):
        logits = self.model.decode_forward(
            self.tokens,
            positions=self.positions,
            rope_indices=self.rope_indices,
            kv_cache=self.kv_cache,
            page_table=self.page_table,
            batch=self.batch_size,
            active_mask=self.active_mask if masked else None,
        )
        if masked:
            # SDPA intentionally leaves inactive output rows unwritten. Clear
            # those logits before sampling so feedback always remains in-vocab.
            logits = ttnn.where(ttnn.reshape(self.active_mask, [1, 1, self.batch_size, 1]), logits, 0.0)
        ttnn.copy(logits, self.logits)
        ttnn.plus_one(self.positions, skip_negative_entries=True)
        ttnn.plus_one(self.rope_indices)
        return self.logits

    def _sample(self, logits):
        if self.sampling_strategy == "argmax":
            return self.sampler(logits, tt_out_tok=self.tokens)
        result = self.sampler(logits, k=self.k, p=self.p, temp=self.temp, seeds=self.seeds, tt_out_tok=self.tokens)
        ttnn.plus_one(self.seeds)
        return result

    def _sample_buffered(self):
        self._sample(self.logits)
        updated = ttnn.indexed_fill(self.output_index, self.output_history, self.tokens)
        ttnn.copy(updated, self.output_history)
        ttnn.plus_one(self.output_index)

    def set_sampling_params(
        self, *, top_k=1, top_p=0.0, temperature=1.0, seed=1234, reset_sampling_state=True, active_slots=None
    ):
        def vector(value, dtype):
            x = torch.as_tensor(value, dtype=dtype).reshape(-1)
            return x.expand(self.batch_size).contiguous() if x.numel() == 1 else x

        k = vector(top_k, torch.int32)
        p = vector(top_p, torch.float32)
        temp = vector(temperature, torch.float32)
        if any(x.numel() != self.batch_size for x in (k, p, temp)):
            raise ValueError("Sampling parameters require one value per fixed slot")
        if (
            (k < 1).any()
            or (k > 32).any()
            or not torch.isfinite(p).all()
            or not torch.isfinite(temp).all()
            or (p < 0).any()
            or (p > 1).any()
            or (temp <= 0).any()
        ):
            raise ValueError("Require k1..32, finite p0..1 and positive finite temperature")
        if self.sampling_strategy == "argmax" and not bool((k == 1).all()):
            raise ValueError("The force-argmax comparison mode is greedy only")
        active = list(range(self.batch_size)) if active_slots is None else list(active_slots)
        if len(set(active)) != len(active) or any(not 0 <= int(slot) < self.batch_size for slot in active):
            raise ValueError("Invalid sampler update slots")
        if not active:
            return
        preserve_inactive = len(active) != self.batch_size
        if preserve_inactive:
            merged = []
            for previous, incoming in zip(self.sampling_values, (k, p, temp)):
                value = previous.clone()
                value[active] = incoming[active]
                merged.append(value)
            k, p, temp = merged
        self.sampling_values = (k.clone(), p.clone(), temp.clone())
        self.host_sampling_greedy = bool((k == 1).all())
        self.refresh(k, self.k)
        self.refresh(p, self.p)
        self.refresh(temp, self.temp)
        if not reset_sampling_state:
            return
        supplied = vector(seed, torch.int64)
        # UINT32_MAX is manual_seed's skip-initialization sentinel. Reserve enough
        # counter space for the full context, preserving the supplied seed exactly.
        seed_limit = (1 << 32) - 1 - self.context_length
        if supplied.numel() != self.batch_size or (supplied < 0).any() or (supplied > seed_limit).any():
            raise ValueError(f"Seeds require one value per slot in0..{seed_limit}")
        if preserve_inactive:
            ttnn.copy(self.seeds, self.prefill_saved_seeds)
        values = self.seed_values.to(torch.int64) & 0xFFFFFFFF
        values[active] = supplied[active]
        self.seed_values = values.to(torch.int32)  # uint32 backing bits, including the high bit
        self.refresh(self.seed_values, self.seeds)
        if preserve_inactive:
            select = torch.arange(32, dtype=torch.int32)
            select[active] += 32
            self.refresh(select, self.prefill_seed_select)
            pair = ttnn.concat(
                [ttnn.reshape(self.prefill_saved_seeds, [1, 1, 1, 32]), ttnn.reshape(self.seeds, [1, 1, 1, 32])], dim=3
            )
            restored = ttnn.gather(pair, 3, index=ttnn.reshape(self.prefill_seed_select, [1, 1, 1, 32]))
            ttnn.copy(ttnn.reshape(restored, self.seeds.shape), self.seeds)

    def reset_state_slots(self, slots):
        """Clear only recurrent/conv state for incoming rows; never clear the KV pool."""
        slots = list(slots)
        if any(not 0 <= int(slot) < self.batch_size for slot in slots):
            raise ValueError("State reset slot is outside generator capacity")
        if not slots:
            return
        mask = torch.zeros(self.batch_size, 1, 1)
        mask[slots] = 1
        self.refresh(mask, self.state_reset_mask)
        for state in self.kv_cache:
            for name in ("conv", "recurrent"):
                value = getattr(state, name, None)
                if value is None:
                    continue
                tiled = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
                reset_mask = (
                    self.state_reset_mask
                    if len(value.shape) == 3
                    else ttnn.reshape(self.state_reset_mask, [self.batch_size, 1, 1, 1])
                )
                # Scalar where requires matching condition/data dtype; keep
                # recurrent state FP32 rather than narrowing its low bits.
                reset_mask = ttnn.typecast(reset_mask, value.dtype)
                cleared = ttnn.where(reset_mask, 0.0, tiled)
                ttnn.copy(ttnn.to_layout(cleared, value.layout), value)
        self._resident_inputs_valid = False

    def remap_state_slots(self, slot_remap):
        """Gather full-capacity recurrent and sampler state before row ownership moves."""
        mapping = torch.as_tensor(slot_remap, dtype=torch.int32).reshape(-1)
        if mapping.numel() != self.batch_size or (mapping < 0).any() or (mapping >= self.batch_size).any():
            raise ValueError("State remap requires one valid source per configured slot")
        if torch.equal(mapping, torch.arange(self.batch_size, dtype=torch.int32)) and not self._warming:
            return
        for name, indices in self.state_remap_indices.items():
            shape = tuple(indices.shape)
            full = mapping.reshape(self.batch_size, *([1] * (len(shape) - 1))).expand(shape).contiguous()
            self.refresh(full, indices)
        for state in self.kv_cache:
            for name, indices in self.state_remap_indices.items():
                value = getattr(state, name, None)
                if value is not None:
                    gathered = ttnn.gather(ttnn.to_layout(value, ttnn.TILE_LAYOUT), 0, index=indices)
                    ttnn.copy(ttnn.to_layout(gathered, value.layout), value)
        self.refresh(mapping, self.parameter_remap_indices)
        for value in (self.k, self.p, self.temp):
            gathered = ttnn.gather(
                ttnn.reshape(value, [1, 1, 1, self.batch_size]),
                3,
                index=ttnn.reshape(self.parameter_remap_indices, [1, 1, 1, self.batch_size]),
            )
            ttnn.copy(ttnn.reshape(gathered, value.shape), value)
        seed_mapping = torch.arange(32, dtype=torch.int32)
        seed_mapping[: self.batch_size] = mapping
        self.refresh(seed_mapping, self.seed_remap_indices)
        gathered_seeds = ttnn.gather(
            ttnn.reshape(self.seeds, [1, 1, 1, 32]), 3, index=ttnn.reshape(self.seed_remap_indices, [1, 1, 1, 32])
        )
        ttnn.copy(ttnn.reshape(gathered_seeds, self.seeds.shape), self.seeds)
        if not self._warming:
            self.seed_values[: self.batch_size] = self.seed_values[mapping.long()].clone()
            self.sampling_values = tuple(value[mapping.long()].clone() for value in self.sampling_values)
        self._resident_inputs_valid = False

    def sample_prefill(self, logits=None, *, active_slots=None, return_device=True):
        """Use the canonical sampler, preserving inactive rows' token/RNG state."""
        if self.sample_trace is None and not self._warming:
            raise RuntimeError("Prepare the generator before sampling prefill")
        active = list(range(self.batch_size)) if active_slots is None else list(active_slots)
        if any(not 0 <= int(slot) < self.batch_size for slot in active):
            raise ValueError("Sampling slot is outside generator capacity")
        source = self.prefill_output if logits is None else logits
        ttnn.copy(ttnn.reshape(source, self.logits.shape), self.logits)
        ttnn.copy(self.tokens, self.prefill_saved_tokens)
        ttnn.copy(self.seeds, self.prefill_saved_seeds)
        token_select = torch.arange(self.batch_size, dtype=torch.int32)
        token_select[active] += self.batch_size
        seed_select = torch.arange(32, dtype=torch.int32)
        seed_select[active] += 32
        self.refresh(token_select.reshape(1, 1, 1, self.batch_size), self.prefill_token_select)
        self.refresh(seed_select, self.prefill_seed_select)
        if self._warming:
            self._sample(self.logits)
        else:
            ttnn.execute_trace(self.mesh, self.sample_trace, cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
        # Integer gather copies bits and avoids float rounding of high uint32 seeds.
        ttnn.copy(
            ttnn.gather(
                ttnn.concat([self.prefill_saved_tokens, self.tokens], dim=3), 3, index=self.prefill_token_select
            ),
            self.tokens,
        )
        seed_pair = ttnn.concat(
            [ttnn.reshape(self.prefill_saved_seeds, [1, 1, 1, 32]), ttnn.reshape(self.seeds, [1, 1, 1, 32])], dim=3
        )
        gathered_seeds = ttnn.gather(seed_pair, 3, index=ttnn.reshape(self.prefill_seed_select, [1, 1, 1, 32]))
        ttnn.copy(ttnn.reshape(gathered_seeds, self.seeds.shape), self.seeds)
        self._resident_inputs_valid = False
        if return_device:
            return self.tokens
        self.counters["readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).reshape(-1).long()

    def prepare_decode(self, *, prefill_lengths=None):
        if self.model_trace is not None:
            return
        preparation_started = time.perf_counter()
        # External pools may contain live prefixes and sparse (-1) maps. Warm
        # only bounded scratch states: one tensor per field/exact TensorSpec,
        # shared across equal-shaped layers. Keep full physical pool geometry so
        # captured programs still match caller-owned allocations.
        # Even default bound maps can be updated in place before preparation.
        # Always warm through a valid temporary map; only external caches need
        # separate scratch states. Restore the original map before capture.
        original = (self.kv_cache, self.page_table, self.page_table_host, self._page_table_host_valid)
        saved_flags = (self._warming, self._warm_layer_indices, set(self.prepared_buckets))
        saved_request = []
        scratch = []
        scratch_states = []
        capturing = None

        def memory_checkpoint(label):
            view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
            self.last_preparation_memory[label] = dict(
                allocated_bytes_per_device=view.total_bytes_allocated_per_bank * view.num_banks,
                free_bytes_per_device=view.total_bytes_free_per_bank * view.num_banks,
            )

        def restore_bindings():
            self.kv_cache, self.page_table, self.page_table_host, self._page_table_host_valid = original

        def preparation_reset():
            if self.owns_cache:
                self.reset()
                return
            # Deduplicate scratch zeroing locally; public reset remains unchanged.
            from types import SimpleNamespace

            bound = self.kv_cache
            self.kv_cache = [SimpleNamespace(**{str(i): value for i, (_, _, value) in enumerate(scratch)})]
            try:
                self.reset()
            finally:
                self.kv_cache = bound

        try:
            self.last_preparation_memory = {"basis": "DRAM allocator checkpoints, not a peak measurement"}
            memory_checkpoint("before")
            if not self.owns_cache:
                for name in ("tokens", "positions", "rope_indices", "active_mask", "seeds", "logits", "prefill_output"):
                    target = getattr(self, name)
                    saved_request.append((target, ttnn.clone(target)))
                for state in original[0]:
                    fields = {}
                    for name, value in vars(state).items():
                        match = next(
                            (tensor for field, spec, tensor in scratch if field == name and spec == value.spec), None
                        )
                        if match is None:
                            match = ttnn.zeros_like(value)
                            if match.spec != value.spec:
                                raise ValueError("Preparation scratch must preserve the exact external TensorSpec")
                            scratch.append((name, value.spec, match))
                        fields[name] = match
                    scratch_states.append(type(state)(**fields))
            capacities = [
                int(value.shape[0])
                for state in original[0]
                for name, value in vars(state).items()
                if name in ("key", "value")
            ]
            count = self.batch_size * int(original[1].shape[-1])
            capacity = min(capacities) if capacities else count
            if capacity < 1:
                raise ValueError("External page pools must contain at least one physical page")
            valid_map = (torch.arange(count, dtype=torch.int32) % capacity).reshape(tuple(original[1].shape))
            scratch_map = self.put(valid_map, ttnn.int32)
            self.kv_cache = original[0] if self.owns_cache else scratch_states
            self.page_table, self.page_table_host = scratch_map, valid_map
            self._page_table_host_valid = True
            self.last_preparation_memory["scratch_specs"] = [
                dict(
                    field=field,
                    shape=list(tensor.shape),
                    padded_shape=list(tensor.padded_shape),
                    dtype=str(tensor.dtype),
                    layout=str(tensor.layout),
                )
                for field, _, tensor in scratch
            ]
            memory_checkpoint("scratch_bound")
            preparation_reset()
            logits = self._decode()
            self._sample(logits)
            complete_domain = prefill_lengths is None
            lengths = (
                sorted(self.buckets) if complete_domain else sorted({self.physical_length(n) for n in prefill_lengths})
            )
            # Programs depend on layer type/geometry, not weight values. Prepare each
            # layer kind, then warm the complete stack before capture. Tracking remains
            # the acceptance check for any missed persistent allocation.
            if complete_domain:
                seen = set()
                self._warm_layer_indices = []
                for i, layer in enumerate(self.model.layers):
                    if layer.layer_type not in seen:
                        seen.add(layer.layer_type)
                        self._warm_layer_indices.append(i)
            self._warming = True
            for physical in lengths:
                if complete_domain:
                    print(f"PREPARE_PREFILL {physical}", flush=True)
                length = min(physical, self.context_length)
                for all_logits in (True, False):
                    preparation_reset()
                    pref = self.prefill_forward(
                        torch.zeros(self.batch_size, length, dtype=torch.int64),
                        page_table=self.page_table,
                        kv_cache=self.kv_cache,
                        prompt_lens=[length] * self.batch_size,
                        return_all_logits=all_logits,
                        return_device=True,
                    )
                    if not all_logits:
                        pref = ttnn.reshape(pref, [1, 1, self.batch_size, self.model.config.vocab_size // 4])
                        ttnn.copy(pref, logits)
                    del pref
                # Preserve the inherited decoder's128-token continuation contract.
                # Odd128 offsets use its K128 SDPA configuration rather than K256.
                if self.context_length > 128 and physical <= self.physical_length(self.context_length - 128):
                    preparation_reset()
                    length = min(physical, self.context_length - 128)
                    pref = self.prefill_forward(
                        torch.zeros(self.batch_size, length, dtype=torch.int64),
                        page_table=self.page_table,
                        kv_cache=self.kv_cache,
                        prompt_lens=[length] * self.batch_size,
                        start_position=128,
                        return_device=True,
                    )
                    del pref
                self.prepared_buckets.add(physical)
            # Compile every request-boundary helper while scratch cache bindings
            # are active. Runtime index values never enter their program keys.
            self.reset_state_slots(range(self.batch_size))
            self.remap_state_slots(list(range(self.batch_size)))
            self.set_sampling_params(
                top_k=self.sampling_values[0],
                top_p=self.sampling_values[1],
                temperature=self.sampling_values[2],
                seed=(self.seed_values[: self.batch_size].to(torch.int64) & 0xFFFFFFFF),
                active_slots=[0],
            )
            self.sample_prefill(active_slots=list(range(self.batch_size)))
            self._warming = False
            self._warm_layer_indices = None
            preparation_reset()
            # Retain the exact decode output during final warmup to stabilize L1 selection.
            modes = (False, True) if self.batch_size > 1 else (False,)
            for masked in modes:
                self._decode(masked)
            self._sample(self.logits)
            self._sample_buffered()
            ttnn.synchronize_device(self.mesh)
            preparation_reset()
            memory_checkpoint("warmed_before_capture")
            restore_bindings()
            for target, saved in saved_request:
                ttnn.copy(saved, target)
            ttnn.synchronize_device(self.mesh)
            self.trace_events.append(
                dict(
                    kind="eager_preparation",
                    batch=self.batch_size,
                    warmup_ms=1000 * (time.perf_counter() - preparation_started),
                    programs=self.mesh.num_program_cache_entries(),
                )
            )
            # Capture binds original addresses but does not replay cache writes.
            for masked in modes:
                capture_started = time.perf_counter()
                trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                capturing = trace
                self._decode(masked)
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                capturing = None
                self.decode_traces[masked] = trace
                self.trace_events.append(
                    dict(
                        kind="decode",
                        batch=self.batch_size,
                        masked=masked,
                        trace_id=str(trace),
                        capture_ms=1000 * (time.perf_counter() - capture_started),
                    )
                )
            self.model_trace = self.decode_traces[False]
            capture_started = time.perf_counter()
            self.sample_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            capturing = self.sample_trace
            self._sample(self.logits)
            ttnn.end_trace_capture(self.mesh, self.sample_trace, cq_id=0)
            capturing = None
            self.trace_events.append(
                dict(
                    kind="sampling",
                    batch=self.batch_size,
                    buffered=False,
                    trace_id=str(self.sample_trace),
                    capture_ms=1000 * (time.perf_counter() - capture_started),
                )
            )
            capture_started = time.perf_counter()
            self.buffered_sample_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            capturing = self.buffered_sample_trace
            self._sample_buffered()
            ttnn.end_trace_capture(self.mesh, self.buffered_sample_trace, cq_id=0)
            capturing = None
            self.trace_events.append(
                dict(
                    kind="sampling",
                    batch=self.batch_size,
                    buffered=True,
                    trace_id=str(self.buffered_sample_trace),
                    capture_ms=1000 * (time.perf_counter() - capture_started),
                )
            )
            self.counters["captures"] += len(modes) + 2
            if self.owns_cache:
                self.reset()
        except BaseException:
            # end_trace_capture may fail *after* record_end clears the queue's
            # capture ID. Cleanup must tolerate that boundary and retain the
            # original exception while attempting every known trace release.
            traces = set(self.decode_traces.values())
            if self.sample_trace is not None:
                traces.add(self.sample_trace)
            if self.buffered_sample_trace is not None:
                traces.add(self.buffered_sample_trace)
            if capturing is not None:
                traces.add(capturing)
            cleanup_errors = []
            if capturing is not None:
                try:
                    ttnn.end_trace_capture(self.mesh, capturing, cq_id=0)
                except BaseException as cleanup_error:
                    cleanup_errors.append(("end_trace_capture", type(cleanup_error).__name__))
            for trace in traces:
                try:
                    ttnn.release_trace(self.mesh, trace)
                    self.counters["trace_retirements"] += 1
                except BaseException as cleanup_error:
                    cleanup_errors.append(("release_trace", type(cleanup_error).__name__))
            self.model_trace = self.sample_trace = self.buffered_sample_trace = None
            self.decode_traces.clear()
            self.last_preparation_cleanup_errors = cleanup_errors
            self.prepared_buckets = saved_flags[2]
            raise
        finally:
            self._warming = False
            self._warm_layer_indices = None
            restore_bindings()
            self._warming, self._warm_layer_indices = saved_flags[:2]
            for target, saved in saved_request:
                ttnn.copy(saved, target)
            ttnn.synchronize_device(self.mesh)
            # Drop preparation-only references after all queued restoration
            # copies complete. Captures reference only the original buffers.
            scratch.clear()
            scratch_states.clear()
            saved_request.clear()
            fields = match = value = state = tensor = saved = target = scratch_map = None
            memory_checkpoint("after")

    def reset(self):
        self._resident_inputs_valid = False
        # In-place reset preserves cache addresses and existing trace bindings.
        for state in self.kv_cache:
            for value in vars(state).values():
                if value.dtype == ttnn.bfloat4_b:
                    # zeros_like packs BF4 on the host. Same-dtype device fill
                    # overwrites this interleaved paged cache without rebinding
                    # addresses referenced by live traces; reset is CQ-ordered.
                    ttnn.fill(value, 0.0, output_tensor=value)
                else:
                    ttnn.copy(ttnn.zeros_like(value), value)
        self.refresh(torch.zeros(self.batch_size, dtype=torch.int32), self.positions)
        self.refresh(torch.zeros(self.batch_size, 1, dtype=torch.int32), self.rope_indices)
        self.refresh(torch.zeros(1, 1, 1, self.batch_size, dtype=torch.int32), self.tokens)
        self.refresh(torch.ones(self.batch_size, 1, 1), self.active_mask)
        self.refresh(self.seed_values, self.seeds)
        ttnn.copy(ttnn.zeros_like(self.prefill_output), self.prefill_output)
        ttnn.copy(ttnn.zeros_like(self.logits), self.logits)

    def host_logits(self, logits):
        self.counters["readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()

    @staticmethod
    def physical_length(length):
        return 1 if length == 1 else ((length + 31) // 32) * 32

    def _fill_page_window(self, physical, start):
        target = self.buckets[physical]["fill_page_table"]
        indices = self.buckets[physical]["page_indices"]
        first_page = start // 32
        required_pages = (physical + 31) // 32
        page_count = self.page_table.shape[-1]
        if first_page + required_pages > page_count:
            raise ValueError("Page table does not cover the physical prefill chunk")
        # The fill table is rounded to256 tokens, while continuations may begin
        # at128-token offsets. Its unused tail can cross the end of the bound
        # map; repeat the last valid page instead of issuing out-of-bounds gather.
        page_indices = (torch.arange(target.shape[-1], dtype=torch.int32) + first_page).clamp_max(page_count - 1)
        self.refresh(page_indices.expand(self.batch_size, -1).contiguous(), indices)
        # Integer RM gather reads the authoritative device map; runtime indices
        # change offsets without position-specialized slice program signatures.
        ttnn.gather(self.page_table, 1, indices, out=target)

    def _set_bucket(self, physical, tokens, logical_lengths, start, active_rows, start_position=0):
        bucket = self.buckets[physical]
        batch = self.batch_size
        ids = torch.zeros(batch, physical, dtype=torch.int32)
        for row in active_rows:
            n = logical_lengths[row]
            ids[row, :n] = tokens[row, start : start + n].to(torch.int32)
        absolute_start = start_position + start
        positions = (
            torch.arange(absolute_start, absolute_start + physical, dtype=torch.int32).expand(batch, -1).contiguous()
        )
        lengths = torch.tensor(logical_lengths, dtype=torch.int32)
        self.refresh(ids, bucket["tokens"])
        self.refresh(positions, bucket["positions"])
        self.refresh((torch.arange(physical)[None, :] < lengths[:, None]).unsqueeze(-1).float(), bucket["valid_mask"])
        self.refresh(
            torch.arange(batch, dtype=torch.int32)[:, None] * (physical + 3)
            + lengths[:, None]
            + torch.arange(3, dtype=torch.int32),
            bucket["history_indices"],
        )
        self.refresh(
            torch.arange(batch, dtype=torch.int32)[:, None] * physical + (lengths - 1).clamp_min(0)[:, None],
            bucket["last_indices"],
        )
        self._fill_page_window(physical, absolute_start)
        self.refresh(torch.tensor([absolute_start], dtype=torch.int32), bucket["chunk_start"])
        self.refresh((lengths > 0).reshape(batch, 1, 1).float(), bucket["active_mask"])
        # Prefill is eager. Scheduling may skip inactive cache fills; warmup covers
        # every row's fill signature. No such Python scheduling occurs in decode.
        return {**bucket, "active_rows": tuple(active_rows), "start_alignment": absolute_start % 256}

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        return_device=False,
        start_position=0,
        **kwargs,
    ):
        batch = self.batch_size
        self._resident_inputs_valid = False
        if kv_cache is not self.kv_cache:
            raise ValueError("External cache must be bound at construction before preparation")
        if not isinstance(start_position, int) or start_position < 0 or start_position % 128:
            raise ValueError("Continuation start_position must be a nonnegative multiple of 128")
        if len(prompt_lens) != batch or tokens.ndim != 2 or tokens.shape[0] != batch:
            raise ValueError("Provide one prompt length and token row per configured fixed slot")
        if any(n < 0 or n + start_position > self.context_length for n in prompt_lens) or not max(prompt_lens):
            raise ValueError("Invalid logical prompt lengths")
        length = max(prompt_lens)
        if tokens.shape[1] < length:
            raise ValueError("Token rows are shorter than their logical prompt lengths")
        ttnn.copy(ttnn.zeros_like(self.prefill_output), self.prefill_output)
        if isinstance(page_table, torch.Tensor):
            self.refresh_page_table(page_table)
        elif page_table is not self.page_table:
            raise ValueError("Page map must be bound at construction or supplied as a host mapping")
        else:
            self._page_table_host_valid = False
        outputs = (
            (torch.empty(0) if self._warming else torch.zeros(batch, length, self.model.config.vocab_size))
            if return_all_logits
            else None
        )
        # Each group shares its rounded recurrence extent; other slots stay inactive.
        for start in range(0, length, 4096):
            tails = [min(4096, max(0, n - start)) for n in prompt_lens]
            groups = sorted({self.physical_length(n) for n in tails if n})
            for physical in groups:
                active = [i for i, n in enumerate(tails) if n and self.physical_length(n) == physical]
                lens = [n if i in active else 0 for i, n in enumerate(tails)]
                if self.model_trace is not None and physical not in self.prepared_buckets:
                    raise RuntimeError("Physical prefill bucket was not prepared before capture")
                meta = self._set_bucket(physical, tokens, lens, start, active, start_position)
                if self.event_sink is not None and not self._warming:
                    self.event_sink(
                        "prefill_chunk",
                        start=start_position + start,
                        physical=physical,
                        logical_lengths=lens,
                        active_slots=active,
                        batch=batch,
                        programs=self.mesh.num_program_cache_entries(),
                    )
                hidden = self.model.prefill_forward(
                    meta["tokens"],
                    positions=meta["positions"],
                    kv_cache=kv_cache,
                    page_table=self.page_table,
                    batch=batch,
                    length=physical,
                    prepared_metadata=meta,
                    return_hidden=True,
                    warm_layer_indices=self._warm_layer_indices,
                )
                if return_all_logits:
                    width = meta["all_indices"].shape[-1]
                    for offset in range(0, 1) if self._warming else range(0, physical, width):
                        count = min(width, physical - offset)
                        idx = (
                            torch.arange(batch, dtype=torch.int32)[:, None] * physical
                            + (torch.arange(width, dtype=torch.int32) + offset).clamp_max(physical - 1)[None, :]
                        )
                        self.refresh(idx, meta["all_indices"])
                        logits = self.model.terminal(self.model.select_rows(hidden, meta["all_indices"]))
                        if not self._warming:
                            host = self.host_logits(logits)
                            for row in active:
                                take = min(count, max(0, lens[row] - offset))
                                if take:
                                    outputs[row, start + offset : start + offset + take] = host[row, :take]
                            del host
                        del logits
                else:
                    logits = self.model.terminal(
                        self.model.select_rows(hidden, meta["last_indices"]),
                        pack_decode_rows=self.pack_prefill_terminal,
                    )
                    if self.pack_prefill_terminal:
                        logits = ttnn.reshape(logits, [batch, 1, self.model.config.vocab_size // 4])
                    merged = ttnn.where(meta["active_mask"], logits, self.prefill_output)
                    ttnn.copy(merged, self.prefill_output)
                    del logits, merged
                del hidden
        if return_all_logits:
            return outputs
        return self.prefill_output if return_device else self.host_logits(self.prefill_output)

    def prefill_logits(self, prompt_token_ids):
        self.reset()
        tokens = torch.zeros(self.batch_size, len(prompt_token_ids), dtype=torch.int64)
        tokens[0] = torch.tensor(prompt_token_ids)
        logits = self.prefill_forward(
            tokens,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)] + [0] * (self.batch_size - 1),
            return_all_logits=True,
        )
        return logits[:1]

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        active_slots=None,
        return_logits=False,
        return_device=False,
        reload_inputs=True,
        **kwargs,
    ):
        if kv_cache is not self.kv_cache:
            raise ValueError("External cache must be bound at construction before trace preparation")
        if self.model_trace is None:
            raise RuntimeError("Prepare the generator before request prefill")
        if page_table is None:
            pass  # Explicit unchanged-map command; no pool-sized host comparison.
        elif isinstance(page_table, torch.Tensor):
            self.refresh_page_table(page_table)
        elif page_table is not self.page_table:
            raise ValueError("Use the bound page-table tensor or a host mapping update")
        else:
            self._page_table_host_valid = False
        if reload_inputs:
            positions = start_pos.to(torch.int32).reshape(self.batch_size).clone()
            if active_slots is not None:
                active = torch.zeros(self.batch_size, dtype=torch.bool)
                active[list(active_slots)] = True
                positions[~active] = -1
            if (positions < -1).any() or (positions >= self.context_length).any():
                raise ValueError("Decode positions must be inactive(-1) or within configured context")
            active = positions >= 0
            if not active.any():
                self._resident_inputs_valid = False
                if return_logits:
                    if return_device:
                        ttnn.copy(ttnn.zeros_like(self.logits), self.logits)
                        return self.logits
                    return torch.zeros(self.batch_size, self.model.config.vocab_size)
                if return_device:
                    self.refresh(torch.zeros(1, 1, 1, self.batch_size, dtype=torch.int32), self.tokens)
                    return self.tokens
                return torch.zeros(self.batch_size, dtype=torch.int64)
            self.model_trace = self.decode_traces[not bool(active.all())]
            self.refresh(active.reshape(self.batch_size, 1, 1).float(), self.active_mask)
            self.refresh(tokens.reshape(1, 1, 1, self.batch_size).to(torch.int32), self.tokens)
            self.refresh(positions, self.positions)
            self.refresh(positions.clamp_min(0).reshape(self.batch_size, 1), self.rope_indices)
            self.counters["token_refreshes"] += 1
            self.counters["position_refreshes"] += 1
            self._resident_active = active.clone()
            self._resident_inputs_valid = True
        else:
            # Host scheduler inputs deliberately lag by one token during overlap.
            # Never inspect them, even for mask/trace selection, on a resident step.
            if not self._resident_inputs_valid or self._resident_active is None:
                raise RuntimeError("Resident decode requires a preceding input reload")
            if return_logits:
                raise ValueError("Host-sampled/logit decode requires authoritative input reload")
            active = self._resident_active
        ttnn.execute_trace(self.mesh, self.model_trace, cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        if return_logits:
            self._resident_inputs_valid = False
            return self.logits if return_device else self.host_logits(self.logits).reshape(self.batch_size, -1)
        ttnn.execute_trace(self.mesh, self.sample_trace, cq_id=0, blocking=False)
        self.counters["sampling_replays"] += 1
        if return_device:
            return self.tokens
        self.counters["readbacks"] += 1
        output = ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).reshape(-1).long()
        output[~active] = 0
        return output

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=False,
        stop_on_eos=True,
        buffered_output=None,
        **kwargs,
    ):
        if buffered_output is None:
            buffered_output = not host_sampling and next_input is None
        if buffered_output and (host_sampling or next_input is not None):
            raise ValueError("Buffered output requires on-device autoregressive sampling")
        if host_sampling and not self.host_sampling_greedy:
            raise ValueError("Host-sampling compatibility supports greedy top_k=1 only")
        if not enable_trace:
            raise ValueError("Generation requires traced decode")
        if self.model_trace is None:
            self.prepare_decode()
        nested = bool(prompt_token_ids) and isinstance(prompt_token_ids[0], (list, tuple))
        prompts = [list(p) for p in prompt_token_ids] if nested else [list(prompt_token_ids)]
        if not prompts or any(not p for p in prompts) or len(prompts) > self.batch_size:
            raise ValueError("Supply nonempty prompts within the configured fixed slot count")
        if next_input is not None and len(prompts) != 1:
            raise ValueError("The readiness teacher-forcing callback is a single-request contract")
        lengths = [len(p) for p in prompts] + [0] * (self.batch_size - len(prompts))
        if max(lengths) + max_new_tokens > self.context_length:
            raise ValueError("Request exceeds configured context")
        if max_new_tokens <= 0:
            return [[] for _ in prompts] if nested else []
        started = time.monotonic()
        self.reset()
        active = torch.tensor([n > 0 for n in lengths])
        self.refresh(active.reshape(self.batch_size, 1, 1).float(), self.active_mask)
        self.model_trace = self.decode_traces[not bool(active.all())]
        input_ids = torch.zeros(self.batch_size, max(lengths), dtype=torch.int64)
        for i, prompt in enumerate(prompts):
            input_ids[i, : len(prompt)] = torch.tensor(prompt)
        logits = self.prefill_forward(
            input_ids, page_table=self.page_table, kv_cache=self.kv_cache, prompt_lens=lengths, return_device=True
        )
        ttnn.copy(ttnn.reshape(logits, [1, 1, self.batch_size, self.model.config.vocab_size // 4]), self.logits)
        del logits
        if host_sampling:
            sampled = self.host_logits(self.logits).reshape(self.batch_size, -1).argmax(-1)
            self.refresh(sampled.reshape(1, 1, 1, self.batch_size).to(torch.int32), self.tokens)
            self.counters["token_refreshes"] += 1
        else:
            ttnn.execute_trace(self.mesh, self.sample_trace, cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
        self.refresh(torch.tensor([n if n else -1 for n in lengths], dtype=torch.int32), self.positions)
        self.refresh(torch.tensor(lengths, dtype=torch.int32).reshape(self.batch_size, 1), self.rope_indices)
        if buffered_output:
            return self._generate_buffered_tail(prompts, lengths, max_new_tokens, started, nested, stop_on_eos)
        result = [[] for _ in prompts]
        finished = [False for _ in prompts]
        steady_before = dict(self.counters)
        first_token_at = None
        for step in range(max_new_tokens):
            if step:
                ttnn.execute_trace(self.mesh, self.model_trace, cq_id=0, blocking=False)
                self.counters["model_replays"] += 1
                if host_sampling:
                    sampled = self.host_logits(self.logits).reshape(self.batch_size, -1).argmax(-1)
                    self.refresh(sampled.reshape(1, 1, 1, self.batch_size).to(torch.int32), self.tokens)
                    self.counters["token_refreshes"] += 1
                else:
                    ttnn.execute_trace(self.mesh, self.sample_trace, cq_id=0, blocking=False)
                    self.counters["sampling_replays"] += 1
            sampled = ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).reshape(-1).tolist()
            self.counters["readbacks"] += 1
            if first_token_at is None:
                first_token_at = time.monotonic()
            for row in range(len(prompts)):
                if not finished[row]:
                    token = int(sampled[row])
                    result[row].append(token)
                    finished[row] = bool(stop_on_eos and next_input is None and token in self.eos_token_ids)
            # Output-side stopping never reconstructs token feedback or device
            # masks. Finished rows remain fixed device slots until all requests end.
            if all(finished):
                break
            if next_input is not None:
                sampled[0] = next_input(step, int(sampled[0]))
                self.refresh(torch.tensor(sampled, dtype=torch.int32).reshape(1, 1, 1, self.batch_size), self.tokens)
                self.counters["token_refreshes"] += 1
        ended = time.monotonic()
        self.last_generation_metrics = dict(
            source="autoregressive" if next_input is None else "teacher_forcing_token_out",
            input_tokens=lengths[: len(prompts)],
            output_tokens=step + 1,
            output_lengths=[len(r) for r in result],
            batch=self.batch_size,
            concurrency=len(prompts),
            ttft_ms=(first_token_at - started) * 1000,
            decode_seconds=ended - first_token_at,
            decode_tps_per_user=step / (ended - first_token_at) if step > 0 else None,
            steady_state_counters={k: self.counters[k] - steady_before[k] for k in self.counters},
            decode_trace=str(self.model_trace),
            sampling_trace=str(self.sample_trace),
            blocking_replay=False,
            host_sampling=host_sampling,
            stop_on_eos=stop_on_eos,
        )
        return result if nested else result[0]

    def _generate_buffered_tail(self, prompts, lengths, count, started, nested, stop_on_eos):
        # Read the first token for honest caller-visible TTFT, then enqueue each
        # bounded window without host input refresh or per-token readback.
        self.refresh(torch.zeros(1, dtype=torch.int32), self.output_index)
        first = ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).reshape(-1).tolist()
        self.counters["readbacks"] += 1
        first_at = time.monotonic()
        result = [[int(first[row])] for row in range(len(prompts))]
        before = dict(self.counters)
        windows = 0
        scheduled = 1
        finished = [bool(stop_on_eos and row[0] in self.eos_token_ids) for row in result]
        window_steps = self.eos_output_window if stop_on_eos else self.output_window
        for start in range(1, count, window_steps):
            if all(finished):
                break
            steps = min(window_steps, count - start)
            if windows:
                self.refresh(torch.zeros(1, dtype=torch.int32), self.output_index)
            for _ in range(steps):
                ttnn.execute_trace(self.mesh, self.model_trace, cq_id=0, blocking=False)
                ttnn.execute_trace(self.mesh, self.buffered_sample_trace, cq_id=0, blocking=False)
                self.counters["model_replays"] += 1
                self.counters["sampling_replays"] += 1
            # One boundary read collects all outputs, never reconstructs feedback.
            host = ttnn.to_torch(ttnn.get_device_tensors(self.output_history)[0]).reshape(
                self.output_window, self.batch_size
            )
            self.counters["readbacks"] += 1
            for row in range(len(prompts)):
                if finished[row]:
                    continue
                chunk = [int(x) for x in host[:steps, row].tolist()]
                if stop_on_eos:
                    stop = next((i for i, t in enumerate(chunk) if t in self.eos_token_ids), None)
                    if stop is not None:
                        chunk = chunk[: stop + 1]
                        finished[row] = True
                result[row].extend(chunk)
            windows += 1
            scheduled += steps
        ended = time.monotonic()
        emitted = max(map(len, result))
        self.last_generation_metrics = dict(
            source="autoregressive",
            input_tokens=lengths[: len(prompts)],
            output_tokens=emitted,
            scheduled_output_tokens=scheduled,
            requested_output_tokens=count,
            output_lengths=[len(r) for r in result],
            batch=self.batch_size,
            concurrency=len(prompts),
            ttft_ms=(first_at - started) * 1000,
            decode_seconds=ended - first_at,
            decode_tps_per_user=(emitted - 1) / (ended - first_at) if emitted > 1 else None,
            steady_state_counters={k: self.counters[k] - before[k] for k in self.counters},
            inner_loop_host_refreshes=0,
            inner_loop_readbacks=0,
            output_windows=windows,
            output_window_steps=window_steps,
            decode_trace=str(self.model_trace),
            sampling_trace=str(self.buffered_sample_trace),
            blocking_replay=False,
            host_sampling=False,
            stop_on_eos=stop_on_eos,
            buffered_output=True,
            eos_policy="trim at first EOS; finish current bounded output window before stopping device work",
        )
        return result if nested else result[0]


def build_generator(model_dir, mesh_device, **kwargs):
    checkpoint = str(kwargs.pop("checkpoint", Path(".cache/qwen38-full")))
    indices = kwargs.pop("layer_indices", None)
    prepare = kwargs.pop("prepare", True)
    precision_config = kwargs.pop("precision_config", None)
    model = Model(checkpoint, mesh_device, layer_indices=indices, precision_config=precision_config)
    generator = Generator(model, checkpoint, **kwargs)
    if prepare:
        generator.prepare_decode()
    return generator

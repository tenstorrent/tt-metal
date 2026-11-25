# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Readiness and low-level request API for the TP4 full model.

Token feedback and position advance belong to device traces. Host logits are
available only at an explicit diagnostic/host-sampling boundary.
"""

import json
import os
import secrets
import time
from collections import Counter
from types import SimpleNamespace

import torch
from .readiness_contract import Generator
from transformers import AutoTokenizer

import ttnn
from agentic_models.models.qwen3_6_27b.vllm.tt.model import Model
from agentic_models.models.qwen3_6_27b.vllm.tt.multichip_decoder import PagedState, RecurrentState
from agentic_models.models.qwen3_6_27b.vllm.tt.page_table_ownership import CertifiedPagedState, page_table_for_attention
from agentic_models.models.qwen3_6_27b.vllm.tt.sampling_history import PenaltyHistory
from .sampling.generator import SamplingGenerator, SamplingParams


class SamplingCCL:
    def __init__(self, model):
        self.model = model

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return self.model.io.ccl.get_barrier_semaphore()

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        return self.model.io.ccl.get_ag_ping_pong_semaphore()

    def get_num_links(self, cluster_axis=None):
        return ttnn.get_num_links(self.model.io.mesh, 1)

    def line_all_gather(self, tensor, *, dim, cluster_axis, memory_config, num_links, buffer_key=None):
        if dim != 3 or cluster_axis not in (None, 1):
            raise ValueError("TP4 vocabulary gather must use the column mesh axis")
        value = self.model.io._collective(ttnn.reshape(tensor, tuple(tensor.shape)[1:]))
        # TTSampling owns and explicitly deallocates its gathered outputs.
        # The decoder's persistent collective buffer must retain its storage.
        return ttnn.clone(ttnn.reshape(value, (1, *tuple(value.shape))))


class QwenGenerator(Generator):
    """Own the entire bound cache pool exclusively until close.

    External cache storage is supported, but sharing its physical pages with an
    independently live generator is not: prepare/reset clear the complete pool.
    """

    decode_input_update_contract = 1

    def __init__(
        self,
        model,
        *,
        batch_size=1,
        max_seq_len=None,
        kv_cache=None,
        page_table=None,
        trace_prefill=True,
        cache_sampling_params=True,
        decode_batch_sizes=None,
        batched_prefill=True,
        spec_width=0,
    ):
        self.model, self.mesh = model, model.mesh
        self.batch_size = batch_size
        if not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be in 1..32")
        self.decode_batch_sizes = tuple(sorted(set(decode_batch_sizes or [batch_size])))
        if batch_size not in self.decode_batch_sizes or any(
            not 1 <= size <= batch_size for size in self.decode_batch_sizes
        ):
            raise ValueError("Decode buckets must be prefixes of the configured slot capacity")
        self.active_decode_batch = batch_size
        self.batched_prefill = batched_prefill
        self.max_seq_len = model.precision["supported_context"] if max_seq_len is None else max_seq_len
        if not 1 <= self.max_seq_len <= model.config.max_position_embeddings:
            raise ValueError("Invalid context allocation")
        pages = max(16, ((self.max_seq_len + 511) // 512) * 16)
        self.page_host = (
            torch.arange(batch_size * pages, dtype=torch.int32).reshape(batch_size, pages)
            if page_table is None
            else page_table.to(torch.int32).clone()
        )
        self.cache = model.create_cache(self.page_host) if kv_cache is None else kv_cache
        if len(self.cache) != len(model.layers):
            raise ValueError("One cache/state per decoder layer is required")
        if tuple(self.page_host.shape) != (batch_size, pages):
            raise ValueError("External page table must match the configured batch and rounded context")
        for layer, state in zip(model.layers, self.cache):
            if layer.kind == "linear_attention":
                if (
                    not isinstance(state, RecurrentState)
                    or tuple(state.conv.shape) != (batch_size, 3, 2560)
                    or tuple(state.recurrent.shape) != (batch_size, 12, 128, 128)
                    or state.conv.dtype != getattr(ttnn, model.precision["activation_dtype"])
                    or state.recurrent.dtype != getattr(ttnn, model.precision["recurrent_state_dtype"])
                ):
                    raise ValueError("External GDN cache must preserve the accepted BF16 state contract")
            elif (
                not isinstance(state, (PagedState, CertifiedPagedState))
                or tuple(state.key.shape)[1:] != (1, 32, 256)
                or tuple(state.value.shape) != tuple(state.key.shape)
                or state.key.dtype != getattr(ttnn, layer.policy["kv_dtype"])
                or state.value.dtype != getattr(ttnn, layer.policy["kv_dtype"])
            ):
                raise ValueError("External KV cache must preserve page32/local-KV1/head256 and selected dtype")
        for state in self.cache:
            fields = (
                ((state.conv, ttnn.ROW_MAJOR_LAYOUT), (state.recurrent, ttnn.TILE_LAYOUT))
                if isinstance(state, RecurrentState)
                else ((state.key, ttnn.TILE_LAYOUT), (state.value, ttnn.TILE_LAYOUT))
            )
            if any(
                tensor.device() != self.mesh
                or tensor.layout != layout
                or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                for tensor, layout in fields
            ):
                raise ValueError("External cache must retain the accepted layouts and DRAM storage on this mesh")
            if not isinstance(state, RecurrentState):
                table = page_table_for_attention(state)
                if (
                    tuple(table.shape) != tuple(self.page_host.shape)
                    or table.dtype != ttnn.int32
                    or table.layout != ttnn.ROW_MAJOR_LAYOUT
                    or table.device() != self.mesh
                    or table.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                ):
                    raise ValueError(
                        "External cache page tables require configured int32 row-major DRAM storage on this mesh"
                    )
        if int(self.page_host.min()) < 0 or any(
            int(self.page_host.max()) >= state.key.shape[0]
            for state in self.cache
            if not isinstance(state, RecurrentState)
        ):
            raise ValueError("Initial page address exceeds the cache allocation")
        # perf/spec: speculative verify of up to spec_width tokens per row (1 + drafts).
        self.spec_width = int(spec_width)
        if self.spec_width:
            if batch_size != 32 or not 2 <= self.spec_width <= 4:
                raise ValueError("Speculative verify needs the 32-slot generator and width 2..4")
            # Slot j only ever holds rows of a verify with width > j (front rows b < B*W <= 32):
            # size it to the largest such bucket (16 rows for slot 1, 8 for slots 2-3 by default).
            sizes = sorted(set(decode_batch_sizes or [batch_size]))
            shapes = [(b, w) for b in sizes for w in range(2, self.spec_width + 1) if b * w <= 32]
            self.spec_slot_rows = [max(b for b, w in shapes if w > j) for j in range(1, self.spec_width)]
            if os.environ.get("QWEN_PERF_FULL_SPEC_SLOTS") == "1":
                self.spec_slot_rows = [batch_size] * (self.spec_width - 1)
            for layer, state in zip(model.layers, self.cache):
                if isinstance(state, RecurrentState):
                    layer.create_spec_slots(state, self.spec_width, self.spec_slot_rows)
        self.reset_zeros = {}
        self.reset_pairs = []
        for state in self.cache:
            tensors = (state.conv, state.recurrent) if isinstance(state, RecurrentState) else (state.key, state.value)
            if isinstance(state, RecurrentState):
                tensors = tensors + tuple(t for pair in getattr(state, "spec_slots", []) for t in pair)
            for tensor in tensors:
                signature = (tuple(tensor.shape), str(tensor.dtype), str(tensor.layout))
                if signature not in self.reset_zeros:
                    # zeros_like on RM tensors uploads host storage: setup only.
                    self.reset_zeros[signature] = ttnn.zeros_like(tensor)
                self.reset_pairs.append((self.reset_zeros[signature], tensor))
        self.tokenizer = AutoTokenizer.from_pretrained(model.checkpoint, local_files_only=True)
        self.tokens = self._upload(torch.zeros((1, 1, 1, 32), dtype=torch.int32), ttnn.uint32)
        self.positions = self._upload(torch.zeros(batch_size, dtype=torch.int32), ttnn.int32)
        self.rope_positions = self._upload(torch.zeros(batch_size, dtype=torch.int32), ttnn.uint32)
        self.active_rows = self._upload(torch.ones(batch_size, dtype=torch.bfloat16), ttnn.bfloat16)
        self.sample_active_rows = model.io._upload(
            torch.nn.functional.pad(torch.ones(1, batch_size), (0, 32 - batch_size)), ttnn.bfloat16
        )
        self.single_slot_active = True
        self.logits = model.io._upload(
            torch.zeros(1, 1, 32, model.config.vocab_size // 4), getattr(ttnn, model.precision["logits_dtype"])
        )
        args = SimpleNamespace(
            vocab_size=model.config.vocab_size,
            padded_vocab_size=model.config.vocab_size,
            max_batch_size=batch_size,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            pad_logits_to_power_of_2=True,
        )
        self.sampling = SamplingGenerator(args=args, mesh_device=self.mesh, tt_ccl=SamplingCCL(model))
        self.sampling.reset_sampling_params(
            SamplingParams(
                temperature=[1.0] * 32,
                top_k=[1] * 32,
                top_p=[0.0] * 32,
                presence_penalty=[0.0] * 32,
                frequency_penalty=[0.0] * 32,
                repetition_penalty=[1.0] * 32,
            )
        )
        self.counters = Counter()
        self.full_page_tables_dirty = False
        # Prefix decode buckets (perf/multiuser: any size < capacity, not only 1).
        # Each borrows the scheduler-owned K/V and full recurrent buffers; only a
        # b-row page table is separate. It is refreshed when that bucket runs.
        self.bucket_page_tables = {}
        self.bucket_caches = {}
        for size in self.decode_batch_sizes:
            if size == batch_size:
                continue
            self.bucket_page_tables[size] = self._upload(self.page_host[:size], ttnn.int32)
            self.bucket_caches[size] = [
                state
                if isinstance(state, RecurrentState)
                else PagedState(state.key, state.value, self.bucket_page_tables[size])
                for state in self.cache
            ]
        if self.spec_width:
            self._spec_init()
        self.single_decode_page_table = self.bucket_page_tables.get(1)
        self.single_decode_cache = self.bucket_caches.get(1)
        self.cache_sampling_params = cache_sampling_params
        self._sampling_signature = None
        if kv_cache is not None:
            # Equal host metadata does not establish the external device table's
            # contents. Install once into its existing storage before preparation.
            self._bind_pages(self.page_host, self.cache, force=True)
        self.traces = {}
        self.prepared = False
        self.last_timings = {}
        self.request_seeds = [None] * batch_size
        self.sampling_values = dict(
            temperature=[1.0] * batch_size,
            top_k=[1] * batch_size,
            top_p=[0.0] * batch_size,
            presence_penalty=[0.0] * batch_size,
            frequency_penalty=[0.0] * batch_size,
            repetition_penalty=[1.0] * batch_size,
        )
        self.decode_active_slots = list(range(batch_size))
        self.penalty_history_slots = set()
        self.penalty_history = PenaltyHistory(self)
        # slice_write operates on row-major storage; never rebind traced logits.
        self.prefill_sample_logits = self._upload(
            torch.zeros(1, 1, 32, model.config.vocab_size // 4), getattr(ttnn, model.precision["sampling_dtype"])
        )
        self.prefill_previous_sample_mask = ttnn.clone(self.sample_active_rows)
        # Runtime index contents may change; their shapes and storage cannot.
        self.remap_indices = {}
        for state in self.cache:
            if isinstance(state, RecurrentState):
                for tensor in (state.conv, state.recurrent):
                    shape = tuple(tensor.shape)
                    key = (shape, tensor.layout)
                    if key not in self.remap_indices:
                        indices = (
                            torch.arange(batch_size, dtype=torch.int32)
                            .reshape((batch_size,) + (1,) * (len(shape) - 1))
                            .expand(shape)
                            .contiguous()
                        )
                        self.remap_indices[key] = model.io._upload(indices, ttnn.uint32, tensor.layout)
        self.remap_vector = model.io._upload(
            torch.arange(32, dtype=torch.int32).reshape(1, 32), ttnn.uint32, ttnn.TILE_LAYOUT
        )
        self.row_move_map = self.row_move_self = None
        if batch_size == 32 and os.environ.get("QWEN_PERF_ROW_MOVE_REMAP", "1") == "1":
            self.row_move_map = self._upload(torch.full((32,), -1, dtype=torch.int32), ttnn.int32)
            self.row_move_self = self._upload(torch.full((32,), -1, dtype=torch.int32), ttnn.int32)
            self.row_move_temp = {}
            self.row_move_maps = {}
            self.row_move_consolidate = self._upload(torch.full((32,), -1, dtype=torch.int32), ttnn.int32)
            for state in self.cache:
                if isinstance(state, RecurrentState):
                    slots = tuple(t for pair in getattr(state, "spec_slots", []) for t in pair)
                    for tensor in (state.conv, state.recurrent) + slots:
                        key = (tuple(tensor.shape), str(tensor.dtype), str(tensor.layout))
                        if key not in self.row_move_temp:
                            self.row_move_temp[key] = ttnn.clone(tensor)
                        rows = tensor.shape[0]
                        if rows != batch_size and rows not in self.row_move_maps:
                            self.row_move_maps[rows] = tuple(
                                self._upload(torch.full((32,), -1, dtype=torch.int32), ttnn.int32) for _ in range(2)
                            )
        self.prefill_single = model.allocate_prefill(self.cache, self.page_host)
        self.prefill_uniform = (
            self.prefill_single
            if batch_size == 1 or not batched_prefill
            else model.allocate_prefill(self.cache, self.page_host, prefill_batch_size=batch_size)
        )
        from agentic_models.models.qwen3_6_27b.vllm.tt.prefill_trace import FirstChunkPrefillTrace

        self.prefill_trace = (
            FirstChunkPrefillTrace(self) if trace_prefill and batch_size == 1 and self.max_seq_len >= 128 else None
        )
        if self.spec_width and getattr(self.model, "mtp", None) is not None:
            self._mtp_init()

    def _remap_graph(self):
        if getattr(self, "row_move_map", None) is not None:
            # perf: page-granular copy of only the remapped rows through a preallocated temp
            # (ttnn.gather moved all 32 rows of every GDN state/candidate slot: ~0.3-1 s per remap).
            from .row_move import move_rows, state_geometry

            for state in self.cache:
                if isinstance(state, RecurrentState):
                    slots = tuple(t for pair in getattr(state, "spec_slots", []) for t in pair)
                    for tensor in (state.conv, state.recurrent) + slots:
                        key = (tuple(tensor.shape), str(tensor.dtype), str(tensor.layout))
                        temp = self.row_move_temp[key]
                        page, ppr = state_geometry(tensor)
                        rows = tensor.shape[0]
                        fwd, back = (
                            (self.row_move_map, self.row_move_self)
                            if rows == self.batch_size
                            else self.row_move_maps[rows]
                        )
                        move_rows(tensor, temp, fwd, rows=rows, page_size=page, pages_per_row=ppr)
                        move_rows(temp, tensor, back, rows=rows, page_size=page, pages_per_row=ppr)
        else:
            for state in self.cache:
                if isinstance(state, RecurrentState):
                    slots = tuple(t for pair in getattr(state, "spec_slots", []) for t in pair)
                    for tensor in (state.conv, state.recurrent) + slots:
                        moved = ttnn.gather(tensor, 0, self.remap_indices[(tuple(tensor.shape), tensor.layout)])
                        ttnn.copy(moved, tensor)
        sampler = self.sampling.tt_sampling
        vectors = (self.tokens, sampler.k_tensor, sampler.p_tensor, sampler.temp_tensor, sampler.seeds_tt_tensor)
        for tensor in vectors:
            tiled = ttnn.to_layout(ttnn.reshape(tensor, (1, 32)), ttnn.TILE_LAYOUT)
            moved = ttnn.gather(tiled, 1, self.remap_vector)
            ttnn.copy(ttnn.reshape(ttnn.to_layout(moved, tensor.layout), tensor.shape), tensor)
        # Greedy tie-breaking is slot-indexed too, unlike hardware RNG core IDs.
        greedy = ttnn.reshape(sampler._greedy_col, (1, 32))
        moved = ttnn.gather(greedy, 1, self.remap_vector)
        ttnn.copy(ttnn.reshape(moved, sampler._greedy_col.shape), sampler._greedy_col)

    def _consolidate_spec_rows(self, remap, accepted_counts):
        """A row whose pending candidate slot cannot follow it to its new row (slot sized below the new row):
        copy that slot row into the base state before the remap and treat the row as having committed 1."""
        from .row_move import move_rows, state_geometry

        counts = torch.as_tensor(accepted_counts, dtype=torch.int32).flatten().clone()
        todo = {}
        for r in range(min(len(counts), self.batch_size)):
            j = int(counts[r]) - 1
            if j >= 1 and r >= self.spec_slot_rows[j - 1]:
                todo.setdefault(j, []).append(int(remap[r]))
                counts[r] = 1
        for j, sources in todo.items():
            mapping = torch.full((32,), -1, dtype=torch.int32)
            for src in sources:
                mapping[src] = src
            self._write(mapping, self.row_move_consolidate)
            for state in self.cache:
                if isinstance(state, RecurrentState):
                    for slot_t, base_t in zip(state.spec_slots[j - 1], (state.conv, state.recurrent)):
                        page, ppr = state_geometry(base_t)
                        move_rows(
                            slot_t,
                            base_t,
                            self.row_move_consolidate,
                            rows=slot_t.shape[0],
                            page_size=page,
                            pages_per_row=ppr,
                        )
            self.counters["spec_consolidations"] += len(sources)
        return counts

    def remap_slots(self, slot_remap, accepted_counts=None):
        remap = torch.as_tensor(slot_remap, dtype=torch.int32).flatten()
        if remap.numel() != self.batch_size or sorted(remap.tolist()) != list(range(self.batch_size)):
            raise ValueError("slot_remap must be a permutation of the configured slot capacity")
        if torch.equal(remap, torch.arange(self.batch_size, dtype=torch.int32)):
            return accepted_counts
        if accepted_counts is not None and self.spec_width and getattr(self, "row_move_map", None) is not None:
            accepted_counts = self._consolidate_spec_rows(remap, accepted_counts)
        if getattr(self, "row_move_map", None) is not None:
            moved = torch.full((32,), -1, dtype=torch.int32)
            changed = remap != torch.arange(self.batch_size, dtype=torch.int32)
            moved[: self.batch_size] = torch.where(changed, remap, -1)
            self._write(moved, self.row_move_map)
            self._write(torch.where(moved >= 0, torch.arange(32, dtype=torch.int32), -1), self.row_move_self)
            for rows, (fwd, back) in self.row_move_maps.items():
                part = moved.clone()
                part[rows:] = -1
                part[part >= rows] = -1
                self._write(part, fwd)
                self._write(torch.where(part >= 0, torch.arange(32, dtype=torch.int32), -1), back)
        else:
            for (shape, _), target in self.remap_indices.items():
                self._write(remap.reshape((self.batch_size,) + (1,) * (len(shape) - 1)).expand(shape), target)
        padded = torch.arange(32, dtype=torch.int32)
        padded[: self.batch_size] = remap
        self._write(padded.reshape(1, 32), self.remap_vector)
        self._remap_graph()
        if self.penalty_history_slots:
            self.penalty_history.remap(padded)
            self.penalty_history_slots = {
                i for i, old in enumerate(padded.tolist()) if old in self.penalty_history_slots
            }
        # Device parameters were permuted independently of the last upload.
        self._sampling_signature = None
        self.request_seeds = [self.request_seeds[index] for index in remap.tolist()]
        self.sampling_values = {
            name: [values[index] for index in remap.tolist()] for name, values in self.sampling_values.items()
        }
        self.counters["slot_remaps"] += 1
        return accepted_counts

    def _upload(self, value, dtype):
        return self.model.io._upload(value, dtype, ttnn.ROW_MAJOR_LAYOUT)

    def _write(self, value, target, counter=None):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=target.dtype,
            layout=target.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, target)
        if counter:
            self.counters[counter] += 1

    def _read_logits(self, tensor):
        self.counters["full_logits_readbacks"] += 1
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))

    def _read_tokens(self):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).flatten().tolist()[: self.batch_size]

    def _decode_trace_key(self, batch):
        return "decode" if batch == self.batch_size else f"decode_b{batch}"

    def _decode_graph(self, logical_batch=None):
        batch = self.batch_size if logical_batch is None else logical_batch
        prefix = batch != self.batch_size
        # Positions are read from the device-owned full-slot vectors inside
        # the trace. No host position feedback is introduced by the B1 bucket.
        positions = self.positions[:batch] if prefix else self.positions
        rope_positions = self.rope_positions[:batch] if prefix else self.rope_positions
        value = self.model.decode(
            self.tokens,
            positions,
            cache=self.bucket_caches[batch] if prefix else self.cache,
            batch=batch,
            rope_positions=rope_positions,
            active_rows=self.active_rows if batch > 1 or self.spec_width else None,
            live_positions=self.positions if self.batch_size == 32 else None,
            **(self._spec_args(batch, 1) if self.spec_width else {}),
        )
        ttnn.copy(value, self.logits)
        ttnn.plus_one(self.positions, skip_negative_entries=True)
        if self.batch_size > 1:
            old = ttnn.clone(self.rope_positions)
            ttnn.plus_one(self.rope_positions)
            mask = ttnn.to_layout(ttnn.reshape(self.active_rows, (1, self.batch_size)), ttnn.TILE_LAYOUT)
            self._restore_inactive_vector(self.rope_positions, old, mask)
        else:
            ttnn.plus_one(self.rope_positions)

    # ---------------------------------------------------------------- perf/spec
    def _spec_init(self):
        zeros = torch.zeros(32, dtype=torch.int32)
        self.spec_select = self._upload(zeros, ttnn.int32)
        self.spec_zero = self._upload(zeros, ttnn.int32)
        self.spec_select_host = zeros.clone()
        self.vtokens = self._upload(torch.zeros((1, 1, 1, 32), dtype=torch.int32), ttnn.uint32)
        self.vpositions = self._upload(torch.full((32,), -1, dtype=torch.int32), ttnn.int32)
        self.vrope = self._upload(torch.zeros(32, dtype=torch.int32), ttnn.uint32)
        self.vupdate = [
            self._upload(torch.full((32,), -1, dtype=torch.int32), ttnn.int32) for _ in range(self.spec_width)
        ]
        # Post-norm target hidden of the last decode/verify rows (MTP drafter input).
        self.spec_hidden = self.model.io._upload(torch.zeros(1, 32, self.model.config.hidden_size))
        self.verify_shapes = [
            (b, w) for b in self.decode_batch_sizes for w in range(2, self.spec_width + 1) if b * w <= 32
        ]
        self.verify_tables, self.verify_caches, self.verify_table_host = {}, {}, {}
        for b, w in self.verify_shapes:
            host = self.page_host[:b].repeat(w, 1)
            self.verify_table_host[(b, w)] = host.clone()
            self.verify_tables[(b, w)] = self._upload(host, ttnn.int32)
            self.verify_caches[(b, w)] = [
                state
                if isinstance(state, RecurrentState)
                else PagedState(state.key, state.value, self.verify_tables[(b, w)])
                for state in self.cache
            ]
        self.spec_eager = False
        self.last_forward = None

    def _mtp_init(self):
        model, mtp = self.model, self.model.mtp
        state = mtp.create_paged_state(self.page_host)
        shared = next(page_table_for_attention(st) for st in self.cache if not isinstance(st, RecurrentState))
        own = page_table_for_attention(state)
        self.mtp_kv = (state.key, state.value)
        self.mtp_states = {"full": PagedState(state.key, state.value, shared)}
        ttnn.deallocate(own)
        for size, table in self.bucket_page_tables.items():
            self.mtp_states[("plain", size)] = PagedState(state.key, state.value, table)
        self.mtp_states[("plain", self.batch_size)] = self.mtp_states["full"]
        for shape, table in self.verify_tables.items():
            self.mtp_states[("verify",) + shape] = PagedState(state.key, state.value, table)
        K = self.spec_width - 1
        up = lambda fill, dtype: self._upload(torch.full((32,), fill, dtype=torch.int32), dtype)
        self.ptokens = self._upload(torch.zeros((1, 1, 1, 32), dtype=torch.int32), ttnn.uint32)
        self.ppos = [up(-1, ttnn.int32) for _ in range(K + 1)]
        self.prope = [up(0, ttnn.uint32) for _ in range(K + 1)]
        self.pupdate = [up(-1, ttnn.int32) for _ in range(self.spec_width)]
        self.dlogits = self.model.io._upload(
            torch.zeros(1, 1, 32, model.config.vocab_size // 4), getattr(ttnn, model.precision["sampling_dtype"])
        )
        self.dtokens = [self._upload(torch.zeros((1, 1, 1, 32), dtype=torch.int32), ttnn.uint32) for _ in range(K)]
        args = SimpleNamespace(
            vocab_size=model.config.vocab_size,
            padded_vocab_size=model.config.vocab_size,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            pad_logits_to_power_of_2=True,
        )
        greedy = SamplingParams(
            temperature=[1.0] * 32,
            top_k=[1] * 32,
            top_p=[0.0] * 32,
            presence_penalty=[0.0] * 32,
            frequency_penalty=[0.0] * 32,
            repetition_penalty=[1.0] * 32,
        )
        self.dsampling = SamplingGenerator(args=args, mesh_device=self.mesh, tt_ccl=SamplingCCL(model))
        self.dsampling.reset_sampling_params(greedy)
        self.vsampling = SamplingGenerator(args=args, mesh_device=self.mesh, tt_ccl=SamplingCCL(model))
        self.vsampling.reset_sampling_params(greedy)
        self._vsampling_signature = None
        self.vtokens_out = self._upload(torch.zeros((1, 1, 1, 32), dtype=torch.int32), ttnn.uint32)
        # prefill seeding (last prompt position of each prefill call, once its next token is known)
        self.seed_table = self._upload(self.page_host[:1].clone(), ttnn.int32)
        self.seed_state = PagedState(state.key, state.value, self.seed_table)
        self.seed_tok = self._upload(torch.zeros((1, 1, 1, 32), dtype=torch.int32), ttnn.uint32)
        self.seed_pos, self.seed_rope = up(-1, ttnn.int32), up(0, ttnn.uint32)
        self.mtp_pending = [
            self.model.io._upload(torch.zeros(1, 1, model.config.hidden_size)) for _ in range(self.batch_size)
        ]
        self.mtp_pending_pos = {}
        for engine in {self.prefill_single, self.prefill_uniform}:
            engine.mtp_hook = self._mtp_prefill_hook
            engine.mtp_state = PagedState(state.key, state.value, engine.full_table)
        self.counters["mtp_enabled"] = 1

    def _mtp_prefill_hook(self, engine, value, tokens, start_pos, absolute, count, end, slot, rope, aligned):
        model = self.model
        if engine.prefill_batch_size != 1:
            raise ValueError("MTP prefill fill supports single-row prefill")
        relative = absolute - start_pos
        nxt = torch.zeros((1, count), dtype=torch.int32)
        known = tokens[:, relative + 1 : relative + 1 + count]
        nxt[:, : known.shape[1]] = known
        normed = model.io._norm(value, "final_norm")
        engine._write(nxt, engine.ids[count])
        x = model.mtp_input(model.embeddings(engine.ids[count]), normed, decode=False)
        if aligned:
            engine._layer_chunk(model.mtp, x, engine.mtp_state, rope)
        else:
            engine._attention_head(model.mtp, x, engine.mtp_state, absolute)
        if absolute + count == end:
            ttnn.copy(ttnn.to_memory_config(normed[:, -1:, :], ttnn.DRAM_MEMORY_CONFIG), self.mtp_pending[slot])
            self.mtp_pending_pos[slot] = end - 1

    def _warm_mtp_prefill(self):
        engine = self.prefill_single
        for length in engine.prepared_lengths:
            value = self.model.embeddings(engine.ids[length])
            rope = self.model.rotations(engine.positions[length], batch=1, length=length)
            self._mtp_prefill_hook(
                engine, value, torch.zeros((1, length), dtype=torch.int32), 0, 0, length, length + 1, 0, rope, True
            )
        ttnn.synchronize_device(self.mesh)
        self.mtp_pending_pos.clear()

    def mtp_seed(self, slot, token):
        """Fill MTP KV at the pending last prompt position of ``slot`` once its next token is known."""
        position = self.mtp_pending_pos.pop(slot, None)
        if position is None or getattr(self.model, "mtp", None) is None:
            return
        self._write(self.page_host[slot : slot + 1].clone(), self.seed_table)
        tok = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
        tok.view(-1)[0] = int(token)
        self._write(tok, self.seed_tok)
        pos = torch.full((32,), -1, dtype=torch.int32)
        pos[0] = position
        self._write(pos, self.seed_pos)
        self._write(pos.clamp_min(0), self.seed_rope)
        self.model.mtp_decode(
            self.seed_tok,
            self.mtp_pending[slot],
            self.seed_pos[:1],
            cache_state=self.seed_state,
            batch=1,
            rope_positions=self.seed_rope[:1],
        )
        self.counters["mtp_seeds"] += 1

    def _propose_key(self, kind, rows, width, steps):
        return f"propose_{kind}{rows}_w{width}_k{steps}"

    def _propose_graph(self, kind, rows, width, steps):
        batch = rows * width
        state = self.mtp_states[("verify", rows, width) if kind == "v" else ("plain", rows)]
        hidden = self.spec_hidden[:, :batch, :]
        source = self.ptokens
        for k in range(max(1, steps)):
            spec = None
            if k == 0 and width > 1:
                spec = SimpleNamespace(width=width, update_positions=[self.pupdate[j][:batch] for j in range(width)])
            logits, hidden = self.model.mtp_decode(
                source,
                hidden,
                self.ppos[k][:batch],
                cache_state=state,
                batch=batch,
                rope_positions=self.prope[k][:batch],
                spec=spec,
            )
            if steps == 0:
                break  # KV fill only
            ttnn.copy(logits, self.dlogits)
            self.dsampling._run_sampling(self.dlogits, penalties_on=False, grammar_on=False, tt_out_tok=self.dtokens[k])
            source = self.dtokens[k]

    def propose_shapes(self):
        shapes = []
        for rows in self.decode_batch_sizes:
            steps = min(self.spec_width - 1, 32 // rows - 1)
            shapes.append(("p", rows, 1, steps))
            for r, w in self.verify_shapes:
                if r == rows:
                    shapes.append(("v", rows, w, steps))
        return shapes

    def propose(self, committed, positions, accepted_counts, num_drafts):
        """MTP drafts after the last forward. committed/positions: [B, >=1] host (the committed block and
        the positions those tokens sit at); accepted_counts [B]. Fills MTP KV for every committed position and
        returns host draft ids [B, num_drafts] (num_drafts may be 0 = KV fill only)."""
        kind, rows, width = self.last_forward
        committed = torch.as_tensor(committed, dtype=torch.int32)
        positions = torch.as_tensor(positions, dtype=torch.int32)
        counts = torch.as_tensor(accepted_counts, dtype=torch.int32).flatten()
        live_rows = min(rows, committed.shape[0])
        batch = rows * width
        steps_cap = min(self.spec_width - 1, 32 // rows - 1)
        steps = min(num_drafts, steps_cap)
        tok = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
        pos = [torch.full((32,), -1, dtype=torch.int32) for _ in range(len(self.ppos))]
        for b in range(live_rows):
            a = int(counts[b])
            if positions[b, 0] < 0:
                continue
            for j in range(min(a, width)):
                tok.view(-1)[j * rows + b] = committed[b, j]
                pos[0][j * rows + b] = positions[b, j] - 1
            last = (min(a, width) - 1) * rows + b
            for k in range(1, steps):
                pos[k][last] = positions[b, min(a, width) - 1] - 1 + k
        self._write(tok, self.ptokens)
        for k in range(len(self.ppos)):
            self._write(pos[k], self.ppos[k])
            self._write(pos[k].clamp_min(0), self.prope[k])
        if width > 1:
            for j in range(width):
                col = torch.full((32,), -1, dtype=torch.int32)
                col[j * rows : (j + 1) * rows] = pos[0][j * rows : (j + 1) * rows]
                self._write(col, self.pupdate[j])
        trace_steps = steps_cap if steps else 0
        key = self._propose_key(kind, rows, width, trace_steps)
        if self.spec_eager or key not in self.traces:
            self._propose_graph(kind, rows, width, trace_steps)
        else:
            self._execute(key)
        self.counters["propose_steps"] += 1
        drafts = torch.full((committed.shape[0], max(num_drafts, 0)), -1, dtype=torch.int64)
        if steps:
            values = [ttnn.to_torch(ttnn.get_device_tensors(t)[0]).flatten()[:32] for t in self.dtokens[:steps]]
            for b in range(live_rows):
                if positions[b, 0] < 0:
                    continue
                last = (min(int(counts[b]), width) - 1) * rows + b
                for k in range(steps):
                    drafts[b, k] = int(values[k][last])
        return drafts

    def _save_hidden(self, normalized):
        rows = normalized.shape[-2]
        value = ttnn.reshape(normalized, (1, rows, normalized.shape[-1]))
        if rows < 32:
            value = ttnn.pad(value, ((0, 0), (0, 32 - rows), (0, 0)), value=0.0)
        ttnn.copy(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG), self.spec_hidden)

    def _spec_args(self, rows, width):
        batch = rows * width
        return dict(
            spec=SimpleNamespace(
                rows=rows,
                width=width,
                select=self.spec_select,
                zero=self.spec_zero,
                update_positions=[self.vupdate[j][:batch] for j in range(width)] if width > 1 else None,
            ),
            save_hidden=self._save_hidden,
        )

    def _verify_key(self, rows, width):
        return f"verify_b{rows}_w{width}"

    def _verify_graph(self, rows, width):
        batch = rows * width
        value = self.model.decode(
            self.vtokens,
            self.vpositions[:batch],
            cache=self.verify_caches[(rows, width)],
            batch=batch,
            rope_positions=self.vrope[:batch],
            active_rows=self.active_rows,
            live_positions=self.vpositions,
            **self._spec_args(rows, width),
        )
        ttnn.copy(value, self.logits)

    def _write_select(self, select):
        select = torch.as_tensor(select, dtype=torch.int32).flatten()
        if not torch.equal(select, self.spec_select_host):
            self._write(select, self.spec_select, "spec_select_refreshes")
            self.spec_select_host = select.clone()

    def spec_select_from_counts(self, accepted_counts, active):
        """Row r's GDN state lives in candidate slot accepted_counts[r]-1 (only for live rows)."""
        select = torch.zeros(32, dtype=torch.int32)
        if accepted_counts is not None:
            counts = torch.as_tensor(accepted_counts, dtype=torch.int32).flatten()
            n = min(len(counts), len(active))
            if bool(((counts[:n] < 1) | (counts[:n] > self.spec_width)).any()):
                raise ValueError(f"accepted_counts outside 1..{self.spec_width}: {counts.tolist()}")
            select[:n] = torch.where(torch.as_tensor(active[:n]), counts[:n] - 1, 0)
        return select

    def _verify_sampling(self, rows, width, vp):
        """Per virtual row (j*B+b): request b's temperature/top-k/top-p; seed = f(request seed, position)."""
        from .sampling.generator import _hash_request_seed_to_device_seed

        t, k, p = [1.0] * 32, [1] * 32, [0.0] * 32
        seeds = [0] * 32
        for r in range(rows * width):
            b = r % rows
            temp = self.sampling_values["temperature"][b]
            if any(
                self.sampling_values[n][b] != d
                for n, d in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1))
            ):
                raise ValueError("Speculative verify does not apply penalties; penalized requests must decode plainly")
            if temp > 0:
                t[r], k[r], p[r] = 1.0 / temp, self.sampling_values["top_k"][b], self.sampling_values["top_p"][b]
            seed = self.request_seeds[b]
            base = secrets.randbelow(999999) + 1 if seed is None else _hash_request_seed_to_device_seed(int(seed), 0, 7)
            seeds[r] = base + max(int(vp[r]), 0)
        signature = (tuple(t), tuple(k), tuple(p))
        if signature != self._vsampling_signature:
            self.vsampling.reset_sampling_params(
                SamplingParams(
                    temperature=t,
                    top_k=k,
                    top_p=p,
                    presence_penalty=[0.0] * 32,
                    frequency_penalty=[0.0] * 32,
                    repetition_penalty=[1.0] * 32,
                )
            )
            self._vsampling_signature = signature
        self._write(torch.tensor(seeds, dtype=torch.int32), self.vsampling.tt_sampling.seeds_tt_tensor)

    def _vsample_graph(self):
        self.vsampling._run_sampling(self.logits, penalties_on=False, grammar_on=False, tt_out_tok=self.vtokens_out)

    def verify_forward(self, tokens, positions, page_table, accepted_counts=None, *, read_logits=True, sample=False):
        """Score B rows x W candidate tokens in one forward (token-major virtual rows j*B+b).

        tokens/positions: [B, W] (position -1 = padded column or inactive row). Returns host
        logits [B, W, V] when read_logits (diagnostic/host path)."""
        tokens = torch.as_tensor(tokens, dtype=torch.int32)
        positions = torch.as_tensor(positions, dtype=torch.int32)
        rows, width = tokens.shape
        if (rows, width) not in self.verify_tables:
            raise ValueError(f"No prepared verify shape for B={rows} W={width}")
        batch = rows * width
        vt = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
        vp = torch.full((32,), -1, dtype=torch.int32)
        vt.view(-1)[:batch] = torch.where(positions >= 0, tokens, 0).T.reshape(-1)
        vp[:batch] = positions.T.reshape(-1)
        self._write(vt, self.vtokens, "verify_token_refreshes")
        self._write(vp, self.vpositions, "verify_position_refreshes")
        self._write(vp.clamp_min(0), self.vrope)
        for j in range(width):
            column = torch.full((32,), -1, dtype=torch.int32)
            column[j * rows : (j + 1) * rows] = vp[j * rows : (j + 1) * rows]
            self._write(column, self.vupdate[j])
        table = torch.as_tensor(page_table, dtype=torch.int32)[:rows]
        width_pad = self.page_host.shape[1] - table.shape[1]
        if width_pad:
            table = torch.nn.functional.pad(table, (0, width_pad))
        host = table.repeat(width, 1)
        if not torch.equal(host, self.verify_table_host[(rows, width)]):
            self._write(host, self.verify_tables[(rows, width)], "verify_page_refreshes")
            self.verify_table_host[(rows, width)] = host
        self._write_select(self.spec_select_from_counts(accepted_counts, positions[:, 0] >= 0))
        key = self._verify_key(rows, width)
        if sample:
            self._verify_sampling(rows, width, vp)
        if self.spec_eager or key not in self.traces:
            self._verify_graph(rows, width)
        else:
            self._execute(key)
        self.counters["verify_steps"] += 1
        self.last_forward = ("v", rows, width)
        if sample:
            if self.spec_eager or "vsample" not in self.traces:
                self._vsample_graph()
            else:
                self._execute("vsample")
            ids = ttnn.to_torch(ttnn.get_device_tensors(self.vtokens_out)[0]).flatten()[:batch].to(torch.int64)
            return ids.reshape(width, rows).T.contiguous()
        if not read_logits:
            return self.logits
        logits = self._read_logits(self.logits)[0, 0, :batch]
        return logits.reshape(width, rows, -1).permute(1, 0, 2)

    def _restore_inactive_vector(self, target, old, mask):
        width = old.volume()
        current = ttnn.to_layout(ttnn.reshape(target, (1, width)), ttnn.TILE_LAYOUT)
        previous = ttnn.to_layout(ttnn.reshape(old, (1, width)), ttnn.TILE_LAYOUT)
        merged = ttnn.where(mask, current, previous)
        ttnn.copy(ttnn.reshape(ttnn.to_layout(merged, target.layout), target.shape), target)

    def _sample_graph(self, penalties_on=False):
        if self.logits.dtype != getattr(ttnn, self.model.precision["sampling_dtype"]):
            raise ValueError("Sampler input violates the selected sampling value dtype")
        old = None
        if self.batch_size > 1:
            old = ttnn.clone(self.tokens), ttnn.clone(self.sampling.tt_sampling.seeds_tt_tensor)
        logits = self.penalty_history.sample_logits() if penalties_on else self.logits
        self.sampling._run_sampling(logits, penalties_on=penalties_on, grammar_on=False, tt_out_tok=self.tokens)
        ttnn.plus_one(self.sampling.tt_sampling.seeds_tt_tensor)
        if old is not None:
            for target, previous in zip((self.tokens, self.sampling.tt_sampling.seeds_tt_tensor), old):
                self._restore_inactive_vector(target, previous, self.sample_active_rows)

    @staticmethod
    def _sampling_rows(value, batch):
        if hasattr(value, "tolist"):
            value = value.tolist()
        values = list(value) if isinstance(value, (list, tuple)) else [value] * batch
        if len(values) != batch:
            raise ValueError("One sampling value per submitted row is required")
        return values

    @staticmethod
    def _validate_device_sampling(sampling_params):
        if bool(torch.as_tensor(getattr(sampling_params, "enable_log_probs", False)).any()):
            raise ValueError("enable_log_probs requires the explicit host-logits compatibility path")

    def _penalty_slots(self, slots):
        return [
            slot
            for slot in slots
            if any(
                self.sampling_values[name][slot] != default
                for name, default in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1))
            )
        ]

    def _sample_trace_key(self, slots=None):
        slots = self.decode_active_slots if slots is None else slots
        if self._penalty_slots(slots):
            # The canonical counted graph increments every emitted row,
            # including neutral rows sharing a penalized batch.
            self.penalty_history_slots.update(slots)
            return "sample_penalties"
        return "sample"

    def configure_sampling(
        self,
        *,
        temperature=0.0,
        top_k=1,
        top_p=0.0,
        seed=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        reset_rng=True,
    ):
        """Update stable sampler inputs at request boundaries; retain both graphs."""

        def rows(value):
            values = self._sampling_rows(value, self.batch_size)
            return values + [values[-1]] * (32 - len(values))

        temperatures, ks, ps = rows(temperature), rows(top_k), rows(top_p)
        presence, frequency, repetition = rows(presence_penalty), rows(frequency_penalty), rows(repetition_penalty)
        for i in range(32):
            if temperatures[i] == 0:
                temperatures[i], ks[i], ps[i] = 1.0, 1, 0.0
            if not (1 <= ks[i] <= 64 and 0 <= ps[i] <= 1 and temperatures[i] > 0):
                raise ValueError("Device sampling supports top_k1..64, top_p0..1, nonnegative temperature")
            if not (-2 <= presence[i] <= 2 and -2 <= frequency[i] <= 2 and repetition[i] > 0):
                raise ValueError("Invalid presence/frequency/repetition penalty")
        seeds = rows(seed)
        values = dict(
            temperature=temperatures,
            top_k=ks,
            top_p=ps,
            presence_penalty=presence,
            frequency_penalty=frequency,
            repetition_penalty=repetition,
        )
        signature = tuple(tuple(value) for value in values.values())
        if not self.cache_sampling_params or signature != self._sampling_signature:
            # Public values remain temperatures so row remaps/reloads do not invert
            # them twice. The canonical low-level sampler consumes 1/T.
            device_values = {**values, "temperature": [1.0 / value for value in temperatures]}
            self.sampling.reset_sampling_params(SamplingParams(**device_values))
            self._sampling_signature = signature
            self.counters["sampling_param_refreshes"] += 1
        self.sampling_values = {name: value[: self.batch_size] for name, value in values.items()}
        self.request_seeds = seeds[: self.batch_size]
        if reset_rng:
            self._reset_rng()

    def _reset_rng(self, positions=None, *, preserve_inactive=False):
        from .sampling.generator import _hash_request_seed_to_device_seed

        seeds = self.request_seeds + [None] * (32 - self.batch_size)
        offsets = [0] * 32
        if positions is not None:
            absolute = torch.as_tensor(positions).flatten()
            if absolute.numel() != self.batch_size:
                raise ValueError("RNG positions must cover every configured slot")
            # Absolute prediction-input positions survive scheduler preemption
            # without requiring optional output-token histories. Prefill's first
            # sample uses end-1; each traced decode advances this stream once.
            offsets[: self.batch_size] = absolute.clamp_min(0).tolist()
        values = [
            (secrets.randbelow(999999) + 1 if s is None else _hash_request_seed_to_device_seed(int(s), 0)) + offset
            for s, offset in zip(seeds, offsets)
        ]
        target = self.sampling.tt_sampling.seeds_tt_tensor
        old = ttnn.clone(target) if preserve_inactive else None
        self._write(torch.tensor(values, dtype=torch.int32), target)
        if old is not None:
            self._restore_inactive_vector(target, old, self.sample_active_rows)

    def _stage_prefill_sample_logits(self, value, *, slot, batch):
        width = self.model.config.vocab_size // 4
        # Untile before merging logical batch rows: each prefill row has its own
        # padded sequence tile, while decode packs all slots into one tile.
        row = ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), (1, 1, batch, width))
        ttnn.copy(ttnn.to_layout(self.logits, ttnn.ROW_MAJOR_LAYOUT), self.prefill_sample_logits)
        # The interleaved slice-write hash excludes outer-dimension starts;
        # slot is a runtime offset. One warmup covers every destination slot.
        ttnn.experimental.slice_write(
            row, self.prefill_sample_logits, [0, 0, slot, 0], [1, 1, slot + batch, width], [1] * 4
        )
        ttnn.copy(ttnn.to_layout(self.prefill_sample_logits, ttnn.TILE_LAYOUT), self.logits)

    def _reset_graph(self):
        for source, target in self.reset_pairs:
            ttnn.copy(source, target)

    def release_request(self, slot):
        """Clear a released slot's recurrent history; paged KV belongs to its table."""
        if not 0 <= slot < self.batch_size:
            raise ValueError("Slot outside configured capacity")
        self._release_sampling_history(slot)
        self._release_recurrent_slot(slot)

    def _release_sampling_history(self, slot):
        """Reset request-owned sampler history outside device graph capture."""
        if slot in self.penalty_history_slots:
            self.penalty_history.reset(slots=[slot])
            self.penalty_history_slots.discard(slot)

    def _release_recurrent_slot(self, slot):
        """Device-only recurrent reset, also used inside the prefill trace."""
        for state in self.cache:
            if isinstance(state, RecurrentState):
                for target in (state.conv, state.recurrent):
                    source = self.reset_zeros[(tuple(target.shape), str(target.dtype), str(target.layout))]
                    if self.batch_size == 1:
                        ttnn.copy(source, target)
                    else:
                        begin = [slot] + [0] * (len(target.shape) - 1)
                        end = [slot + 1] + list(target.shape)[1:]
                        if target.layout == ttnn.TILE_LAYOUT:
                            staging = ttnn.to_layout(target, ttnn.ROW_MAJOR_LAYOUT)
                            row = ttnn.to_layout(source[:1], ttnn.ROW_MAJOR_LAYOUT)
                            ttnn.experimental.slice_write(row, staging, begin, end, [1] * len(target.shape))
                            ttnn.copy(ttnn.to_layout(staging, target.layout), target)
                        else:
                            ttnn.experimental.slice_write(source[:1], target, begin, end, [1] * len(target.shape))

    def _capture(self, key, function):
        if key in self.traces:
            return
        begin = time.perf_counter()
        print(f"CAPTURE {key}", flush=True)
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            function()
        except BaseException:
            ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
            ttnn.release_trace(self.mesh, trace)
            raise
        else:
            ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self.traces[key] = trace
        self.counters["captures"] += 1
        print(
            "QWEN_TRACE "
            + json.dumps(
                dict(
                    event="capture",
                    key=key,
                    trace_id=int(trace),
                    batch=self.batch_size,
                    logical_batch=int(key[8:]) if key.startswith("decode_b") else self.batch_size,
                    cache_pages=self.prefill_single.physical_pages,
                    seconds=time.perf_counter() - begin,
                    captures=self.counters["captures"],
                    retirements=self.counters["retirements"],
                )
            ),
            flush=True,
        )

    def _execute(self, key):
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
        self.counters[key + "_replays"] += 1
        if key.startswith("decode_b"):
            self.counters["decode_replays"] += 1

    def precision_runtime_summary(self):
        """Observe prepared storage without reading device contents or exposing page leases."""
        if not self.prepared:
            raise RuntimeError("Precision runtime summary requires completed preparation")
        policy = self.model.precision

        def tensor(value, field=None):
            if field is not None and value.dtype != getattr(ttnn, policy[field]):
                raise ValueError(f"Runtime tensor violates {field}={policy[field]}")
            return dict(dtype=str(value.dtype), shape=list(value.shape), layout=str(value.layout))

        states = []
        for layer, state in zip(self.model.layers, self.cache):
            row = dict(layer=layer.layer_idx, kind=layer.kind)
            if isinstance(state, RecurrentState):
                row.update(
                    conv=tensor(state.conv, "activation_dtype"),
                    recurrent=tensor(state.recurrent, "recurrent_state_dtype"),
                )
            else:
                if policy["kv_cache_layout"] != "TILE/page32/local_kv1/head256":
                    raise ValueError("Unsupported selected KV layout")
                if tuple(state.key.shape)[1:] != (1, 32, 256) or state.key.layout != ttnn.TILE_LAYOUT:
                    raise ValueError("Observed key layout violates the selected KV contract")
                if tuple(state.value.shape) != tuple(state.key.shape) or state.value.layout != ttnn.TILE_LAYOUT:
                    raise ValueError("Observed value layout violates the selected KV contract")
                row.update(
                    key=tensor(state.key, "kv_cache_dtype"),
                    value=tensor(state.value, "kv_cache_dtype"),
                    page_table=tensor(page_table_for_attention(state)),
                )
            states.append(row)
        sampler = self.sampling.tt_sampling
        return dict(
            version=1,
            context_allocation=self.max_seq_len,
            page_host_shape=list(self.page_host.shape),
            states=states,
            logits=tensor(self.logits, "logits_dtype"),
            sampling_input=tensor(self.logits, "sampling_dtype"),
            prefill_sampling_input=tensor(self.prefill_sample_logits, "sampling_dtype"),
            sampling_top_p=tensor(sampler.p_tensor, "sampling_dtype"),
            sampling_temperature=tensor(sampler.temp_tensor, "sampling_dtype"),
            token_output=tensor(self.tokens),
        )

    def prepare(self, *, prefill_lengths=()):
        """Prepare every physical prefill signature and stable trace input first."""
        if self.prepared:
            return
        begin = time.perf_counter()

        def warm(key, function):
            started = time.perf_counter()
            before = int(self.mesh.num_program_cache_entries())
            function()
            print(
                "QWEN_WARM "
                + json.dumps(
                    dict(
                        key=key,
                        host_seconds=time.perf_counter() - started,
                        programs_before=before,
                        programs_after=int(self.mesh.num_program_cache_entries()),
                    )
                ),
                flush=True,
            )

        warm("prefill_b1", self.prefill_single.prepare)
        if getattr(self.prefill_single, "mtp_hook", None) is not None:
            # perf/spec: the MTP prefill fill has length-dependent full-width ops; compile every
            # prepared length now instead of on a live request (first-use compiles cost ~1 s TTFT).
            warm("mtp_prefill", self._warm_mtp_prefill)
        if self.prefill_uniform is not self.prefill_single:
            warm(f"prefill_b{self.batch_size}", self.prefill_uniform.prepare)
        if self.prefill_trace is not None:
            self.prefill_trace.prepare()
        print("PREPARE decode", flush=True)
        if self.batch_size > 1:
            self._remap_graph()
        for slot in range(self.batch_size):
            self.release_request(slot)
        for batch in self.decode_batch_sizes:
            warm(self._decode_trace_key(batch), lambda batch=batch: self._decode_graph(batch))
        for rows, width in getattr(self, "verify_shapes", []):
            warm(self._verify_key(rows, width), lambda r=rows, w=width: self._verify_graph(r, w))
        if getattr(self, "dsampling", None) is not None:
            for kind, rows, width, steps in self.propose_shapes():
                for st in sorted({0, steps}):
                    warm(
                        self._propose_key(kind, rows, width, st),
                        lambda a=(kind, rows, width, st): self._propose_graph(*a),
                    )
            warm("vsample", self._vsample_graph)
            self.mtp_pending_pos[0] = 0
            warm("mtp_seed", lambda: self.mtp_seed(0, 0))
        for length in prefill_lengths:
            self._prefill_last_into_logits([1] * length)
        for batch in sorted({1, self.batch_size}):
            value = self.model.io._upload(torch.zeros(batch, 1, self.model.config.vocab_size // 4))
            self._stage_prefill_sample_logits(value, slot=0, batch=batch)
            del value
        ttnn.copy(self.sample_active_rows, self.prefill_previous_sample_mask)
        self._reset_rng(preserve_inactive=True)
        ttnn.copy(self.prefill_previous_sample_mask, self.sample_active_rows)
        warm("penalty_history", self.penalty_history.warm)
        self.configure_sampling(presence_penalty=0.5, frequency_penalty=0.25, repetition_penalty=1.1)
        warm("sample_penalties_first", lambda: self._sample_graph(True))
        warm("sample_penalties_second", lambda: self._sample_graph(True))
        self.penalty_history.reset()
        self.configure_sampling()
        print("PREPARE sample", flush=True)
        warm("sample_first", self._sample_graph)
        warm("sample_second", self._sample_graph)
        print("PREPARE reset", flush=True)
        self._reset_graph()
        self._write(torch.zeros(self.batch_size, dtype=torch.int32), self.positions)
        self._write(torch.zeros(self.batch_size, dtype=torch.int32), self.rope_positions)
        self._write(torch.zeros((1, 1, 1, 32), dtype=torch.int32), self.tokens)
        for batch in self.decode_batch_sizes:
            self._capture(self._decode_trace_key(batch), lambda batch=batch: self._decode_graph(batch))
        for rows, width in getattr(self, "verify_shapes", []):
            self._capture(self._verify_key(rows, width), lambda r=rows, w=width: self._verify_graph(r, w))
        if getattr(self, "dsampling", None) is not None:
            for kind, rows, width, steps in self.propose_shapes():
                for st in sorted({0, steps}):
                    self._capture(
                        self._propose_key(kind, rows, width, st),
                        lambda a=(kind, rows, width, st): self._propose_graph(*a),
                    )
            self._capture("vsample", self._vsample_graph)
        self._capture("sample", self._sample_graph)
        self._capture("sample_penalties", lambda: self._sample_graph(True))
        self.penalty_history_slots = set(range(self.batch_size))  # capture executed counted sampling
        self._capture("reset", self._reset_graph)
        if self.prefill_trace is not None:
            self._capture(self.prefill_trace.trace_key, self.prefill_trace.graph)
        self.prepared = True
        self.reset()
        self.precision_runtime_summary()
        print(
            "QWEN_TRACE "
            + json.dumps(
                dict(
                    event="prepared",
                    program_cache_entries=int(self.mesh.num_program_cache_entries()),
                    batch=self.batch_size,
                    seconds=time.perf_counter() - begin,
                    traces={key: int(trace) for key, trace in self.traces.items()},
                    captures=self.counters["captures"],
                    retirements=self.counters["retirements"],
                    prefill_engines=[
                        dict(
                            batch=e.prefill_batch_size,
                            lengths=e.prepared_lengths,
                            cache_pages=e.physical_pages,
                            mode="eager",
                        )
                        for e in {self.prefill_single, self.prefill_uniform}
                    ],
                )
            ),
            flush=True,
        )

    def reset(self):
        if self.penalty_history_slots:
            self.penalty_history.reset()
            self.penalty_history_slots.clear()
        self.decode_active_slots = list(range(self.batch_size))
        if self.prepared:
            self._execute("reset")
        else:
            self._reset_graph()
        self._write(torch.zeros(self.batch_size, dtype=torch.int32), self.positions)
        self._write(torch.zeros(self.batch_size, dtype=torch.int32), self.rope_positions)
        self._write(torch.zeros((1, 1, 1, 32), dtype=torch.int32), self.tokens)
        self._write(torch.arange(32, dtype=torch.int32), self.sampling.tt_sampling.seeds_tt_tensor)
        self._write(torch.ones(self.batch_size, dtype=torch.bfloat16), self.active_rows)
        self._write(
            torch.nn.functional.pad(torch.ones(1, self.batch_size), (0, 32 - self.batch_size)), self.sample_active_rows
        )
        self.single_slot_active = True

    def _bind_pages(self, page_table, kv_cache, *, force=False, prefix_only=False):
        if kv_cache is not self.cache:
            raise ValueError("Bind external cache at generator construction; allocations must survive every trace")
        page_table = page_table.to(torch.int32)
        if page_table.shape != self.page_host.shape:
            raise ValueError("Page table must retain its configured shape")
        if prefix_only and prefix_only not in self.bucket_page_tables:
            raise ValueError("Prefix page refresh requires a prepared bucket table")
        if torch.equal(page_table, self.page_host) and not force and (prefix_only or not self.full_page_tables_dirty):
            return
        if int(page_table.min()) < 0:
            raise ValueError("Use allocated scratch pages for inactive rows")
        max_page = int(page_table.max())
        if any(max_page >= state.key.shape[0] for state in self.cache if not isinstance(state, RecurrentState)):
            raise ValueError("Page address exceeds the cache allocation")
        if prefix_only:
            self._write(page_table[:prefix_only], self.bucket_page_tables[prefix_only], "prefix_page_refreshes")
            # B1 attention reads the shared prefix table, never the per-layer
            # full tables. Defer those uploads until prefill/full decode needs
            # them; even an unchanged host map must flush this dirty state.
            self.full_page_tables_dirty = True
            self.page_host.copy_(page_table)
            return
        written = set()
        certified = None
        for state in self.cache:
            if not isinstance(state, RecurrentState):
                if isinstance(state, CertifiedPagedState) and state._table_lease.certified:
                    # A revoked lease never becomes certified again. Internal
                    # updates may only preserve or revoke its original proof.
                    # perf/multiuser: vectorized (the list version cost ~20 ms
                    # of host time per bind) and skipped once revoked.
                    if certified is None:
                        owned = page_table[page_table >= 0]
                        certified = bool(owned.numel() == torch.unique(owned).numel())
                    state._table_lease.certified &= certified
                table = page_table_for_attention(state)
                # perf/multiuser: attention layers may share one device table.
                if table.buffer_address() in written:
                    continue
                written.add(table.buffer_address())
                self._write(page_table, table, "page_table_refreshes")
        self.page_host.copy_(page_table)
        self.full_page_tables_dirty = False

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        return_all_logits=False,
        return_device=False,
        empty_slots=None,
        sampling_params=None,
        sampling_mask=None,
        original_prompt_lens=None,
        enable_trace=True,
        page_tables_per_layer=None,
        **kwargs,
    ):
        # Prefill's bounded graph is eager for either enable_trace setting; its
        # optional final sampling always uses the already prepared sample trace.
        if page_tables_per_layer is not None:
            raise ValueError("Use one page table for the homogeneous full-attention KV group")
        if sampling_params is not None:
            self._validate_device_sampling(sampling_params)
            if return_all_logits or return_device:
                raise ValueError("Sampled prefill returns request-order host token IDs")
            if not self.prepared:
                raise RuntimeError("Call prepare before sampled prefill")
        tokens = torch.as_tensor(tokens, dtype=torch.int32)
        if tokens.ndim != 2 or tokens.shape[0] < 1:
            raise ValueError("Prefill tokens must contain one or more request rows")
        batch = tokens.shape[0]
        if sampling_mask is not None:
            if sampling_params is None:
                raise ValueError("A prefill sampling mask requires device sampling parameters")
            if (
                not isinstance(sampling_mask, (list, tuple))
                or len(sampling_mask) != batch
                or any(type(value) is not bool for value in sampling_mask)
            ):
                raise ValueError("One Boolean sampling mask value per submitted row is required")
        else:
            sampling_mask = [True] * batch
        sample_rows = [row for row, final in enumerate(sampling_mask) if final]
        if original_prompt_lens is not None:
            if (
                not isinstance(original_prompt_lens, (list, tuple))
                or len(original_prompt_lens) != batch
                or any(type(value) is not int or not 0 <= value <= self.max_seq_len for value in original_prompt_lens)
            ):
                raise ValueError("One valid original prompt length per submitted row is required")
        starts = [0] * batch if start_pos is None else list(map(int, start_pos))
        ends = list(map(int, prompt_lens))
        slots = list(range(batch)) if empty_slots is None else list(map(int, empty_slots))
        if len(ends) != batch or len(starts) != batch or len(slots) != batch:
            raise ValueError("One start, end and slot per prompt row is required")
        if len(set(slots)) != batch or any(not 0 <= slot < self.batch_size for slot in slots):
            raise ValueError("Prefill destinations must be distinct configured slots")
        for row, (start, end) in enumerate(zip(starts, ends)):
            if not 0 <= start < end <= min(self.max_seq_len, tokens.shape[1]):
                raise ValueError("Prompt interval outside configured capacity")
            ids = tokens[row, start:end]
            if bool(((ids < 0) | (ids >= self.model.config.vocab_size)).any()):
                raise ValueError("Prompt token outside vocabulary")
        if return_device and return_all_logits:
            raise ValueError("All-position diagnostic logits use the explicit host boundary")
        if sampling_params is not None:
            updates = {
                name: self._sampling_rows(getattr(sampling_params, name, default), batch)
                for name, default in (
                    ("temperature", 0.0),
                    ("top_k", 1),
                    ("top_p", 0.0),
                    ("seed", None),
                    ("presence_penalty", 0.0),
                    ("frequency_penalty", 0.0),
                    ("repetition_penalty", 1.0),
                )
            }
            for temperature, k, p in zip(updates["temperature"], updates["top_k"], updates["top_p"]):
                if temperature != 0 and not (temperature > 0 and 1 <= k <= 64 and 0 <= p <= 1):
                    raise ValueError("Device sampling supports top_k1..64, top_p0..1, nonnegative temperature")
        # External prefill tables follow submitted request rows, whereas decode
        # tables follow fixed state slots. Preserve mappings for unscheduled rows.
        table = torch.as_tensor(page_table, dtype=torch.int32)
        if table.ndim != 2 or table.shape[0] < batch or table.shape[1] != self.page_host.shape[1]:
            raise ValueError("Prefill page table must supply each submitted row at configured width")
        mapped = self.page_host.clone()
        mapped[slots] = table[:batch].clone()
        self._bind_pages(mapped, kv_cache)
        uniform = (
            self.batched_prefill
            and batch == self.batch_size
            and slots == list(range(batch))
            and len(set(starts)) == 1
            and len(set(ends)) == 1
        )
        legacy_release = uniform or return_all_logits or not self.prepared
        if legacy_release:
            for start, slot in zip(starts, slots):
                if start == 0:
                    self.release_request(slot)
        groups = (
            [(list(range(batch)), self.prefill_uniform)]
            if uniform
            else [([row], self.prefill_single) for row in range(batch)]
        )
        results = []
        for rows, engine in groups:
            row = rows[0]
            pieces = []
            # Plan immediately before running: a later row's snapshot may evict an
            # entry an earlier row restores, which is safe only once that row ran.
            plan = (
                {}
                if legacy_release
                else self._prefill_plan(tokens[row], starts[row], ends[row], slots[row], sampling_mask[row])
            )
            begin = plan.get("start", starts[row])
            if getattr(self, "mtp_pending_pos", None) and self.mtp_pending_pos.get(slots[row]) == starts[row] - 1:
                # perf/spec: the previous scheduler chunk's last position now knows its next token.
                self.mtp_seed(slots[row], int(tokens[row, starts[row]]))
            elif getattr(self, "mtp_pending_pos", None):
                self.mtp_pending_pos.pop(slots[row], None)
            for _, _, logits in engine.run(
                tokens[rows, begin : ends[row]],
                slot=slots[row],
                start_pos=begin,
                page_table=self.page_host,
                return_all_logits=return_all_logits,
                allow_unprepared=not self.prepared,
                restore=plan.get("restore"),
                zero_state=plan.get("zero", False),
                snapshot_at=plan.get("snapshot_at"),
                snapshot=plan.get("snapshot"),
            ):
                if sampling_params is not None:
                    if any(sampling_mask[index] for index in rows):
                        self._stage_prefill_sample_logits(logits, slot=slots[row], batch=len(rows))
                    # The loop target otherwise retains the yielded terminal
                    # allocation across replay, outside the capture snapshot.
                    del logits
                else:
                    pieces.append(logits if return_device else self._read_logits(logits))
            if sampling_params is None:
                results.append(pieces[-1] if return_device else torch.cat(pieces, dim=1))
        if sampling_params is not None:
            # An intermediate scheduler chunk still exhausts engine.run above:
            # its final yield is followed by publication of recurrent state.
            # It has no sample, RNG update, or logits/token D2H boundary.
            if not sample_rows:
                return torch.zeros((batch, 1), dtype=torch.int64)
            sample_slots = [slots[row] for row in sample_rows]
            merged = {name: values.copy() for name, values in self.sampling_values.items()}
            merged["seed"] = self.request_seeds.copy()
            for name, values in updates.items():
                for row in sample_rows:
                    merged[name][slots[row]] = values[row]
            self.configure_sampling(**merged, reset_rng=False)
            penalty_slots = self._penalty_slots(sample_slots)
            if penalty_slots:
                if original_prompt_lens is None:
                    raise ValueError("Device prefill penalties require original_prompt_lens")
                prompts = torch.full((self.batch_size, tokens.shape[1]), -1, dtype=torch.int64)
                outputs = torch.full_like(prompts, -1)
                for row in sample_rows:
                    slot, end, boundary = slots[row], ends[row], original_prompt_lens[row]
                    if boundary > end:
                        raise ValueError("Final prefill row must include its complete original prompt")
                    prompts[slot, :boundary] = tokens[row, :boundary]
                    outputs[slot, : end - boundary] = tokens[row, boundary:end]
                self.penalty_history.reset(slots=penalty_slots, prompt_tokens=prompts, output_tokens=outputs)
                self.penalty_history_slots.update(penalty_slots)
            ttnn.copy(self.sample_active_rows, self.prefill_previous_sample_mask)
            mask = torch.zeros(1, 32, dtype=torch.bfloat16)
            mask[0, sample_slots] = 1
            self._write(mask, self.sample_active_rows)
            try:
                positions = torch.zeros(self.batch_size, dtype=torch.int32)
                positions[sample_slots] = torch.tensor([ends[row] - 1 for row in sample_rows], dtype=torch.int32)
                self._reset_rng(positions, preserve_inactive=True)
                self._execute(self._sample_trace_key(sample_slots))
                sampled = self._read_tokens()
            finally:
                ttnn.copy(self.prefill_previous_sample_mask, self.sample_active_rows)
            if getattr(self, "mtp_pending_pos", None):
                for row in sample_rows:
                    self.mtp_seed(slots[row], sampled[slots[row]])
            return torch.tensor(
                [sampled[slot] if final else 0 for slot, final in zip(slots, sampling_mask)], dtype=torch.int64
            ).reshape(batch, 1)
        if return_device:
            return results
        if return_all_logits and len({x.shape[1] for x in results}) > 1:
            width = max(x.shape[1] for x in results)
            results = [torch.nn.functional.pad(x, (0, 0, 0, width - x.shape[1])) for x in results]
        return torch.cat(results, dim=0)

    # ---- perf/multiuser: GDN prefix cache (recurrent-state snapshots) ----------
    @staticmethod
    def _prefix_hash(row_tokens, length):
        import hashlib

        return hashlib.blake2b(row_tokens[:length].numpy().tobytes(), digest_size=16).digest()

    def enable_prefix_cache(self, snapshots, min_tokens=256):
        """Keep up to `snapshots` GDN state snapshots taken at 32-aligned prompt ends.

        vLLM's prefix cache restores attention KV pages; the model restores the
        matching recurrent/conv state from the longest snapshot whose token prefix
        hash matches, then recomputes from that point (KV rewrites of already
        cached positions recompute the same values).
        """
        if self.traces:
            raise RuntimeError("Snapshot buffers must be allocated before any trace capture")
        self.prefix_snapshots = snapshots
        self.prefix_min_tokens = min_tokens
        # Preallocate: buffers created after capture could alias trace scratch.
        self.prefix_free = [self.prefill_single.new_snapshot() for _ in range(snapshots)]
        self.prefix_entries = []  # dicts: length, key, buffers, used
        self.prefix_clock = 0

    def _prefix_lookup(self, row_tokens, limit):
        best = None
        hashes = {}
        for entry in self.prefix_entries:
            length = entry["length"]
            if length > limit or (best is not None and length <= best["length"]):
                continue
            if length not in hashes:
                hashes[length] = self._prefix_hash(row_tokens, length)
            if hashes[length] == entry["key"]:
                best = entry
        return best

    def _prefix_allocate(self, length, key, protect):
        for entry in self.prefix_entries:
            if entry["length"] == length and entry["key"] == key:
                return None  # already present
        if self.prefix_free:
            entry = dict(buffers=self.prefix_free.pop())
            self.prefix_entries.append(entry)
        else:
            candidates = [e for e in self.prefix_entries if e is not protect]
            entry = min(candidates, key=lambda e: e["used"])
            self.counters["prefix_evictions"] += 1
        self.prefix_clock += 1
        entry.update(length=length, key=key, used=self.prefix_clock)
        return entry

    def _prefill_plan(self, row_tokens, start, end, slot, final):
        """Decide state source, effective start and snapshot point for one prefill row."""
        if not hasattr(self, "prefill_pending"):
            self.prefill_pending = {}
        caching = getattr(self, "prefix_snapshots", 0) > 0
        plan = {}
        pending = self.prefill_pending.pop(slot, None)
        continuation = (
            start > 0
            and pending is not None
            and pending[0] == start
            and (not caching or pending[1] == self._prefix_hash(row_tokens, start))
        )
        if not continuation:
            # New (or resumed / prefix-hit) request in this slot: never import
            # the slot's stale state. Zero the work state instead of zeroing the
            # slot in place (the final publish overwrites the slot anyway).
            self._release_sampling_history(slot)
            entry = self._prefix_lookup(row_tokens, start) if caching and start > 0 else None
            if entry is not None:
                self.prefix_clock += 1
                entry["used"] = self.prefix_clock
                plan.update(start=entry["length"], restore=entry["buffers"], entry=entry)
                self.counters["prefix_hits"] += 1
                self.counters["prefix_tokens_reused"] += entry["length"]
                self.counters["prefix_tokens_recomputed"] += start - entry["length"]
            else:
                plan.update(start=0, zero=True)
                if start > 0:
                    self.counters["prefix_misses"] += 1
                    self.counters["prefix_tokens_recomputed"] += start
        begin = plan.get("start", start)
        if final:
            if caching:
                boundary = end // 32 * 32
                if boundary > begin and boundary >= self.prefix_min_tokens:
                    key = self._prefix_hash(row_tokens, boundary)
                    entry = self._prefix_allocate(boundary, key, plan.get("entry"))
                    if entry is not None:
                        plan.update(snapshot_at=boundary, snapshot=entry["buffers"])
                        self.counters["prefix_snapshots"] += 1
        else:
            self.prefill_pending[slot] = (end, self._prefix_hash(row_tokens, end) if caching else None)
        if caching or not continuation:
            print(
                "QWEN_PREFIX "
                + json.dumps(
                    dict(
                        slot=slot,
                        start=start,
                        end=end,
                        final=bool(final),
                        continuation=continuation,
                        begin=begin,
                        restored=plan.get("restore") is not None,
                        snapshot_at=plan.get("snapshot_at"),
                        entries=len(getattr(self, "prefix_entries", [])),
                    )
                ),
                flush=True,
            )
        plan.pop("entry", None)
        return plan

    def prefill_logits(self, prompt_token_ids):
        self.reset()
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_host,
            kv_cache=self.cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    def _prefill_last_into_logits(self, prompt_token_ids):
        if (
            self.prefill_trace is not None
            and len(prompt_token_ids) == self.prefill_trace.length
            and self.prefill_trace.trace_key in self.traces
        ):
            self.prefill_trace.upload(prompt_token_ids)
            self._execute(self.prefill_trace.trace_key)
            self.prefill_trace.record_replay()
            return
        value = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_host,
            kv_cache=self.cache,
            prompt_lens=[len(prompt_token_ids)],
            return_device=True,
        )[0]
        value = ttnn.reshape(value, (1, 1, 1, self.model.config.vocab_size // 4))
        padded = ttnn.pad(value, ((0, 0), (0, 0), (0, 31), (0, 0)), value=0)
        ttnn.copy(padded, self.logits)

    def prefill_logits_continuation(self, prompt_token_ids, start, *, positions=None, prefill_prefix=True):
        self.reset()
        tokens = torch.tensor([prompt_token_ids])
        if start and prefill_prefix:
            self.prefill_forward(tokens, page_table=self.page_host, kv_cache=self.cache, prompt_lens=[start])
        result = self.prefill_forward(
            tokens,
            page_table=self.page_host,
            kv_cache=self.cache,
            prompt_lens=[len(prompt_token_ids)],
            start_pos=[start],
            return_all_logits=True,
        )
        return result if positions is None else result[:, [p - start for p in positions], :]

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        sample_on_device=True,
        resident_inputs=None,
        slot_remap=None,
        read_from_device=True,
        enable_trace=True,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        sampling_params=None,
        output_tokens=None,
        prompt_tokens=None,
        page_tables_per_layer=None,
        logical_batch=None,
        **kwargs,
    ):
        if page_tables_per_layer is not None:
            raise ValueError("Use one page table for the homogeneous full-attention KV group")
        if "reset_batch" in kwargs:
            raise TypeError("Legacy reset_batch is incompatible with decode_input_update_contract=1")
        if resident_inputs is not None:
            reload_inputs = not resident_inputs
        resident_inputs = not reload_inputs
        if reload_inputs and reload_page_table:
            raise ValueError("reload_inputs already includes page tables")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Sampling-state reset requires authoritative inputs")
        if reload_sampling_params and sampling_params is None:
            raise ValueError("Sampling reload requires sampling_params")
        if reload_sampling_params and sample_on_device:
            self._validate_device_sampling(sampling_params)
        if not enable_trace:
            raise ValueError("Low-level decode requires the prepared device traces")
        if not self.prepared:
            raise RuntimeError("Call prepare before prefill to retain request state")
        logical_batch = self.batch_size if logical_batch is None else logical_batch
        if logical_batch not in self.decode_batch_sizes:
            raise ValueError("Decode batch has no prepared trace")
        if logical_batch != self.active_decode_batch and not reload_inputs:
            raise ValueError("A decode bucket transition requires authoritative inputs")
        if slot_remap is not None:
            if resident_inputs:
                raise ValueError("A slot transition requires authoritative token/position inputs")
        if not resident_inputs:
            start_pos = torch.as_tensor(start_pos, dtype=torch.int32).flatten()
            incoming = torch.as_tensor(tokens, dtype=torch.int32).flatten()
            if incoming.numel() != self.batch_size or start_pos.numel() != self.batch_size:
                raise ValueError("Decode inputs must cover the configured fixed slots")
            if bool(((start_pos < -1) | (start_pos >= self.max_seq_len)).any()):
                raise ValueError("Decode position outside configured context")
            active = start_pos >= 0
            if logical_batch < self.batch_size and bool(active[logical_batch:].any()):
                raise ValueError("Prefix-bucket decode requires inactive rows beyond the bucket")
            if bool((((incoming < 0) | (incoming >= self.model.config.vocab_size)) & active).any()):
                raise ValueError("Active token outside vocabulary")
        if reload_inputs or reload_page_table:
            self._bind_pages(
                page_table,
                kv_cache,
                force=True,
                prefix_only=logical_batch if logical_batch < self.batch_size else False,
            )
        elif kv_cache is not self.cache:
            raise ValueError("Resident decode must retain its bound cache allocation")
        if slot_remap is not None:
            kwargs["accepted_counts"] = self.remap_slots(slot_remap, kwargs.get("accepted_counts"))
        if reload_sampling_params:
            self.configure_sampling(
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
                seed=sampling_params.seed,
                presence_penalty=getattr(sampling_params, "presence_penalty", 0.0),
                frequency_penalty=getattr(sampling_params, "frequency_penalty", 0.0),
                repetition_penalty=getattr(sampling_params, "repetition_penalty", 1.0),
                reset_rng=False,
            )
        if reset_sampling_state:
            self._reset_rng(start_pos)
            penalty_slots = self._penalty_slots(active.nonzero().flatten().tolist()) if sample_on_device else []
            if penalty_slots:
                if prompt_tokens is None or output_tokens is None:
                    raise ValueError("Device penalty reset requires authoritative prompt/output history")
                self.penalty_history.reset(
                    slots=penalty_slots, prompt_tokens=prompt_tokens, output_tokens=output_tokens
                )
                self.penalty_history_slots.update(penalty_slots)
        if not resident_inputs:
            values = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
            values.flatten()[: self.batch_size] = torch.where(active, incoming, 0)
            self._write(values, self.tokens, "token_refreshes")
            self._write(start_pos.to(torch.int32).flatten(), self.positions, "position_refreshes")
            self._write(start_pos.to(torch.int32).flatten().clamp_min(0), self.rope_positions, "rope_refreshes")
            self._write(active.to(torch.bfloat16), self.active_rows, "active_row_refreshes")
            self._write(
                torch.nn.functional.pad(active[None].to(torch.bfloat16), (0, 32 - self.batch_size)),
                self.sample_active_rows,
            )
            self.single_slot_active = bool(active[0])
            self.decode_active_slots = active.nonzero().flatten().tolist()
            self.active_decode_batch = logical_batch
        if self.spec_width and not resident_inputs:
            self._write_select(self.spec_select_from_counts(kwargs.get("accepted_counts"), active))
        elif self.spec_width and kwargs.get("accepted_counts") is not None:
            raise ValueError("A speculative commit requires authoritative decode inputs")
        if logical_batch == 1 and not self.single_slot_active:
            if not read_from_device:
                return self.tokens if sample_on_device else self.logits
            return torch.tensor([[-1]]) if sample_on_device else torch.zeros(1, 1, self.model.config.vocab_size)
        self._execute(self._decode_trace_key(logical_batch))
        if self.spec_width:
            self.last_forward = ("p", logical_batch, 1)
        if sample_on_device:
            self._execute(self._sample_trace_key())
            # Borrowed persistent output: consume before the next replay or
            # retain it as device-owned feedback. No transfer or synchronization.
            if not read_from_device:
                return self.tokens
            return torch.tensor(self._read_tokens(), dtype=torch.int64).reshape(self.batch_size, 1)
        if not read_from_device:
            return self.logits
        return self._read_logits(self.logits)[0, 0, : self.batch_size, None, :]

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=False,
        temperature=0.0,
        top_k=1,
        top_p=0.0,
        seed=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        host_sampler=None,
        ignore_eos=False,
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("Untraced generation is not a supported runtime path")
        if self.batch_size != 1:
            raise ValueError("Use prefill_forward/decode_forward for multiple fixed request slots")
        if host_sampling and temperature != 0 and host_sampler is None:
            raise ValueError("Non-greedy host compatibility requires an explicit host_sampler callback")
        if host_sampler is not None and not host_sampling:
            raise ValueError("host_sampler requires the explicit host_sampling compatibility mode")
        if (
            not host_sampling
            and next_input is not None
            and any(
                value != neutral
                for value, neutral in ((presence_penalty, 0.0), (frequency_penalty, 0.0), (repetition_penalty, 1.0))
            )
        ):
            raise ValueError("Teacher-forced next_input does not support device sampling penalties")
        if len(prompt_token_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError("Generation exceeds the configured context capacity")
        if max_new_tokens < 1:
            return []
        self.prepare(prefill_lengths=(len(prompt_token_ids),))
        begin = time.perf_counter()
        self.reset()
        if not host_sampling:
            self.configure_sampling(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
            )
        self._prefill_last_into_logits(prompt_token_ids)
        if not host_sampling:
            if self._penalty_slots([0]):
                self.penalty_history.reset(slots=[0], prompt_tokens=torch.tensor([prompt_token_ids]))
                self.penalty_history_slots.add(0)
            self._reset_rng(torch.tensor([len(prompt_token_ids) - 1]))
        if host_sampling:
            # The explicit callback owns host RNG/filtering/penalty policy.
            # Device top-k limits do not restrict that compatibility boundary.
            logits = self._read_logits(self.logits)[0, 0, 0]
            token = int(logits.argmax() if host_sampler is None else host_sampler(logits))
        else:
            self._execute(self._sample_trace_key())
            token = self._read_tokens()[0]
        first = time.perf_counter()
        output = [token]
        eos = self.model.config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        stop_at_eos = next_input is None and not ignore_eos
        initial = token if next_input is None else next_input(0, token)
        values = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
        values.flatten()[0] = initial
        if next_input is not None or host_sampling:
            self._write(values, self.tokens, "token_refreshes")
        self._write(torch.tensor([len(prompt_token_ids)], dtype=torch.int32), self.positions, "position_refreshes")
        self._write(torch.tensor([len(prompt_token_ids)], dtype=torch.int32), self.rope_positions, "rope_refreshes")
        for step in range(1, max_new_tokens):
            if stop_at_eos and output[-1] in eos_ids:
                break
            self._execute("decode")
            if host_sampling:
                logits = self._read_logits(self.logits)[0, 0, 0]
                token = int(logits.argmax() if host_sampler is None else host_sampler(logits))
            else:
                self._execute(self._sample_trace_key())
                token = self._read_tokens()[0]
            output.append(token)
            if next_input is not None or host_sampling:
                values.flatten()[0] = token if next_input is None else next_input(step, token)
                self._write(
                    values, self.tokens, "teacher_forcing_refreshes" if next_input else "host_sampling_refreshes"
                )
        self.last_timings = dict(
            ttft_seconds=first - begin,
            decode_seconds=time.perf_counter() - first,
            prompt_length=len(prompt_token_ids),
            generated=len(output),
        )
        return output

    def close(self):
        for key, trace in self.traces.items():
            ttnn.release_trace(self.mesh, trace)
            print("QWEN_TRACE " + json.dumps(dict(event="shutdown_release", key=key, trace_id=int(trace))), flush=True)
        self.traces.clear()


def build_generator(model_dir, mesh_device, **kwargs):
    model = Model(
        mesh_device,
        checkpoint=kwargs.pop("checkpoint", None),
        layer_indices=kwargs.pop("layer_indices", None),
        precision_config=kwargs.pop("precision_config", None),
    )
    return QwenGenerator(model, **kwargs)

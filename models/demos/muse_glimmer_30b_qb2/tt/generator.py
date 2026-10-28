# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness generator with persistent state and canonical split sampling.

Host logits are an explicit diagnostic/compatibility boundary. Optimized decode
replays model and sampling traces, with device token feedback and positions.
External caches must be bound before preparation and retain their allocations.
"""

import json
import time
from collections import Counter
from contextlib import contextmanager

import torch

try:
    from readiness_check.contract import Generator
except ImportError:  # Serving images (TTI/CI) do not ship the bring-up runtime.
    from models.demos.muse_glimmer_30b_qb2.tt.readiness_contract import Generator

from transformers import TokenizersBackend
from tt_transformers.modules.sampling.sampling_1d import Sampling1D

import ttnn
from models.demos.muse_glimmer_30b_qb2.tt.model import SNAPSHOT, MuseGlimmerModel
from models.demos.muse_glimmer_30b_qb2.tt.prepared_prefill import PreparedPrefill


class MuseGlimmerGenerator(Generator):
    BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
    APPEND_ALIGNMENTS = (32, 64, 96)
    TRACE_COUNT = 2 * len(BUCKETS) + len(APPEND_ALIGNMENTS) + 5

    def __init__(
        self,
        model,
        *,
        batch_size=1,
        num_pages=4096,
        kv_cache=None,
        tokenizer=None,
        prepare=True,
        trace_reset=True,
        clear_cache_on_generate=False,
        per_layer_page_tables=False,
        page_table_groups=None,
        event_sink=None,
    ):
        if not 1 <= batch_size <= 32 or num_pages < 4096:
            raise ValueError("Require batch 1..32 and at least 4096 physical cache pages")
        self.model, self.mesh, self.batch_size = model, model.mesh, batch_size
        self.num_pages = num_pages
        self.event_sink = event_sink
        self.per_layer_page_tables = per_layer_page_tables
        table_count = len(model.layers) if per_layer_page_tables else 1
        groups = list(range(table_count)) if page_table_groups is None else list(page_table_groups)
        if len(groups) != table_count:
            raise ValueError("One page-table group is required per bound layer table")
        first_in_group = {}
        self._page_table_representatives = [first_in_group.setdefault(group, i) for i, group in enumerate(groups)]
        self._unique_page_table_indices = list(first_in_group.values())
        self.tokenizer = tokenizer or TokenizersBackend.from_pretrained(SNAPSHOT, local_files_only=True)
        generation_config = json.loads((SNAPSHOT / "generation_config.json").read_text())
        eos_ids = generation_config.get("eos_token_id", model.config.eos_token_id)
        self.stop_token_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        self.kv_cache = (
            kv_cache if kv_cache is not None else model.allocate_cache(batch_size=batch_size, num_pages=num_pages)
        )
        if len(self.kv_cache) != len(model.layers) or any(
            len(pair) != 2
            or any(tuple(cache.shape) != (num_pages, 1, 32, 128) or cache.dtype != layer.kv_dtype for cache in pair)
            for pair, layer in zip(self.kv_cache, model.layers)
        ):
            raise ValueError("External cache must match the layer count and selected dtype and paged-cache geometry")
        self.counters = Counter()
        self.trace_reset = trace_reset
        self.clear_cache_on_generate = clear_cache_on_generate
        self.TRACE_COUNT = type(self).TRACE_COUNT + int(trace_reset)
        self.traces = {}
        self.prepared = False
        self._page_host = None
        self.tokens = self._upload(
            torch.zeros((1, 1, 1, batch_size), dtype=torch.int32),
            getattr(ttnn, model.precision_config["sampling_index_dtype"]),
        )
        self.positions = self._upload(torch.full((batch_size,), -1, dtype=torch.int32), ttnn.int32)
        self.rope_indices = self._upload(torch.zeros((1, batch_size), dtype=torch.int32), ttnn.uint32)
        self.page_table = self._upload(torch.zeros((batch_size, 4096), dtype=torch.int32), ttnn.int32)
        self.layer_page_tables = [self.page_table]
        if per_layer_page_tables:
            for i, representative in enumerate(self._page_table_representatives[1:], 1):
                self.layer_page_tables.append(
                    self._upload(torch.zeros((batch_size, 4096), dtype=torch.int32), ttnn.int32)
                    if representative == i
                    else self.layer_page_tables[representative]
                )
        self._layer_page_hosts = [None] * len(self.layer_page_tables)
        self.logits = ttnn.zeros(
            (1, 1, batch_size, model.padded_vocab_size // 4),
            device=self.mesh,
            dtype=model.logits_dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        self.prefill_logits_buffer = ttnn.zeros_like(self.logits)
        self.sampler = Sampling1D(
            model.padded_vocab_size,
            mesh_device=self.mesh,
            valid_vocab_size=model.vocab_size,
            max_batch_size=batch_size,
            max_top_k=32,
            tt_ccl=model.terminal_ccl,
            num_gather_links=2,
            pad_to_power_of_2=True,
            allow_force_argmax=True,
            ag_topology=ttnn.Topology.Ring,
        )
        self.sampler.load_device_buffers()
        # Admission snapshots are persistent and bit-exact, including uint32
        # RNG seeds. They are used only at a mixed-request prefill boundary.
        self.admission_mask = model.upload(torch.zeros((1, 1, 1, batch_size), dtype=torch.int32), dtype=ttnn.uint32)
        self.saved_tokens = ttnn.zeros_like(self.admission_mask)
        self.saved_seeds = ttnn.zeros_like(self.admission_mask)
        self.k = self._upload(torch.ones(batch_size, dtype=torch.int32), ttnn.uint32)
        self.p = self._upload(
            torch.zeros(batch_size), getattr(ttnn, model.precision_config["sampling_parameter_dtype"])
        )
        self.temperature = self._upload(
            torch.ones(batch_size), getattr(ttnn, model.precision_config["sampling_parameter_dtype"])
        )
        self.prefill = {}
        for size in self.BUCKETS:
            self.prefill[size] = dict(
                tokens=self._upload(torch.zeros((1, size), dtype=torch.int32), ttnn.uint32),
                rope=self._upload(torch.arange(size, dtype=torch.int32)[None], ttnn.uint32),
                table=self._upload(torch.arange(4096, dtype=torch.int32)[None], ttnn.int32),
                fill=self._upload(torch.arange(size // 32, dtype=torch.int32)[None], ttnn.int32),
                start=self._upload(torch.zeros(1, dtype=torch.int32), ttnn.int32),
                tail_start=self._upload(torch.full((1,), 32, dtype=torch.int32), ttnn.int32) if size == 64 else None,
                hidden=ttnn.zeros(
                    (1, 1, size, 6656), device=self.mesh, dtype=model.residual_dtype, layout=ttnn.TILE_LAYOUT
                ),
            )
            if per_layer_page_tables:
                state = self.prefill[size]
                for name, width in (("table", 4096), ("fill", size // 32)):
                    values = [state[name]]
                    for i, representative in enumerate(self._page_table_representatives[1:], 1):
                        values.append(
                            self._upload(torch.arange(width, dtype=torch.int32)[None], ttnn.int32)
                            if representative == i
                            else values[representative]
                        )
                    state[name] = values
        # Device-only slice consumes a rank-length vector, even when only one
        # dimension varies. slice_dim controls output shape, not index rank.
        self.slice_start = self._upload(self._slice_coordinates(0), ttnn.int32)
        self.slice_end = self._upload(self._slice_coordinates(32, end=True), ttnn.int32)
        self.last_indices = self._upload(torch.zeros((1, batch_size), dtype=torch.int32), ttnn.uint32)
        self.row_mask = model.upload(torch.zeros((1, 1, batch_size, 1)))
        self._host_positions = [-1] * batch_size
        if prepare:
            self.prepare()

    def _upload(self, value, dtype):
        return self.model.upload(value, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    @staticmethod
    def _slice_coordinates(row, *, end=False):
        # The current slice reader stages a whole BF16 tile (2048 bytes) from
        # each control tensor, while consuming only the first four uint32s.
        # Supply owned backing storage for that transfer as well as full-rank
        # coordinates. This avoids reading beyond a tiny RM allocation.
        coordinates = torch.zeros(512, dtype=torch.int32)
        coordinates[:4] = torch.tensor([1, 1, row, 6656] if end else [0, 0, row, 0])
        return coordinates

    def _write(self, value, tensor, counter=None):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, tensor)
        if counter:
            self.counters[counter] += 1

    def _read_tokens(self):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).flatten().tolist()[: self.batch_size]

    def _read_logits(self, logits):
        self.counters["full_logits_readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))[
            ..., : self.model.vocab_size
        ]

    @contextmanager
    def _prefill_policy(self, size):
        saved = []
        if size <= 64:
            for layer in self.model.layers:
                old = layer.norm_configs["prefill"]["input_layernorm"]
                saved.append((layer, layer.policy, old))
                layer.policy = dict(
                    layer.policy, **PreparedPrefill.POLICY, **PreparedPrefill.KIND_POLICY.get(layer.kind, {})
                )
                if layer.kind == "sliding_attention":
                    layer.norm_configs["prefill"]["input_layernorm"] = ttnn.init_device_compute_kernel_config(
                        self.mesh.arch(),
                        math_fidelity=getattr(
                            ttnn.MathFidelity,
                            self.model.precision_config["short_prefill_sliding_input_norm"]["fidelity"],
                        ),
                        math_approx_mode=False,
                        fp32_dest_acc_en=self.model.precision_config["short_prefill_sliding_input_norm"]["fp32"],
                        packer_l1_acc=True,
                    )
        try:
            yield
        finally:
            for layer, policy, config in saved:
                layer.policy = policy
                layer.norm_configs["prefill"]["input_layernorm"] = config

    def _prefill_graph(self, size, alignment=0):
        state = self.prefill[size]
        with self._prefill_policy(size):
            hidden = self.model.prefill_chunk(
                state["tokens"],
                kv_cache=self.kv_cache,
                page_table=state["table"],
                fill_page_table=state["fill"],
                start_pos=state["start"],
                rope_indices=state["rope"],
                tail_start=state["tail_start"],
                alignment=alignment,
            )
        ttnn.copy(hidden, state["hidden"])

    def _head_graph(self, size, all_rows=False):
        hidden = ttnn.slice(
            self.prefill[size]["hidden"], self.slice_start, self.slice_end, slice_dim=2, num_devices=size // 32
        )
        if not all_rows:
            hidden = ttnn.reshape(
                ttnn.embedding(self.last_indices, hidden, layout=ttnn.TILE_LAYOUT), (1, 1, self.batch_size, 6656)
            )
        logits = self.model.head(hidden, decode=True)
        if all_rows:
            return logits
        ttnn.where(self.row_mask, logits, self.prefill_logits_buffer, output_tensor=self.prefill_logits_buffer)

    def _decode_graph(self):
        logits = self.model.decode_forward(
            self.tokens,
            kv_cache=self.kv_cache,
            page_table=self.layer_page_tables if self.per_layer_page_tables else self.page_table,
            current_pos=self.positions,
            rope_indices=self.rope_indices,
        )
        ttnn.copy(logits, self.logits)
        ttnn.plus_one(self.positions, skip_negative_entries=True)
        ttnn.plus_one(self.rope_indices)

    def _sample_graph(self, force_argmax=False):
        self.sampler.decode_forward(
            self.logits,
            tt_out_tok=self.tokens,
            **({} if force_argmax else dict(k=self.k, p=self.p, temp=self.temperature)),
        )
        if not force_argmax:
            ttnn.plus_one(self.sampler._seeds)

    def _save_admission_graph(self):
        ttnn.copy(self.logits, self.prefill_logits_buffer)
        ttnn.copy(ttnn.to_layout(self.tokens, ttnn.TILE_LAYOUT), self.saved_tokens)
        seeds = ttnn.reshape(self.sampler._seeds, (1, 1, 1, self.batch_size))
        ttnn.copy(ttnn.to_layout(seeds, ttnn.TILE_LAYOUT), self.saved_seeds)

    def _restore_admission_graph(self):
        for current, saved in ((self.tokens, self.saved_tokens), (self.sampler._seeds, self.saved_seeds)):
            tiled = ttnn.to_layout(ttnn.reshape(current, (1, 1, 1, self.batch_size)), ttnn.TILE_LAYOUT)
            selected = ttnn.where(self.admission_mask, tiled, saved)
            restored = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy(ttnn.reshape(restored, current.shape), current)

    def _capture(self, key, forward):
        if key in self.traces:
            return
        print(f"CAPTURE {key}", flush=True)
        started = time.perf_counter()
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        forward()
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self.traces[key] = trace
        self.counters["captures"] += 1
        self._event("capture", key=key, trace_id=int(trace), seconds=time.perf_counter() - started)

    def _event(self, event, **fields):
        if self.event_sink is not None:
            self.event_sink(dict(event=event, time=time.time(), counters=dict(self.counters), **fields))

    def _execute(self, key):
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
        self.counters["trace_replays"] += 1
        self.counters["replay_" + (key[0] if isinstance(key, tuple) else key)] += 1

    def prepare(self):
        """Generator-wide preparation precedes the first capture, once per binding."""
        if self.prepared:
            return
        started = time.perf_counter()
        # Establish persistent decode CCL buffers before short-prefill variants.
        self._write(torch.zeros(self.batch_size, dtype=torch.int32), self.positions)
        self._decode_graph()
        for size in self.BUCKETS:
            print(f"PREPARE_PREFILL {size}", flush=True)
            self._prefill_graph(size)
            print(f"PREPARE_HEAD_LAST {size}", flush=True)
            self._head_graph(size)
            print(f"PREPARE_HEAD_ALL {size}", flush=True)
            out = self._head_graph(size, all_rows=True)
            del out
        for alignment in self.APPEND_ALIGNMENTS:
            self._prefill_graph(32, alignment)
        print("PREPARE_SAMPLING", flush=True)
        self._sample_graph()
        print("PREPARE_ARGMAX", flush=True)
        self._sample_graph(force_argmax=True)
        print("PREPARE_COPY", flush=True)
        ttnn.where(self.row_mask, self.logits, self.prefill_logits_buffer, output_tensor=self.prefill_logits_buffer)
        ttnn.copy(self.prefill_logits_buffer, self.logits)
        print("PREPARE_ADMISSION", flush=True)
        self._save_admission_graph()
        self._restore_admission_graph()
        # Short-prefill preparation can add persistent L1 CCL buffers. Compile
        # decode once more with the final allocation state before any capture.
        print("PREPARE_DECODE_FINAL", flush=True)
        self._decode_graph()
        print("PREPARE_RESET", flush=True)
        self.reset()
        ttnn.synchronize_device(self.mesh)
        self.counters["synchronizations"] += 1
        self.prepare_seconds = time.perf_counter() - started
        self._event("prepared", seconds=self.prepare_seconds, buckets=self.BUCKETS)
        # Each graph writes into storage allocated before every live trace. No
        # capture-wide exemptions or retained scratch/output allocations.
        if self.trace_reset:
            self._capture("reset", self._reset_cache_graph)
        self._capture("admission_save", self._save_admission_graph)
        self._capture("admission_restore", self._restore_admission_graph)
        for size in self.BUCKETS:
            self._capture(("prefill", size), lambda size=size: self._prefill_graph(size))
            self._capture(("head", size), lambda size=size: self._head_graph(size))
        for alignment in self.APPEND_ALIGNMENTS:
            self._capture(("prefill", 32, alignment), lambda alignment=alignment: self._prefill_graph(32, alignment))
        self._capture("decode", self._decode_graph)
        self._capture("sample", self._sample_graph)
        self._capture("argmax", lambda: self._sample_graph(force_argmax=True))
        self.prepared = True

    def _check_cache(self, cache):
        if len(cache) != len(self.kv_cache) or any(
            len(pair) != len(bound) or any(a is not b for a, b in zip(pair, bound))
            for pair, bound in zip(cache, self.kv_cache)
        ):
            raise ValueError("Bind external KV allocations when constructing the generator, before preparation")

    def _refresh_table(self, page_table, page_tables_per_layer=None):
        tables = page_tables_per_layer if self.per_layer_page_tables else [page_table]
        if tables is None or len(tables) != len(self.layer_page_tables):
            raise ValueError("Serving requires the bound per-layer page tables")
        for idx, (value, device) in enumerate(zip(tables, self.layer_page_tables)):
            representative = self._page_table_representatives[idx]
            if representative != idx:
                if value is not tables[representative] and not torch.equal(value, tables[representative]):
                    raise ValueError(f"Page mappings disagree within group at layer {idx}")
                self._layer_page_hosts[idx] = self._layer_page_hosts[representative]
                continue
            table = torch.as_tensor(value, dtype=torch.int32)
            if tuple(table.shape) != (self.batch_size, 4096):
                raise ValueError("Page table must contain one 4096-column row per configured slot")
            if table.min() < 0 or table.max() >= self.num_pages:
                raise ValueError("Physical page id is outside the bound cache pool")
            old = self._layer_page_hosts[idx]
            if old is None or not torch.equal(table, old):
                self._write(table, device, "page_table_refreshes")
                self._layer_page_hosts[idx] = table.clone()
        self._page_host = self._layer_page_hosts[0]

    def _reset_cache_graph(self):
        for layer in self.kv_cache:
            for cache in layer:
                ttnn.multiply(cache, 0, output_tensor=cache)
        for logits in (self.logits, self.prefill_logits_buffer):
            ttnn.multiply(logits, 0, output_tensor=logits)

    def reset(self, *, clear_cache=True):
        """Reset the bound pool; standalone generation may elide physical clears.

        Elision is valid only when a new prompt starts at position zero and
        overwrites every KV entry that subsequent causal attention can read.
        Low-level/scheduler callers keep the default physical reset semantics.
        """
        if clear_cache:
            if self.prepared and self.trace_reset:
                self._execute("reset")
            else:
                self._reset_cache_graph()
            self.counters["cache_clears"] += 1
        else:
            self.counters["cache_clear_elisions"] += 1
        self._write(torch.zeros((1, 1, 1, self.batch_size), dtype=torch.int32), self.tokens, "token_refreshes")
        self._write(torch.full((self.batch_size,), -1, dtype=torch.int32), self.positions, "position_refreshes")
        self._write(torch.zeros((1, self.batch_size), dtype=torch.int32), self.rope_indices, "rope_refreshes")
        self._write(torch.zeros((self.batch_size, 4096), dtype=torch.int32), self.page_table)
        self._write(torch.arange(self.batch_size, dtype=torch.int32), self.sampler._seeds)
        self._write(torch.ones(self.batch_size, dtype=torch.int32), self.k)
        self._write(torch.zeros(self.batch_size), self.p)
        self._write(torch.ones(self.batch_size), self.temperature)
        self._page_host = None
        self._layer_page_hosts = [None] * len(self.layer_page_tables)
        self._host_positions = [-1] * self.batch_size
        self.counters["resets"] += 1

    def set_sampling_params(self, *, top_k=1, top_p=0.0, temperature=1.0, seeds=None):
        """Set scalar or fixed-slot vectors at a request/scheduler boundary."""
        values = []
        for value, dtype in ((top_k, torch.int32), (top_p, torch.float32), (temperature, torch.float32)):
            value = torch.as_tensor(value, dtype=dtype).flatten()
            if value.numel() == 1:
                value = value.expand(self.batch_size).clone()
            if value.numel() != self.batch_size:
                raise ValueError("Sampling vectors must match configured fixed slots")
            values.append(value)
        k, p, temp = values
        if torch.any((k < 1) | (k > 32)) or torch.any((p < 0) | (p > 1)) or torch.any(temp <= 0):
            raise ValueError("Device sampling requires k in1..32, p in0..1 and positive temperature")
        for value, tensor in zip(values, (self.k, self.p, self.temperature)):
            self._write(value, tensor)
        if seeds is not None:
            seeds = torch.as_tensor(seeds, dtype=torch.int32).flatten()
            if seeds.numel() == 1:
                seeds = seeds.expand(self.batch_size).clone()
            if seeds.numel() != self.batch_size:
                raise ValueError("Seed vector must match configured fixed slots")
            self._write(seeds, self.sampler._seeds)

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        slots=None,
        start_pos=None,
        sample=True,
        page_tables_per_layer=None,
        sampling_params=None,
        sample_slots=None,
        read_from_device=True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported prefill options: {sorted(kwargs)}")
        self._check_cache(kv_cache)
        self._refresh_table(page_table, page_tables_per_layer)
        tokens = torch.as_tensor(tokens, dtype=torch.int32)
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        starts = [0] * len(slots) if start_pos is None else list(start_pos)
        if (
            tokens.ndim != 2
            or len(prompt_lens) != len(slots)
            or len(starts) != len(slots)
            or tokens.shape[0] != len(slots)
        ):
            raise ValueError("One logical prompt length, start position and slot is required per input row")
        if any(length > tokens.shape[1] for length in prompt_lens):
            raise ValueError("Logical prompt length exceeds supplied tokens")
        if len(set(slots)) != len(slots) or any(not 0 <= slot < self.batch_size for slot in slots):
            raise ValueError("Prefill slots must be distinct and in range")
        if any(
            length < 0 or start < 0 or start + length > self.model.config.max_position_embeddings
            for length, start in zip(prompt_lens, starts)
        ):
            raise ValueError("Prompt is outside the context contract")
        updated_slots = {slot for slot, length in zip(slots, prompt_lens) if length}
        emitted_slots = updated_slots if sample_slots is None else set(sample_slots)
        if not emitted_slots <= updated_slots:
            raise ValueError("Sampling slots must be among the prefilled rows")
        # Sampling runs across the fixed batch. Preserve every non-emitting
        # slot, including a newly admitted intermediate prefill or inactive row.
        preserve_live = len(emitted_slots) < self.batch_size
        if preserve_live:
            mask = torch.zeros((1, 1, 1, self.batch_size), dtype=torch.int32)
            for slot in emitted_slots:
                mask[0, 0, 0, slot] = 1
            self._write(mask, self.admission_mask)
            self._execute("admission_save")
        if sampling_params is not None:
            self.set_sampling_params(**sampling_params)
        results = []
        for row, (slot, length, start) in enumerate(zip(slots, prompt_lens, starts)):
            if length == 0:
                results.append(None)
                continue
            pieces = []
            mask = torch.zeros((1, 1, self.batch_size, 1))
            mask[0, 0, slot, 0] = 1
            self._write(mask, self.row_mask)
            offset = 0
            while offset < length:
                absolute = start + offset
                if absolute % 32:
                    # Unaligned append prefixes use the SAME prepared model
                    # decode graph. All other slots are inactive for this
                    # prefill-only step; caller prompt ids supply its inputs.
                    positions = torch.full((self.batch_size,), -1, dtype=torch.int32)
                    positions[slot] = absolute
                    token_row = torch.zeros((1, 1, 1, self.batch_size), dtype=torch.int32)
                    token_row[0, 0, 0, slot] = tokens[row, offset]
                    self._write(token_row, self.tokens, "prefill_token_refreshes")
                    self._write(positions, self.positions, "prefill_position_refreshes")
                    self._write(positions.clamp_min(0)[None], self.rope_indices, "prefill_rope_refreshes")
                    self._execute("decode")
                    self._event(
                        "prefill_chunk",
                        slot=slot,
                        start=absolute,
                        length=1,
                        bucket=1,
                        key="decode",
                        active_decode=[
                            i for i, p in enumerate(self._host_positions) if p >= 0 and i not in updated_slots
                        ],
                    )
                    ttnn.where(
                        self.row_mask,
                        self.logits,
                        self.prefill_logits_buffer,
                        output_tensor=self.prefill_logits_buffer,
                    )
                    if return_all_logits:
                        pieces.append(self._read_logits(self.logits)[0, 0, slot : slot + 1])
                    offset += 1
                    continue
                remaining_context = self.model.config.max_position_embeddings - absolute
                alignment = absolute % 128
                max_bucket = 32 if alignment else max(bucket for bucket in self.BUCKETS if bucket <= remaining_context)
                valid = min(max_bucket, length - offset)
                size = next(bucket for bucket in self.BUCKETS if bucket >= valid)
                state = self.prefill[size]
                ids = torch.zeros((1, size), dtype=torch.int32)
                ids[0, :valid] = tokens[row, offset : offset + valid]
                self._write(ids, state["tokens"])
                self._write(torch.arange(absolute, absolute + size, dtype=torch.int32)[None], state["rope"])
                self._write(torch.tensor([absolute], dtype=torch.int32), state["start"])
                if size == 64:
                    self._write(torch.tensor([absolute + 32], dtype=torch.int32), state["tail_start"])
                state_tables = state["table"] if self.per_layer_page_tables else [state["table"]]
                state_fills = state["fill"] if self.per_layer_page_tables else [state["fill"]]
                for table_idx in self._unique_page_table_indices:
                    table = self._layer_page_hosts[table_idx]
                    target, fill_target = state_tables[table_idx], state_fills[table_idx]
                    self._write(table[slot : slot + 1], target)
                    fill = table[slot : slot + 1, absolute // 32 : (absolute + size) // 32].clone()
                    # Padded pages are not owned by this request.
                    fill[:, (valid + 31) // 32 :] = -1
                    self._write(fill, fill_target)
                key = ("prefill", size, alignment) if alignment else ("prefill", size)
                self._execute(key)
                self._event(
                    "prefill_chunk",
                    slot=slot,
                    start=absolute,
                    length=valid,
                    bucket=size,
                    key=key,
                    active_decode=[i for i, p in enumerate(self._host_positions) if p >= 0 and i not in updated_slots],
                )
                if return_all_logits:
                    for tile in range(0, valid, 32):
                        self._write(self._slice_coordinates(tile), self.slice_start)
                        self._write(self._slice_coordinates(tile + 32, end=True), self.slice_end)
                        logits = self._head_graph(size, all_rows=True)
                        pieces.append(self._read_logits(logits).reshape(32, -1)[: min(32, valid - tile)])
                        del logits
                if offset + valid == length:
                    tile = (valid - 1) // 32 * 32
                    self._write(self._slice_coordinates(tile), self.slice_start)
                    self._write(self._slice_coordinates(tile + 32, end=True), self.slice_end)
                    self._write(
                        torch.full((1, self.batch_size), (valid - 1) % 32, dtype=torch.int32), self.last_indices
                    )
                    self._execute(("head", size))
                offset += valid
            self._host_positions[slot] = start + length
            results.append(torch.cat(pieces) if return_all_logits else None)
        self._write(torch.tensor(self._host_positions, dtype=torch.int32), self.positions, "position_refreshes")
        self._write(
            torch.tensor([max(0, x) for x in self._host_positions], dtype=torch.int32)[None],
            self.rope_indices,
            "rope_refreshes",
        )
        ttnn.copy(self.prefill_logits_buffer, self.logits)
        if sample and emitted_slots and not return_all_logits:
            self._execute("sample")
        if preserve_live:
            self._execute("admission_restore")
        if return_all_logits:
            output = torch.zeros((len(slots), max(prompt_lens), self.model.vocab_size), dtype=torch.bfloat16)
            for row, value in enumerate(results):
                if value is not None:
                    output[row, : value.shape[0]] = value
            return output
        if sample:
            return torch.tensor(self._read_tokens()) if read_from_device else self.tokens
        return self._read_logits(self.logits).reshape(self.batch_size, 1, -1) if read_from_device else self.logits

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        sample=True,
        page_tables_per_layer=None,
        reload_inputs=True,
        reload_page_table=True,
        read_from_device=True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported decode options: {sorted(kwargs)}")
        self._check_cache(kv_cache)
        if reload_inputs or reload_page_table:
            self._refresh_table(page_table, page_tables_per_layer)
        if not reload_inputs:
            if not sample:
                raise ValueError("Resident token feedback requires device sampling")
            output = self.decode_steps(1)
            return torch.tensor(self._read_tokens()) if read_from_device else output
        self._write(
            torch.as_tensor(tokens, dtype=torch.int32).reshape(1, 1, 1, self.batch_size),
            self.tokens,
            "token_refreshes",
        )
        positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(self.batch_size)
        if torch.any(positions >= self.model.config.max_position_embeddings) or torch.any(positions < -1):
            raise ValueError("Decode positions must be in context, or -1 for inactive rows")
        self._host_positions = [int(pos) + 1 if pos >= 0 else -1 for pos in positions]
        self._write(positions, self.positions, "position_refreshes")
        self._write(positions.clamp_min(0)[None], self.rope_indices, "rope_refreshes")
        self._execute("decode")
        if sample:
            self._execute("sample")
            return torch.tensor(self._read_tokens()) if read_from_device else self.tokens
        return self._read_logits(self.logits).reshape(self.batch_size, -1) if read_from_device else self.logits

    def _standalone_table(self):
        table = torch.zeros((self.batch_size, 4096), dtype=torch.int32)
        table[0] = torch.arange(4096)
        return table

    def decode_steps(self, num_steps, *, sampler_mode="sample"):
        """Queue fixed-step token-out replay over caller-prepared device state.

        Returns the persistent last-token device tensor. There are no host
        refreshes, readbacks or synchronization in this method. The caller owns
        cache admission and active slot positions, and must
        consume the output before scheduling another step. EOS-aware callers
        use the output-returning generation API instead.
        """
        if not self.prepared:
            raise RuntimeError("Prepare the generator before decode_steps")
        if num_steps < 0 or sampler_mode not in ("sample", "argmax"):
            raise ValueError("Require nonnegative steps and sample/argmax mode")
        if any(
            pos >= 0 and pos + num_steps > self.model.config.max_position_embeddings for pos in self._host_positions
        ):
            raise ValueError("Queued decode exceeds the context contract")
        for _ in range(num_steps):
            self._execute("decode")
            self._execute(sampler_mode)
        # Keep scheduler metadata current without uploading or reading device
        # positions. A later prefill in another slot must retain these positions.
        self._host_positions = [pos + num_steps if pos >= 0 else -1 for pos in self._host_positions]
        return self.tokens

    def prefill_logits(self, prompt_token_ids):
        self.reset()
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self._standalone_table(),
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=False,
        sampler_mode="sample",
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seeds=None,
        stop_on_eos=True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported generation options: {sorted(kwargs)}")
        if not enable_trace:
            raise ValueError("This generator requires traced decode")
        if not prompt_token_ids:
            raise ValueError("Generation requires a nonempty prompt")
        if sampler_mode not in ("sample", "argmax"):
            raise ValueError("Sampler mode must be sample or argmax")
        if sampler_mode == "argmax" and (top_k != 1 or top_p != 0.0):
            raise ValueError("Argmax is a greedy-only comparison mode")
        if host_sampling and (top_k != 1 or top_p != 0.0):
            raise ValueError("Host-sampling compatibility supports greedy sampling")
        if (
            max_new_tokens < 0
            or len(prompt_token_ids) + max(0, max_new_tokens - 1) > self.model.config.max_position_embeddings
        ):
            raise ValueError("Generation exceeds the context contract")
        if not max_new_tokens:
            return []
        start = time.perf_counter()
        # This standalone API owns a quiescent pool and prefills a complete new
        # slot0 prompt from position zero. Every subsequently attended KV entry
        # is overwritten; inactive rows are not outputs. Explicit reset() and
        # the caller-owned prefill/decode APIs retain their original semantics.
        self.reset(clear_cache=self.clear_cache_on_generate)
        self.set_sampling_params(top_k=top_k, top_p=top_p, temperature=temperature, seeds=seeds)
        first = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self._standalone_table(),
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            sample=not host_sampling,
        )
        predicted = int(first.reshape(-1)[0]) if not host_sampling else int(first[0, 0].argmax())
        self.last_ttft_seconds = time.perf_counter() - start
        output = [predicted]
        start = time.perf_counter()
        if next_input is None and not host_sampling and not stop_on_eos:
            # A fixed-length caller needs all tokens only when this call
            # returns. CQ0 orders each tiny D2H snapshot between sampling and
            # the next decode; no host token is read or fed back in the loop.
            pending = []
            token_device = ttnn.get_device_tensors(self.tokens)[0]
            for _ in range(max_new_tokens - 1):
                self._execute("decode")
                self._execute(sampler_mode)
                pending.append(ttnn.from_device(token_device, blocking=False, cq_id=0))
                self.counters["async_token_readbacks"] += 1
            if pending:
                ttnn.synchronize_device(self.mesh)
                self.counters["synchronizations"] += 1
                output.extend(int(ttnn.to_torch(value).flatten()[0]) for value in pending)
            self.last_decode_seconds = time.perf_counter() - start
            self._host_positions = [pos + len(output) - 1 if pos >= 0 else -1 for pos in self._host_positions]
            return output
        for step in range(max_new_tokens - 1):
            if next_input is None and stop_on_eos and predicted in self.stop_token_ids:
                break
            if next_input is not None or host_sampling:
                forced = next_input(step, predicted) if next_input is not None else predicted
                value = torch.zeros((1, 1, 1, self.batch_size), dtype=torch.int32)
                value[0, 0, 0, 0] = forced
                self._write(value, self.tokens, "token_refreshes")
            self._execute("decode")
            if host_sampling:
                predicted = int(self._read_logits(self.logits)[0, 0, 0].argmax())
            else:
                self._execute(sampler_mode)
                predicted = self._read_tokens()[0]
            output.append(predicted)
        if next_input is not None:
            next_input(max_new_tokens - 1, predicted)
        self.last_decode_seconds = time.perf_counter() - start
        self._host_positions = [pos + len(output) - 1 if pos >= 0 else -1 for pos in self._host_positions]
        return output

    def close(self):
        for key, trace in self.traces.items():
            ttnn.release_trace(self.mesh, trace)
            self.counters["teardown_releases"] += 1
            self._event("teardown_release", key=key, trace_id=int(trace))
        self.traces.clear()

    def teardown(self):
        self.close()


def build_generator(model_dir, mesh_device, **kwargs):
    model_keys = {
        key: kwargs.pop(key)
        for key in ("checkpoint", "layer_indices", "policy", "terminal_policy", "precision_config")
        if key in kwargs
    }
    model = MuseGlimmerModel(mesh_device, **model_keys)
    return MuseGlimmerGenerator(model, **kwargs)

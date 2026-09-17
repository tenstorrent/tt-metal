# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced TP4 inference with explicit cache ownership and sampling."""

import time
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

import ttnn
from models.common.sampling.tt_sampling import TTSampling
from models.demos.llama31_8b_qb2.tt.model import LlamaModel
from models.demos.llama31_8b_qb2.tt.token_history import history_program


class QB2Sampling(TTSampling):
    """Common selection algorithm over the four-device QB2 ring."""

    def _perform_all_gather(self, tensor, dim, cluster_axis, memory_config, num_links, buffer_key=None):
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=memory_config,
            chunks_per_sync=1,
            num_workers_per_link=1,
            num_buffers_per_channel=2,
        )

    def _get_force_argmax_all_gather_config(self, cluster_axis):
        # The common <8-device heuristic forces Linear even when configured
        # Ring. QB2 preserves the same qualified route for both algorithms.
        return 2, ttnn.Topology.Ring


class LlamaGenerator:
    def __init__(
        self,
        mesh_device,
        *,
        max_batch_size=1,
        cache_pages=1025,
        host_sampling=False,
        trace_prefill=True,
        trace_prefill_max_batch_size=None,
        clear_cache_on_generate=False,
        owns_kv_cache=True,
        record_token_history=True,
        specialize_single_user=True,
    ):
        if cache_pages < 1024:
            raise ValueError("Physical cache must cover the fixed 1024-entry page-table width")
        self.mesh_device = mesh_device
        self.model = LlamaModel(
            mesh_device,
            max_batch_size=max_batch_size,
        )
        self.sampling_dtype = getattr(ttnn, self.model.precision_policy["sampling_dtype"])
        if self.sampling_dtype != self.model.logits_dtype:
            raise ValueError("Sampler input and model logits dtypes must match")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.folder, local_files_only=True)
        self.max_batch_size = max_batch_size
        self.cache_pages = cache_pages
        self.host_sampling = host_sampling
        self.owns_kv_cache = owns_kv_cache
        self.record_token_history = record_token_history
        self.trace_prefill = trace_prefill
        self.trace_prefill_max_batch_size = (
            max_batch_size if trace_prefill_max_batch_size is None else trace_prefill_max_batch_size
        )
        if not 1 <= self.trace_prefill_max_batch_size <= max_batch_size:
            raise ValueError("Prefill trace batch limit must be within the supported batch size")
        self.clear_cache_on_generate = clear_cache_on_generate
        self.prefill_trace = None
        self.prefill_trace_key = None
        self.prefill_trace_inputs = []
        self._pending_prefill = None
        self.kv_cache = self.model.allocate_cache(cache_pages) if owns_kv_cache else None
        self.stats = Counter()
        self.tokens = self._device(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32),
            getattr(ttnn, self.model.precision_policy["sampling_token_dtype"]),
        )
        # Sampling replicates the complete token row on every TP device. Keep
        # a storage view so deferred serving reads transfer one replica only.
        self.token_read_view = ttnn.get_device_tensors(self.tokens)[0]
        self.token_read_range = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))
        self.history_index = self._device(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        self.token_history = ttnn.empty(
            (1, 1, self.model.supported_context, 32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.history_program = history_program(self.tokens, self.history_index, self.token_history)
        self.positions = self._device(torch.full((max_batch_size,), -1, dtype=torch.int32), ttnn.int32)
        self.rotary_positions = self._device(torch.zeros(max_batch_size, dtype=torch.int32), ttnn.int32)
        self.page_table_host = torch.zeros(max_batch_size, 1024, dtype=torch.int32)
        self.page_table = self._device(self.page_table_host, ttnn.int32)
        self.decode_execution_batch = max_batch_size
        self.decode_family_states = {max_batch_size: (self.positions, self.rotary_positions, self.page_table)}
        if specialize_single_user and max_batch_size > 1:
            self.model.prepare_single_user_decode()
            self.decode_family_states[1] = (
                self._device(torch.full((1,), -1, dtype=torch.int32), ttnn.int32),
                self._device(torch.zeros(1, dtype=torch.int32), ttnn.int32),
                self._device(self.page_table_host[:1], ttnn.int32),
            )
        self.logits = ttnn.empty(
            (1, 1, 32, self.model.padded_vocab_size // 4),
            dtype=self.sampling_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.prefill_zero = ttnn.zeros(
            (1, 1, 1, self.model.padded_vocab_size // 4),
            dtype=self.sampling_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        args = SimpleNamespace(
            vocab_size=self.model.vocab_size,
            padded_vocab_size=self.model.padded_vocab_size,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=mesh_device.shape,
            pad_logits_to_power_of_2=True,
            model_config={
                "SAMPLING_AG_CONFIG": {"allow_force_argmax": False, "num_links": 2, "topology": ttnn.Topology.Ring}
            },
        )
        self.sampler = QB2Sampling(mesh_device, self.model.ccl, args)
        self.model_trace = None
        self.model_traces = {}
        self.compiled_decode_graphs = set()
        self._cache_misses_allowed = True
        self.sampling_traces = {}
        self._decode_cache = None
        self.trace_cache = None
        self.trace_cache_signature = None
        self.prefill_specializations = set()
        self.history_read_lengths = set()
        self.sampling_mode = "split"
        self.last_perf = {}
        # In-place reset still creates a cached binary on its first use.
        if self.owns_kv_cache:
            self._zero_cache()

    def _device(self, value, dtype):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _copy(self, value, target, counter):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=target.dtype,
            layout=target.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, target)
        self.stats[counter] += 1

    def refresh_page_table(self, page_table):
        """The caller may change mappings; unchanged tables trigger no copy."""
        table = torch.as_tensor(page_table, dtype=torch.int32)
        if table.ndim != 2 or table.shape[0] != self.max_batch_size or table.shape[1] > 1024:
            raise ValueError("Expected one page-table row per fixed slot, at most 1024 pages")
        if (table < 0).any():
            raise ValueError("Page IDs must be nonnegative; inactive rows use position -1")
        padded = torch.nn.functional.pad(table, (0, 1024 - table.shape[1]))
        if not torch.equal(padded, self.page_table_host):
            self._copy(padded, self.page_table, "page_table_refreshes")
            self.page_table_host = padded.clone()
            if self.max_batch_size > 1 and 1 in self.decode_family_states:
                self._copy(padded[:1], self.decode_family_states[1][2], "family_page_table_refreshes")

    def refresh_decode_inputs(self, tokens, start_pos, *, page_table=None):
        ids = torch.as_tensor(tokens, dtype=torch.int32).reshape(-1)
        positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
        if len(ids) != self.max_batch_size or len(positions) != self.max_batch_size:
            raise ValueError("Decode state must name every fixed slot")
        if ((positions < -1) | (positions >= self.model.supported_context)).any():
            raise ValueError("Decode positions must be -1 (inactive) or within the supported context")
        if ((ids < 0) | (ids >= self.model.vocab_size)).any():
            raise ValueError("Token outside the real vocabulary")
        values = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        values.flatten()[: len(ids)] = ids
        self._copy(values, self.tokens, "token_refreshes")
        self._copy(positions, self.positions, "position_refreshes")
        self._copy(positions.clamp_min(0), self.rotary_positions, "rotary_refreshes")
        if self.max_batch_size > 1 and 1 in self.decode_family_states:
            pos1, rotary1, _ = self.decode_family_states[1]
            self._copy(positions[:1], pos1, "family_position_refreshes")
            self._copy(positions[:1].clamp_min(0), rotary1, "family_rotary_refreshes")
            self.decode_execution_batch = (
                1 if positions[0] >= 0 and (positions[1:] == -1).all() else self.max_batch_size
            )
            self.model_trace = self.model_traces.get(self.decode_execution_batch)
        if page_table is not None:
            self.refresh_page_table(page_table)

    def set_sampling(self, *, top_k=1, top_p=0.0, temperature=1.0, seed=0, force_argmax=False):
        """Request-boundary parameter refresh; sampling and seed advance are traced."""

        def rows(value, dtype):
            tensor = torch.as_tensor(value, dtype=dtype).flatten()
            if tensor.numel() == 1:
                return tensor.repeat(32)
            if tensor.numel() not in (self.max_batch_size, 32):
                raise ValueError("Sampling parameters must be scalar or per fixed slot")
            return torch.nn.functional.pad(tensor, (0, 32 - tensor.numel()), value=tensor[0].item())

        k, p, temp = rows(top_k, torch.int64), rows(top_p, torch.float32), rows(temperature, torch.float32)
        if (temp < 0).any():
            raise ValueError("Temperature must be nonnegative")
        k = torch.where(temp == 0, 1, k)
        p = torch.where(temp == 0, 0.0, p)
        temp = torch.where(temp == 0, 1.0, temp)
        if force_argmax and not ((k == 1).all() and ((p == 0) | (p == 1)).all() and (temp == 1).all()):
            raise ValueError("Argmax comparison must be semantically greedy")
        # The TT kernel multiplies logits by temp. Match the common sampling
        # formatter: public temperature T therefore binds inverse temperature.
        self.sampler.reset_params(k=k, p=p, temp=temp.reciprocal())
        if seed is not None:
            self._copy(rows(seed, torch.int64).to(torch.int32), self.sampler.seeds_tt_tensor, "seed_refreshes")
        self.sampler._force_argmax_sampling = force_argmax
        self.sampling_mode = "argmax" if force_argmax else "split"

    def _model_step(self, cache, execution_batch=None):
        batch = self.decode_execution_batch if execution_batch is None else execution_batch
        positions, rotary, table = self.decode_family_states[batch]
        logits = self.model.decode(
            self.tokens,
            current_pos=positions,
            rotary_pos=rotary,
            page_table=table,
            kv_cache=cache,
            execution_batch=batch,
        )
        ttnn.copy(logits, self.logits)
        ttnn.plus_one(positions, skip_negative_entries=True)
        ttnn.plus_one(rotary)
        if batch != self.max_batch_size:
            # Keep the public fixed-slot state coherent for callers and EOS.
            ttnn.plus_one(self.positions, skip_negative_entries=True)
            ttnn.plus_one(self.rotary_positions)

    def _sample_step(self):
        output = self.sampler(self.logits, tt_out_tok=self.tokens)
        ttnn.plus_one(self.sampler.seeds_tt_tensor)
        if self.record_token_history:
            ttnn.generic_op([self.tokens, self.history_index, self.token_history], self.history_program)
        return output

    @contextmanager
    def _forbid_cache_misses(self, enabled=True):
        previous = self._cache_misses_allowed
        if enabled:
            self.mesh_device.set_program_cache_misses_allowed(False)
            self._cache_misses_allowed = False
        try:
            yield
        finally:
            if enabled:
                self.mesh_device.set_program_cache_misses_allowed(previous)
                self._cache_misses_allowed = previous

    def _capture_trace(self, operation):
        with self._forbid_cache_misses():
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            try:
                operation()
            except BaseException:
                ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                ttnn.release_trace(self.mesh_device, trace_id)
                raise
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        return trace_id

    def _capture_decode_graphs(self, cache, signature):
        try:
            batch = self.decode_execution_batch
            if batch not in self.model_traces:
                self.model_traces[batch] = self._capture_trace(lambda: self._model_step(cache, batch))
            if self.sampling_mode not in self.sampling_traces:
                self.sampling_traces[self.sampling_mode] = self._capture_trace(self._sample_step)
        except BaseException:
            self.release_traces(clear_prefill=False)
            raise
        self.model_trace = self.model_traces[batch]
        self.trace_cache = tuple(tuple(pair) for pair in cache)
        self.trace_cache_signature = signature

    def _prefill_input_identity(self):
        return tuple(
            (slot, ids.buffer_unique_id(), last.buffer_unique_id()) for slot, ids, last in self.prefill_trace_inputs
        )

    def _capture_prefill_trace(self, cache):
        try:
            self.prefill_trace = self._capture_trace(lambda: self._prefill_device(cache))
        except BaseException:
            self.release_traces(clear_prefill=False)
            raise
        self.stats["prefill_trace_captures"] += 1

    def prepare_traces(self, *, kv_cache=None):
        """Prepare the selected decode family and materialize any pending prefill."""
        cache = self.kv_cache if kv_cache is None else kv_cache
        self._validate_cache(cache)
        signature = self._cache_signature(cache)
        pending = self._pending_prefill
        if pending is not None and pending["key"][2] != signature:
            # A caller may replace its KV arena between prefill and decode.
            # Never record or later replay the obsolete pending cache binding.
            self.release_traces()
            pending = None
        try:
            if pending is not None and (
                pending["key"] != self.prefill_trace_key
                or pending["inputs"] != self._prefill_input_identity()
                or self._cache_signature(pending["cache"]) != signature
            ):
                raise RuntimeError("Pending prefill inputs or cache binding changed")
            self._prepare_decode_traces(cache, signature)
            if pending is not None:
                # Recording does not execute the old prompt or advance state.
                # Current scheduler family and sampler may differ from prefill.
                self._capture_prefill_trace(cache)
                self._pending_prefill = None
                self.stats["pending_prefill_materializations"] += 1
        except BaseException:
            self.release_traces()
            raise

    def _prepare_decode_traces(self, cache, signature):
        selected = self.model_traces.get(self.decode_execution_batch)
        if (
            self.trace_cache_signature == signature
            and selected is not None
            and self.sampling_mode in self.sampling_traces
        ):
            self.model_trace = selected
            return
        compiled_key = (signature, self.sampling_mode, tuple(self.decode_family_states))
        compiled = compiled_key in self.compiled_decode_graphs
        if compiled and self.trace_cache_signature == signature:
            # Both families were compiled before any trace. Append only a
            # missing handle while retaining existing model/sampler/prefill owners.
            with self._forbid_cache_misses():
                self._capture_decode_graphs(cache, signature)
            self.stats["decode_graph_appends"] += 1
            return
        # New bindings/modes can allocate persistent binaries or semaphores.
        # Release every old handle before the original full inactive warmup.
        self.release_traces(clear_prefill=False)
        if compiled:
            with self._forbid_cache_misses():
                self._capture_decode_graphs(cache, signature)
            self.stats["decode_graph_recaptures"] += 1
            return
        saved_logits = ttnn.clone(self.logits)
        # Warmup mutates cache/token/position/seed state; save only small state
        # here. This is setup, outside the token loop and performance window.
        state_tensors = [self.tokens, self.sampler.seeds_tt_tensor, self.history_index]
        for positions, rotary, _ in self.decode_family_states.values():
            state_tensors.extend((positions, rotary))
        state = [self._read_replicated(t).clone() for t in state_tensors]
        # Preserve caller-owned cached prefixes during compilation. Negative
        # cache positions skip updates/SDPA; RoPE still needs valid indices.
        for batch, (positions, rotary, _) in self.decode_family_states.items():
            self._copy(torch.full((batch,), -1, dtype=torch.int32), positions, "capture_inactive_warmup")
            self._copy(torch.zeros(batch, dtype=torch.int32), rotary, "capture_inactive_warmup")
        # Compile both families before either trace can own allocator addresses.
        for batch in self.decode_family_states:
            self._model_step(cache, batch)
        self._sample_step()
        ttnn.synchronize_device(self.mesh_device)
        self.stats["capture_synchronizations"] += 1
        for host, target in zip(state, state_tensors):
            self._copy(host, target, "capture_state_restores")
        self._capture_decode_graphs(cache, signature)
        ttnn.copy(saved_logits, self.logits)
        del saved_logits
        self.compiled_decode_graphs.add(compiled_key)

    @staticmethod
    def _cache_signature(cache):
        return tuple(
            tuple(
                (tensor.buffer_unique_id(), tuple(tensor.shape), str(tensor.dtype), str(tensor.memory_config()))
                for tensor in pair
            )
            for pair in cache
        )

    def _validate_cache(self, cache):
        if len(cache) != self.model.num_layers or any(len(pair) != 2 for pair in cache):
            raise ValueError("One K/V cache pair per loaded layer is required")
        largest_page = int(self.page_table_host.max())
        for pair in cache:
            for tensor in pair:
                if (
                    tuple(tensor.shape)[1:] != (2, 128, 128)
                    or tensor.shape[0] < 1024
                    or largest_page >= tensor.shape[0]
                    or tensor.dtype != self.model.cache_dtype
                    or tensor.layout != ttnn.TILE_LAYOUT
                ):
                    raise ValueError(
                        "Expected TP4 policy-matched tiled cache [pages>=1024,2,128,128] covering every page ID"
                    )

    def replay_decode(self, *, sample=True):
        if self._pending_prefill is not None:
            self.prepare_traces(kv_cache=self._pending_prefill["cache"])
        if self.model_trace is None or (sample and self.sampling_mode not in self.sampling_traces):
            raise RuntimeError("Prepare model and sampling traces before replay")
        ttnn.execute_trace(self.mesh_device, self.model_trace, cq_id=0, blocking=False)
        self.stats["model_trace_replays"] += 1
        if sample:
            self.replay_sampling()
        return self.tokens if sample else self.logits

    def replay_sampling(self):
        ttnn.execute_trace(self.mesh_device, self.sampling_traces[self.sampling_mode], cq_id=0, blocking=False)
        self.stats["sampling_trace_replays"] += 1

    def _read_replicated(self, tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])

    def read_tokens(self):
        self.stats["token_readbacks"] += 1
        return self._read_replicated(self.tokens).flatten()[: self.max_batch_size].to(torch.int64)

    def read_token_history(self, count):
        self.stats["history_readbacks"] += 1
        part = self.token_history[:, :, :count, :]
        return self._read_replicated(part).reshape(count, 32)[:, : self.max_batch_size].to(torch.int64)

    def read_logits(self, tensor=None):
        """Explicit host-logit boundary for readiness and compatibility only."""
        self.stats["full_logits_readbacks"] += 1
        tensor = self.logits if tensor is None else tensor
        return torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)], dim=-1)[
            ..., : self.model.vocab_size
        ]

    def _counter_delta(self, before):
        delta = self.stats - before
        for name in (
            "model_trace_replays",
            "sampling_trace_replays",
            "token_refreshes",
            "position_refreshes",
            "rotary_refreshes",
            "page_table_refreshes",
            "cache_resets",
            "capture_synchronizations",
            "token_readbacks",
            "history_readbacks",
            "full_logits_readbacks",
            "teacher_or_host_token_refreshes",
            "eos_position_refreshes",
        ):
            delta.setdefault(name, 0)
        return dict(delta)

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        slots=None,
        sample_on_device=True,
        **kwargs,
    ):
        """Mixed logical prompt lengths; slots select caller-owned cache rows."""
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        trace_lengths = self._prefill_trace_lengths(prompt_lens, return_all_logits)
        program_lengths = tuple(prompt_lens) if trace_lengths is None else trace_lengths
        key = (
            trace_lengths is not None,
            program_lengths,
            tuple(slots),
            return_all_logits,
            self._cache_signature(kv_cache),
        )
        known = key in self.prefill_specializations
        if not known:
            self.release_traces()
        elif (
            not return_all_logits
            and (key[-1], self.sampling_mode, tuple(self.decode_family_states)) not in self.compiled_decode_graphs
        ):
            # set_sampling can select an unseen mode on a known prefill. Its
            # full warmup must not inherit a miss-forbidden live-trace scope.
            self.release_traces(clear_prefill=False)
        # A missing selected alias can coexist with other retained families.
        # Protect every live owner, not only the currently selected handle.
        forbid_misses = known and bool(self.model_traces or self.sampling_traces or self.prefill_trace is not None)
        with self._forbid_cache_misses(forbid_misses):
            result = self._prefill_forward(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                return_all_logits=return_all_logits,
                slots=slots,
            )
        self.prefill_specializations.add(key)
        if return_all_logits:
            return result
        if sample_on_device:
            if result is True:
                # Only this call produced real eager logits and deferred its
                # already-compiled captures. Draw token one exactly once.
                try:
                    with self._forbid_cache_misses():
                        self._sample_step()
                except BaseException:
                    self.release_traces()
                    raise
                self.stats["prefill_eager_samples"] += 1
            else:
                self.prepare_traces(kv_cache=kv_cache)
                self.replay_sampling()
            return self.read_tokens()[slots]
        if result is True:
            # Host sampling never carries capture work past its logits return.
            self.prepare_traces(kv_cache=kv_cache)
        return self.read_logits().reshape(32, -1)[slots, None, :]

    def _prefill_forward(self, tokens, *, page_table, kv_cache, prompt_lens, return_all_logits, slots):
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        if tokens.ndim != 2 or len(prompt_lens) != tokens.shape[0] or len(slots) != len(prompt_lens):
            raise ValueError("Tokens, lengths and slots must describe the same prompt batch")
        if len(set(slots)) != len(slots) or any(s < 0 or s >= self.max_batch_size for s in slots):
            raise ValueError("Slots must be distinct valid fixed rows")
        self.refresh_page_table(page_table)
        self._validate_cache(kv_cache)
        for row, length in enumerate(prompt_lens):
            if not 1 <= length <= self.model.supported_context or length > tokens.shape[1]:
                raise ValueError("Invalid logical prompt length")
            ids = tokens[row, :length]
            if ((ids < 0) | (ids >= self.model.vocab_size)).any():
                raise ValueError("Token outside the real vocabulary")
        if self._prefill_trace_lengths(prompt_lens, return_all_logits) is not None:
            return self._traced_prefill(tokens, prompt_lens, slots, kv_cache)
        outputs = []
        final_rows = {}
        for row, (slot, length) in enumerate(zip(slots, prompt_lens)):
            ids = tokens[row : row + 1, :length]
            tt_ids = self._device(ids.to(torch.int32), ttnn.uint32)
            table = self.page_table[slot : slot + 1, :]
            hidden = self.model.prefill_hidden(tt_ids, page_table=table, kv_cache=kv_cache)
            if return_all_logits:
                pieces = []
                for start in range(0, length, 128):
                    part = hidden[:, :, start : min(start + 128, length), :]
                    logits = self.model.prefill_head(part)
                    pieces.append(self.read_logits(logits).reshape(1, part.shape[2], -1))
                outputs.append(torch.cat(pieces, dim=1))
            else:
                last = hidden[:, :, length - 1 : length, :]
                logits = self.model.prefill_head(last)
                final_rows[slot] = logits
        if return_all_logits:
            size = max(prompt_lens)
            return torch.cat([torch.nn.functional.pad(o, (0, 0, 0, size - o.shape[1])) for o in outputs], dim=0)
        zero = ttnn.zeros(
            (1, 1, 1, self.model.padded_vocab_size // 4),
            dtype=self.sampling_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        combined = self._pack_prefill_logits(final_rows, zero)
        ttnn.copy(combined, self.logits)
        # No prefill allocation survives into decode/sampling replay. Their
        # values have reached the pre-capture persistent logits allocation.
        del combined, final_rows, zero, logits, last, hidden, table, tt_ids
        return None

    def _prefill_trace_lengths(self, lengths, return_all_logits=False):
        if (
            self.trace_prefill
            and not return_all_logits
            and len(lengths) <= self.trace_prefill_max_batch_size
            and sum(lengths) <= 1024
        ):
            return tuple((length + 31) // 32 * 32 for length in lengths)
        return None

    def _pack_prefill_logits(self, rows, zero):
        """Unpad each real row and the shared inactive row only once."""
        if self.sampling_dtype != ttnn.bfloat16:
            return ttnn.concat([rows.get(slot, zero) for slot in range(32)], dim=2)

        def unpad(tensor):
            ends = ttnn.Shape(tuple(size - 1 for size in tensor.shape))
            return ttnn.untilize_with_unpadding(tensor, ends)

        real = {slot: unpad(tensor) for slot, tensor in rows.items()}
        zero_rm = unpad(zero) if len(real) < 32 else None
        joined = ttnn.concat(
            [real.get(slot, zero_rm) for slot in range(32)], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.tilize_with_val_padding(
            joined, self.logits.padded_shape, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _prefill_device(self, kv_cache):
        rows = {}
        for slot, ids, last_index in self.prefill_trace_inputs:
            table = self.page_table[slot : slot + 1, :]
            hidden = self.model.prefill_hidden(ids, page_table=table, kv_cache=kv_cache)
            # The bucket is static; the final logical row changes at replay.
            last = ttnn.embedding(last_index, hidden, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            last = ttnn.reshape(last, (1, 1, 1, hidden.shape[-1]))
            rows[slot] = self.model.prefill_head(last)
        ttnn.copy(self._pack_prefill_logits(rows, self.prefill_zero), self.logits)

    def _traced_prefill(self, tokens, lengths, slots, kv_cache):
        buckets = self._prefill_trace_lengths(lengths)
        key = (buckets, tuple(slots), self._cache_signature(kv_cache))
        # Rounding to32 stays within the same allocated128-token cache pages.
        # Future padding cannot affect a valid causal prefix. Decode continues
        # at the caller's logical length and overwrites these future entries.
        padded_tokens = [
            torch.nn.functional.pad(tokens[row : row + 1, :length].to(torch.int32), (0, bucket - length))
            for row, (length, bucket) in enumerate(zip(lengths, buckets))
        ]
        last_indices = [torch.tensor([[length - 1]], dtype=torch.int32) for length in lengths]
        eager_executed = key != self.prefill_trace_key
        if eager_executed:
            self.release_traces()
            self.prefill_trace_inputs = [
                (slot, self._device(ids, ttnn.uint32), self._device(index, ttnn.uint32))
                for slot, ids, index in zip(slots, padded_tokens, last_indices)
            ]
            self.prefill_trace_key = key
            # Warm both phases before capturing either. No first-use program
            # binaries or persistent inputs may appear behind a live trace.
            self._prefill_device(kv_cache)
            self.stats["prefill_eager_executions"] += 1
        else:
            for row, (slot, ids, last_index) in enumerate(self.prefill_trace_inputs):
                self._copy(padded_tokens[row], ids, "prefill_token_refreshes")
                self._copy(last_indices[row], last_index, "prefill_last_row_refreshes")
        compiled_key = (key[2], self.sampling_mode, tuple(self.decode_family_states))
        if eager_executed and compiled_key in self.compiled_decode_graphs:
            self._pending_prefill = {
                "key": key,
                "inputs": self._prefill_input_identity(),
                "cache": tuple(tuple(pair) for pair in kv_cache),
            }
            self.stats["prefill_first_replays_skipped"] += 1
            return True
        self.prepare_traces(kv_cache=kv_cache)
        if self.prefill_trace is None:
            self._capture_prefill_trace(kv_cache)
        if eager_executed:
            # Full first-use warmup restores these real eager prefill logits.
            # Capture alone has not executed or consumed them.
            self.stats["prefill_first_replays_skipped"] += 1
            return None
        ttnn.execute_trace(self.mesh_device, self.prefill_trace, cq_id=0, blocking=False)
        self.stats["prefill_trace_replays"] += 1
        return None

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        sample_on_device=True,
        reset_batch=True,
        reload_page_table=False,
        read_from_device=True,
        **kwargs,
    ):
        # Resident decode owns tokens and positions. Page growth updates only
        # the mapping, preserving device feedback from queued decode steps.
        if reset_batch or not sample_on_device:
            self.refresh_decode_inputs(tokens, start_pos, page_table=page_table)
        elif reload_page_table:
            self.refresh_page_table(page_table)
        if (
            self._pending_prefill is not None
            or self.model_trace is None
            or self._decode_cache is not kv_cache
            or (sample_on_device and self.sampling_mode not in self.sampling_traces)
        ):
            self.prepare_traces(kv_cache=kv_cache)
            self._decode_cache = kv_cache
        output = self.replay_decode(sample=sample_on_device)
        if not read_from_device:
            return output
        return self.process_decode_output_host(output, is_tokens=sample_on_device)

    def read_decode_output(self, output, *, async_read=False):
        """Enqueue the caller-visible output read before the next trace writes it."""
        source = self.token_read_view if output is self.tokens else output
        host = ttnn.from_device(source, blocking=not async_read)
        self.stats["token_readbacks" if output is self.tokens else "full_logits_readbacks"] += 1
        if output is self.tokens:
            self.stats["token_readback_replicas"] += 1
        if async_read:
            # Only the read replica must be host-visible. Every device retains
            # CQ0 ordering for model/sampler feedback and the next replay.
            read_range = self.token_read_range if output is self.tokens else None
            return host, [ttnn.record_event(self.mesh_device, 0, device_range=read_range)]
        return host

    def process_decode_output_host(self, output, *, is_tokens=True):
        if is_tokens:
            return self._read_replicated(output).flatten()[: self.max_batch_size].to(torch.int64)
        return self.read_logits(output).reshape(32, -1)[: self.max_batch_size]

    def _allocate_pages(self, lengths):
        table = torch.zeros_like(self.page_table_host)
        next_page = 1  # Page zero is a safe mapping for masked/inactive tails.
        for slot, length in enumerate(lengths):
            pages = (length + 127) // 128
            if length > self.model.supported_context or next_page + pages > self.cache_pages:
                raise ValueError("Request exceeds context or this cache arena; supply a larger physical cache")
            table[slot, :pages] = torch.arange(next_page, next_page + pages, dtype=torch.int32)
            next_page += pages
        return table

    def prefill_logits(self, prompt_token_ids):
        self.reset()
        prompt = list(prompt_token_ids)
        table = self._allocate_pages([len(prompt)])
        return self.prefill_forward(
            torch.tensor([prompt]),
            page_table=table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt)],
            return_all_logits=True,
        )

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=None,
        stop_on_eos=False,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=0,
        readback_each_token=False,
        **kwargs,
    ):
        start = time.perf_counter()
        before = self.stats.copy()
        if not enable_trace:
            raise ValueError("This generator requires traced decode")
        if not len(prompt_token_ids):
            raise ValueError("Prompt must contain at least one token")
        prompts = (
            [list(prompt_token_ids)] if isinstance(prompt_token_ids[0], int) else [list(p) for p in prompt_token_ids]
        )
        batch = len(prompts)
        if batch > self.max_batch_size or max_new_tokens < 1:
            raise ValueError("Invalid batch or generation length")
        if next_input is not None and batch != 1:
            raise ValueError("Readiness teacher-forcing callback describes one request")
        lengths = [len(p) for p in prompts]
        if any(length < 1 for length in lengths):
            raise ValueError("Every prompt must contain at least one token")
        table = self._allocate_pages([s + max_new_tokens - 1 for s in lengths])
        compatibility = self.host_sampling if host_sampling is None else host_sampling
        if compatibility and not (torch.as_tensor(top_k).eq(1).all()):
            raise ValueError(
                "Host-sampling compatibility implements greedy selection; use device sampling for top-k/top-p"
            )
        if max_new_tokens not in self.history_read_lengths:
            # Slice programs own persistent binaries. Warm each read shape
            # before capture, including the final batched output boundary.
            self.release_traces()
            part = self.token_history[:, :, :max_new_tokens, :]
            del part
            self.history_read_lengths.add(max_new_tokens)
        self.reset(clear_cache=self.clear_cache_on_generate)
        self.set_sampling(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
        # Compile with valid pages/positions. Prefill overwrites every visible
        # cache prefix; causal masking excludes later warmup writes.
        positions = torch.full((self.max_batch_size,), -1, dtype=torch.int32)
        positions[:batch] = torch.tensor(lengths).clamp_max(self.model.supported_context - 1)
        self.refresh_decode_inputs(torch.zeros(self.max_batch_size, dtype=torch.int32), positions, page_table=table)
        tokens = torch.zeros(batch, max(lengths), dtype=torch.int64)
        for row, prompt in enumerate(prompts):
            tokens[row, : len(prompt)] = torch.tensor(prompt)
        setup_ms = (time.perf_counter() - start) * 1000
        first = self.prefill_forward(
            tokens, page_table=table, kv_cache=self.kv_cache, prompt_lens=lengths, sample_on_device=not compatibility
        )
        if compatibility:
            first = first[:, 0, :].argmax(dim=-1)
            self.prepare_traces()
        ttft = time.perf_counter() - start
        outputs = [[int(t)] for t in first]
        stop_ids = self.model.config.eos_token_id
        stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids
        finished = [stop_on_eos and o[-1] in stop_ids for o in outputs]
        prediction = int(first[0])
        override = next_input(0, prediction) if next_input else None
        steady_before = self.stats.copy()
        deactivated = set()
        decode_start = time.perf_counter()
        deferred_readback = self.record_token_history and not (
            next_input is not None or compatibility or stop_on_eos or readback_each_token
        )
        for step in range(1, max_new_tokens):
            if all(finished):
                break
            # EOS is a scheduler event: deactivate only newly finished slots.
            # Fixed-step generation performs no host position refreshes.
            newly_finished = [row for row, done in enumerate(finished) if done and row not in deactivated]
            if newly_finished:
                positions = self._read_replicated(self.positions).to(torch.int32)
                positions[newly_finished] = -1
                self._copy(positions, self.positions, "eos_position_refreshes")
                deactivated.update(newly_finished)
            if next_input is not None or compatibility:
                ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                ids.flatten()[:batch] = torch.tensor([override] if next_input else [o[-1] for o in outputs])
                self._copy(ids, self.tokens, "teacher_or_host_token_refreshes")
            self.replay_decode(sample=not compatibility)
            if deferred_readback:
                continue
            predicted = (
                self.read_logits().reshape(32, -1)[:batch].argmax(dim=-1)
                if compatibility
                else self.read_tokens()[:batch]
            )
            for row, token in enumerate(predicted):
                if not finished[row]:
                    outputs[row].append(int(token))
                    finished[row] = stop_on_eos and int(token) in stop_ids
            if next_input:
                override = next_input(step, int(predicted[0]))
        if deferred_readback and max_new_tokens > 1:
            history = self.read_token_history(max_new_tokens)
            outputs = history[:, :batch].T.tolist()
        elapsed = time.perf_counter() - decode_start
        self.last_perf = {
            "ttft_ms": ttft * 1000,
            "decode_ms": elapsed * 1000,
            "setup_ms": setup_ms,
            "decode_tokens": len(outputs[0]) - 1,
            "batch": batch,
            "prompt_lens": lengths,
            "teacher_forcing": next_input is not None,
            "host_sampling": compatibility,
            "readback_each_token": not deferred_readback,
            "counters": self._counter_delta(before),
            "steady_counters": self._counter_delta(steady_before),
        }
        return outputs[0] if batch == 1 else outputs

    def _zero_cache(self):
        if not self.owns_kv_cache:
            raise RuntimeError("Serving KV cache belongs to vLLM; standalone clearing is unavailable")
        for pair in self.kv_cache:
            for tensor in pair:
                ttnn.multiply(tensor, 0.0, output_tensor=tensor)
        self.stats["cache_resets"] += 1

    def reset(self, *, clear_cache=True):
        """Reset owned request state; physical cache clearing is explicit.

        A new paged prompt overwrites every visible prefix entry before decode.
        Causal attention masks later positions and page zero remains a safe tail.
        External callers retain the historical clearing reset by default.
        """
        if clear_cache:
            self._zero_cache()
        self._copy(torch.zeros(1, 1, 1, 32, dtype=torch.int32), self.history_index, "history_index_refreshes")
        self.refresh_decode_inputs(
            torch.zeros(self.max_batch_size, dtype=torch.int32),
            torch.full((self.max_batch_size,), -1, dtype=torch.int32),
            page_table=torch.zeros_like(self.page_table_host),
        )

    def release_traces(self, *, clear_prefill=True):
        if self.prefill_trace is not None:
            ttnn.release_trace(self.mesh_device, self.prefill_trace)
        self.prefill_trace = None
        if clear_prefill:
            self._pending_prefill = None
            self.prefill_trace_key = None
            self.prefill_trace_inputs = []
        for trace in self.sampling_traces.values():
            ttnn.release_trace(self.mesh_device, trace)
        self.sampling_traces.clear()
        for trace in self.model_traces.values():
            ttnn.release_trace(self.mesh_device, trace)
        self.model_traces.clear()
        self.model_trace = None
        self.trace_cache = None
        self.trace_cache_signature = None

    def teardown(self):
        self.release_traces()

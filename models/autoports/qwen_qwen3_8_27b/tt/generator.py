# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving state and canonical split sampling for the TP4 Qwen text model."""

import time
from collections import Counter
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import QwenModel
from models.common.readiness_check.contract import Generator
from models.common.sampling.tt_sampling import TTSampling


class QwenGenerator(Generator):
    def __init__(self, model, *, host_sampling=False, sampling_strategy="split"):
        self.model, self.mesh = model, model.mesh
        self.tokenizer = AutoTokenizer.from_pretrained(model.snapshot, local_files_only=True)
        self.host_sampling = host_sampling
        self.seed = 0
        args = SimpleNamespace(
            vocab_size=model.config.vocab_size,
            padded_vocab_size=model.config.vocab_size,
            cluster_shape=(1, 4),
            max_batch_size=32,
            max_top_k=32,
            pad_logits_to_power_of_2=False,
        )
        if sampling_strategy not in ("split", "argmax"):
            raise ValueError("Unknown common sampling strategy")
        args.model_config = {
            "SAMPLING_AG_CONFIG": dict(
                allow_force_argmax=sampling_strategy == "argmax", num_links=2, topology=ttnn.Topology.Ring
            )
        }
        self.sampling_strategy = sampling_strategy
        self.active_slots = None
        self.reset_active_slots = False
        self.rope_indices = None
        self.sampler = TTSampling(self.mesh, model.ccl, args)
        self.remaining_steps = None
        self.owns_cache = True
        self.cache = None
        self.trace = self.sample_trace = None
        self.tokens = model.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.positions = self.page_table = self.logits = None
        self.prefill_signatures = set()
        self.page_host = None
        self.counters = Counter()
        self.last_perf = {}

    def _copy(self, host, target, counter):
        source = ttnn.from_torch(host.contiguous(), dtype=target.dtype, layout=target.layout)
        ttnn.copy_host_to_device_tensor(source, target)
        self.counters[counter] += 1

    def _read_tokens(self):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).reshape(-1).long()

    def _host_logits(self, logits):
        self.counters["full_logits_readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()

    def _release_traces(self):
        for trace in (self.trace, self.sample_trace):
            if trace is not None:
                ttnn.release_trace(self.mesh, trace)
        self.trace = self.sample_trace = None

    def _ensure_cache(self, batch, capacity):
        if (
            self.owns_cache
            and self.cache is not None
            and self.cache.batch_size == batch
            and self.cache.capacity >= capacity
        ):
            return self.cache
        self._release_traces()
        self.prefill_signatures.clear()
        self.remaining_steps = None
        self.owns_cache = True
        self.cache = self.model.allocate_cache(
            batch_size=batch, capacity=min(self.model.context, ((capacity + 31) // 32) * 32)
        )
        self.positions = self.model.upload(
            torch.zeros(batch, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        pages = self.cache.num_pages // batch
        self.page_host = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
        self.page_table = self.model.upload(self.page_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.rope_indices = self.model.upload(
            torch.zeros(batch, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.active_slots = None
        self.counters["page_table_allocations"] += 1
        return self.cache

    def _refresh_table(self, table):
        if isinstance(table, ttnn.Tensor):
            if table is not self.page_table:
                raise ValueError("Device page tables must retain their bound tensor identity; update in place")
            self.page_host = None  # Caller may have mutated the device table in place.
            return
        table = torch.as_tensor(table, dtype=torch.int32)
        if (table < 0).any() or (table >= self.cache.num_pages).any():
            raise ValueError("Page IDs lie outside the bound physical cache")
        if self.page_host is not None and torch.equal(table, self.page_host):
            return
        if tuple(table.shape) != tuple(self.page_table.shape):
            raise ValueError("Page-table geometry changes require a new cache binding")
        self._copy(table, self.page_table, "page_table_refreshes")
        self.page_host = table.clone()

    def reset(self):
        if self.cache is not None:
            self.model.reset_cache(self.cache)
        if self.positions is not None:
            self._copy(torch.zeros(self.cache.batch_size, dtype=torch.int32), self.positions, "position_refreshes")
        self._copy(torch.zeros(1, 1, 1, 32, dtype=torch.int32), self.tokens, "token_refreshes")
        if self.rope_indices is not None:
            self._copy(torch.zeros(self.cache.batch_size, dtype=torch.int32), self.rope_indices, "rope_refreshes")
        self._copy(torch.arange(32, dtype=torch.int32) + self.seed + 1, self.sampler.seeds_tt_tensor, "seed_refreshes")
        self.remaining_steps = None
        self.reset_active_slots = True
        self.counters["resets"] += 1

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
        **kwargs,
    ):
        if kv_cache is not self.cache:
            self.bind_cache(kv_cache, page_table)
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        if tokens.ndim != 2 or tokens.shape[0] != len(prompt_lens):
            raise ValueError("Prefill needs one token row per prompt")
        if (tokens < 0).any() or (tokens >= self.model.config.vocab_size).any():
            raise ValueError("Token IDs lie outside the vocabulary")
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        starts = [0] * len(slots) if start_pos is None else list(start_pos)
        if len(starts) != len(slots) or len(slots) != len(prompt_lens) or len(set(slots)) != len(slots):
            raise ValueError("Each prompt must name one distinct fixed slot")
        signatures = {
            (kv_cache.batch_size, tuple(page_table.shape), slot, start, length, return_all_logits)
            for slot, start, length in zip(slots, starts, prompt_lens)
        }
        # New prefill programs own persistent buffers. Compile them before a
        # decode trace reserves scratch addresses; known shapes reuse the trace.
        if not signatures.issubset(self.prefill_signatures):
            self._release_traces()
        self._refresh_table(page_table)
        table = self.page_table
        results = []
        for row, (slot, length, start) in enumerate(zip(slots, prompt_lens, starts)):
            if not 0 <= slot < kv_cache.batch_size or not 1 <= length <= tokens.shape[-1]:
                raise ValueError("Invalid slot or logical prompt length")
            if start < 0 or start + length > kv_cache.capacity:
                raise ValueError("Prompt exceeds cache capacity")
            # Chunk across the whole stack, retaining only one chunk of activations.
            chunks = []
            for offset in range(0, length, 4096):
                count = min(4096, length - offset)
                ids = self.model.upload(
                    tokens[row : row + 1, offset : offset + count].int(),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                logits = self.model.prefill(
                    ids,
                    cache=kv_cache,
                    page_table=table,
                    length=count,
                    start_pos=start + offset,
                    slot=slot,
                    all_logits=return_all_logits,
                )
                if return_all_logits:
                    chunks.append(self._host_logits(logits))
            results.append(torch.cat(chunks, dim=1) if return_all_logits else logits)
        self.prefill_signatures.update(signatures)
        if return_all_logits:
            maximum = max(prompt_lens)
            return torch.cat([torch.nn.functional.pad(x, (0, 0, 0, maximum - x.shape[1])) for x in results], dim=0)
        return results

    def prefill_logits(self, prompt_token_ids):
        length = len(prompt_token_ids)
        cache = self._ensure_cache(1, length)
        self.reset()
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table,
            kv_cache=cache,
            prompt_lens=[length],
            return_all_logits=True,
        )

    def _model_step(self):
        logits = self.model.decode(
            self.tokens,
            self.positions,
            cache=self.cache,
            page_table=self.page_table,
            rope_indices=self.rope_indices,
            active_slots=self.active_slots,
        )
        b = self.cache.batch_size
        if b != 32:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - b), (0, 0)], value=0.0)
        ttnn.plus_one(self.positions, skip_negative_entries=True)
        ttnn.plus_one(self.rope_indices)
        return logits

    def _sampling_step(self, logits):
        self.sampler(logits, tt_out_tok=self.tokens)
        ttnn.plus_one(self.sampler.seeds_tt_tensor)

    def _capture(self):
        """Warm both graphs before capture; restore only mutable request state."""
        backups = [
            (state, name, ttnn.clone(getattr(state, name)))
            for state in self.cache.layers
            for name in ("conv", "recurrent")
            if getattr(state, name) is not None
        ]
        token_backup, pos_backup = ttnn.clone(self.tokens), ttnn.clone(self.positions)
        rope_backup = ttnn.clone(self.rope_indices)
        seed_backup = ttnn.clone(self.sampler.seeds_tt_tensor)
        warm = self._model_step()
        self._sampling_step(warm)
        for state, name, tensor in backups:
            ttnn.copy(tensor, getattr(state, name))
        for source, target in (
            (token_backup, self.tokens),
            (pos_backup, self.positions),
            (rope_backup, self.rope_indices),
            (seed_backup, self.sampler.seeds_tt_tensor),
        ):
            ttnn.copy(source, target)
        ttnn.synchronize_device(self.mesh)
        captured = []
        open_trace = None
        try:
            open_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            captured.append(open_trace)
            logits = self._model_step()
            ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
            open_trace = None
            open_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            captured.append(open_trace)
            self._sampling_step(logits)
            ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
            open_trace = None
        except BaseException:
            if open_trace is not None:
                ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
            for trace in captured:
                ttnn.release_trace(self.mesh, trace)
            raise
        self.trace, self.sample_trace = captured
        self.logits = logits
        self.counters["trace_captures"] += 2

    def decode_forward(
        self,
        tokens=None,
        start_pos=None,
        *,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        host_sampling=None,
        active_slots=None,
        **kwargs,
    ):
        if kv_cache is not self.cache:
            raise ValueError("Bind external cache with bind_cache before traced decode")
        if self.reset_active_slots:
            if active_slots is None and self.active_slots is not None:
                self._release_traces()
                self.active_slots = None
            self.reset_active_slots = False
        if active_slots is not None and tuple(active_slots) != self.active_slots:
            if start_pos is None:
                raise ValueError("Changing active slots requires authoritative positions")
            proposed = tuple(active_slots)
            if len(set(proposed)) != len(proposed) or any(i < 0 or i >= kv_cache.batch_size for i in proposed):
                raise ValueError("Invalid active-slot set")
            supplied_positions = torch.as_tensor(start_pos).reshape(-1)
            if supplied_positions.numel() != kv_cache.batch_size or any(
                supplied_positions[i] < 0 or supplied_positions[i] >= kv_cache.capacity for i in proposed
            ):
                raise ValueError("Active slots require positions inside the cache")
            self._release_traces()
            self.active_slots = proposed
        if not enable_trace:
            raise ValueError("Optimized decode requires tracing")
        self._refresh_table(page_table)
        if tokens is not None:
            supplied = torch.as_tensor(tokens).reshape(-1)
            if (
                supplied.numel() != kv_cache.batch_size
                or (supplied < 0).any()
                or (supplied >= self.model.config.vocab_size).any()
            ):
                raise ValueError("Decode requires one valid token per fixed slot")
            ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            ids.reshape(-1)[: kv_cache.batch_size] = supplied
            self._copy(ids, self.tokens, "token_refreshes")
        if start_pos is not None:
            positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
            if self.active_slots is not None:
                positions = positions.clone()
                positions[[i for i in range(kv_cache.batch_size) if i not in self.active_slots]] = -1
            if (
                positions.numel() != kv_cache.batch_size
                or (positions < -1).any()
                or (positions >= kv_cache.capacity).any()
            ):
                raise ValueError("Decode positions lie outside the cache")
            if self.active_slots is None and (positions < 0).any():
                raise ValueError("Negative positions require an explicit active-slot set")
            active = list(range(kv_cache.batch_size)) if self.active_slots is None else list(self.active_slots)
            if active and (positions[active] < 0).any():
                raise ValueError("Active slots require nonnegative positions")
            self.remaining_steps = int((kv_cache.capacity - positions[active]).min()) if active else 0
            self._copy(positions, self.positions, "position_refreshes")
            self._copy(positions.clamp_min(0), self.rope_indices, "rope_refreshes")
        if self.remaining_steps is None or self.remaining_steps <= 0:
            raise ValueError("No decode positions are bound, or cache capacity is exhausted")
        if self.trace is None:
            self._capture()
        ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        self.remaining_steps -= 1
        if self.host_sampling if host_sampling is None else host_sampling:
            # Explicit test-only compatibility boundary. Never used in token-out timings.
            logits = self._host_logits(self.logits)[0, 0, : kv_cache.batch_size]
            return logits
        ttnn.execute_trace(self.mesh, self.sample_trace, cq_id=0, blocking=False)
        self.counters["sampling_replays"] += 1
        return self._read_tokens()[: kv_cache.batch_size] if read_from_device else self.tokens

    def bind_cache(self, cache, page_table):
        if not 1 <= cache.batch_size <= 32 or not 1 <= cache.capacity <= self.model.context:
            raise ValueError("External cache dimensions exceed the model contract")
        shape = (
            tuple(page_table.shape) if isinstance(page_table, ttnn.Tensor) else tuple(torch.as_tensor(page_table).shape)
        )
        if len(shape) != 2 or shape[0] != cache.batch_size or shape[1] < (cache.capacity + 31) // 32:
            raise ValueError("Page table must cover the advertised cache capacity for every slot")
        if len(cache.layers) != len(self.model.layers):
            raise ValueError("Cache must cover every model layer")
        for layer, state in zip(self.model.layers, cache.layers):
            if layer.kind == "full_attention":
                if any(
                    t is None
                    or t.dtype != ttnn.bfloat8_b
                    or t.layout != ttnn.TILE_LAYOUT
                    or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                    or tuple(t.shape) != (cache.num_pages, 1, 32, 256)
                    for t in (state.key, state.value)
                ):
                    raise ValueError("Full-attention cache violates the TP4 BFP8 page contract")
            elif (
                state.recurrent is None
                or tuple(state.recurrent.shape) != (cache.batch_size, 12, 128, 128)
                or state.recurrent.dtype != ttnn.float32
                or state.recurrent.layout != ttnn.TILE_LAYOUT
                or state.recurrent.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                or state.conv is None
                or tuple(state.conv.shape) != (cache.batch_size, 3, 2560)
                or state.conv.dtype != ttnn.bfloat16
                or state.conv.layout != ttnn.ROW_MAJOR_LAYOUT
                or state.conv.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise ValueError("Linear state violates the FP32 recurrence / BF16 row-major convolution contract")
        if isinstance(page_table, ttnn.Tensor):
            if (
                page_table.dtype != ttnn.int32
                or page_table.layout != ttnn.ROW_MAJOR_LAYOUT
                or page_table.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise ValueError("Device page table must be INT32 row major in interleaved DRAM")
            page_host = None
        else:
            page_host = torch.as_tensor(page_table, dtype=torch.int32).clone()
            if (page_host < 0).any() or (page_host >= cache.num_pages).any():
                raise ValueError("Invalid external page table")
        self._release_traces()
        self.prefill_signatures.clear()
        self.remaining_steps = None
        self.cache = cache
        self.owns_cache = False
        self.positions = self.model.upload(
            torch.zeros(cache.batch_size, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.rope_indices = self.model.upload(
            torch.zeros(cache.batch_size, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.active_slots = None
        self.page_host = page_host
        if isinstance(page_table, ttnn.Tensor):
            self.page_table = page_table
        else:
            self.page_table = self.model.upload(self.page_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=None,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=0,
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("Readiness and generation require traced decode")
        if (
            max_new_tokens < 0
            or not prompt_token_ids
            or len(prompt_token_ids) + max(0, max_new_tokens - 1) > self.model.context
        ):
            raise ValueError("Invalid generation length or context")
        if max_new_tokens == 0:
            return []
        compat = self.host_sampling if host_sampling is None else host_sampling
        if compat and top_k != 1:
            raise ValueError("Host compatibility implements greedy sampling; use device sampling for top-k/top-p")
        start = time.perf_counter()
        before_request = self.counters.copy()
        self.set_sampling_params(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
        cache = self._ensure_cache(1, len(prompt_token_ids) + max_new_tokens - 1)
        self.reset()
        logits = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table,
            kv_cache=cache,
            prompt_lens=[len(prompt_token_ids)],
        )[0]
        padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
        if compat:
            first = int(self._host_logits(logits).reshape(-1, self.model.config.vocab_size)[-1].argmax())
        else:
            self._sampling_step(padded)
            first = int(self._read_tokens()[0])
        del padded, logits
        ttft = time.perf_counter() - start
        outputs = [first]
        forced = next_input(0, first) if next_input is not None else first
        self._copy(torch.tensor([len(prompt_token_ids)], dtype=torch.int32), self.positions, "position_refreshes")
        self._copy(torch.tensor([len(prompt_token_ids)], dtype=torch.int32), self.rope_indices, "rope_refreshes")
        if next_input is not None or compat:
            ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            ids.reshape(-1)[0] = forced
            self._copy(ids, self.tokens, "token_refreshes")
        self.remaining_steps = cache.capacity - len(prompt_token_ids)
        before_steady = self.counters.copy()
        begin = time.perf_counter()
        for step in range(1, max_new_tokens):
            out = self.decode_forward(page_table=self.page_table, kv_cache=cache, host_sampling=compat)
            predicted = int(out[0].argmax()) if compat else int(out[0])
            outputs.append(predicted)
            forced = next_input(step, predicted) if next_input is not None else predicted
            if (next_input is not None or compat) and step + 1 < max_new_tokens:
                ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                ids.reshape(-1)[0] = forced
                self._copy(ids, self.tokens, "token_refreshes")
        elapsed = time.perf_counter() - begin
        self.last_perf = dict(
            ttft_s=ttft,
            decode_s=elapsed,
            decode_tokens=max_new_tokens - 1,
            tokens_per_second=(max_new_tokens - 1) / elapsed,
            counters=dict(self.counters - before_request),
            steady_state_counters=dict(self.counters - before_steady),
            host_sampling=compat,
            teacher_forcing=next_input is not None,
        )
        return outputs

    def set_sampling_params(self, *, top_k=1, top_p=0.0, temperature=1.0, seed=0):
        if not 1 <= top_k <= 32 or not 0 <= top_p <= 1 or temperature <= 0:
            raise ValueError("Sampling requires k1..32, p0..1 and positive temperature")
        self.seed = seed
        previous = self.sampler.force_argmax_sampling
        # The common native sampler multiplies by inverse temperature.
        self.sampler.reset_params([top_k] * 32, [top_p] * 32, [1.0 / temperature] * 32)
        self._copy(torch.arange(32, dtype=torch.int32) + seed + 1, self.sampler.seeds_tt_tensor, "seed_refreshes")
        if previous != self.sampler.force_argmax_sampling:
            self._release_traces()

    def teardown(self):
        self.close()

    def close(self):
        self._release_traces()


def build_generator(model_dir, mesh_device, **kwargs):
    indices = kwargs.pop("layer_indices", None)
    model = QwenModel(
        mesh_device,
        snapshot=kwargs.pop("snapshot", None),
        layer_indices=indices,
        head_strategy=kwargs.pop("head_strategy", "dram"),
    )
    return QwenGenerator(
        model,
        host_sampling=kwargs.pop("host_sampling", False),
        sampling_strategy=kwargs.pop("sampling_strategy", "split"),
    )

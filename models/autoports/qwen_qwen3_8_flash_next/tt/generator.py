# SPDX-License-Identifier: Apache-2.0
"""Readiness generator with persistent split decode/sampling trace inputs.

Explicit host-sampling compatibility is diagnostic; the default token-out path
uses common Sampling1D and writes its output directly into the next token input.
"""

import json
import os
import secrets
import time
from collections import Counter

import torch
from readiness_check.contract import Generator

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.model import QwenFlashNextModel
from models.autoports.qwen_qwen3_8_flash_next.tt.state_ops import advance_positions, prepare_sample_seeds, record_tokens
from models.common.modules.sampling.sampling_1d import Sampling1D


class QwenFlashNextGenerator(Generator):
    def __init__(
        self,
        model,
        tokenizer,
        *,
        sampling_mode="device",
        prefill_trace_max=None,
        last_token_only=True,
        trace_cache_reset=True,
        record_history=True,
    ):
        self.model, self.mesh, self.tokenizer = model, model.mesh, tokenizer
        self.batch_size = model.batch_size
        self.kv_cache = model.cache
        if sampling_mode not in ("device", "host"):
            raise ValueError("sampling_mode must be device or host")
        self.sampling_mode = sampling_mode
        self._record_history = bool(record_history)
        self.counters = Counter()
        self.event_path = os.environ.get("QWEN_FLASH_TRACE_EVENTS")
        self.traces = {}
        self.prepared = False
        self.page_table = torch.arange(self.batch_size * model.context // 32, dtype=torch.int32).reshape(
            self.batch_size, -1
        )
        if model.serving_pool:
            # Preparation owns only private pages. No B*context public pool is
            # fabricated, and every fixed row has disjoint writable scratch.
            self.page_table = torch.stack(
                [
                    model.scratch_page_base + i * 16 + torch.arange(model.context // 32) % 16
                    for i in range(self.batch_size)
                ]
            ).to(torch.int32)
        self._page_host = self.page_table.clone()
        self._request_page_host = self.page_table.clone()
        self._request_slot_host = list(range(self.batch_size))
        self.tt_page = self._upload(self.page_table, ttnn.int32)
        self.tt_pos = self._upload(torch.zeros(self.batch_size, 32, dtype=torch.int32), ttnn.uint32)
        slot = torch.zeros(self.batch_size, 32, dtype=torch.int32)
        slot[:, 0] = torch.arange(self.batch_size)
        self.tt_slots = self._upload(slot, ttnn.uint32)
        self._slot_host = list(range(self.batch_size))
        self.tt_active = self._upload(torch.ones(self.batch_size, 32, dtype=torch.int32), ttnn.uint32)
        self.tt_logit_mask = self.model.ops.upload(torch.ones(1, 1, self.batch_size, 1), ttnn.bfloat16)
        self._active = [True] * self.batch_size
        # Sampling output's required logical shape; decode reshapes this same
        # backing storage to [B,1] inside the recorded model graph.
        self.tt_token = self._upload(torch.zeros(1, 1, 1, self.batch_size, dtype=torch.int32), ttnn.uint32)
        self.tt_logits = ttnn.empty(
            (1, 1, self.batch_size, model.cfg.vocab_size // 4),
            dtype=model.logits_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
        )
        # A full-context prefill may still produce its next-token prediction.
        # Keep this allocation in serving even when recording is disabled:
        # the cache/context memory budget and trace allocation plan stay fixed.
        self.tt_history = ttnn.empty(
            (self.batch_size, model.context + 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh
        )
        self.seed_init = self._upload(torch.arange(1, 33, dtype=torch.int32), ttnn.uint32)
        self.buckets = [1 << i for i in range(10 if self.batch_size == 1 else 4)]
        if prefill_trace_max is None:
            prefill_trace_max = 512 if self.batch_size == 1 else 0
        self.prefill_trace_buckets = [n for n in self.buckets if n <= prefill_trace_max]
        self.last_token_only = last_token_only
        self.trace_cache_reset = trace_cache_reset
        self.prefill_inputs = {
            n: self._upload(torch.zeros(self.batch_size, n, dtype=torch.int32), ttnn.uint32) for n in self.buckets
        }
        self.model.prepare(self.buckets)
        self.sampler = Sampling1D(
            model.cfg.vocab_size,
            mesh_device=self.mesh,
            tt_ccl=model.ops.ccl,
            max_batch_size=max(32, self.batch_size),
            max_top_k=model.precision_policy["sampling"]["max_top_k"],
            num_gather_links=2,
            allow_force_argmax=True,
            pad_to_power_of_2=True,
        )
        self.sampler.load_device_buffers()
        self.sample_params = {}
        for name, k, p, temp in [("greedy", 1, 0.0, 1.0), ("sample", 32, 0.9, 0.8)]:
            self.sample_params[name] = (
                self._upload(torch.full((self.batch_size,), k, dtype=torch.int32), ttnn.uint32),
                self._upload(torch.full((self.batch_size,), p), ttnn.bfloat16),
                self._upload(torch.full((self.batch_size,), 1.0 / temp), ttnn.bfloat16),
            )
        self.seeds = self._upload(torch.full((32,), 0xFFFFFFFF, dtype=torch.int64), ttnn.uint32)
        self.sample_seeds = self._upload(torch.ones(32, dtype=torch.int32), ttnn.uint32)
        self.last_perf = {}
        # The view is built only after the external owner cache is bound, and
        # before any warmup/capture. B1 and standalone generators are unchanged.
        self.prefill_view = None
        self.decode_views = {}
        self.compact_decode_enabled = True
        self.last_decode_execution_batch = self.batch_size

    def decode_runtime_summary(self):
        return {
            "enabled": self.compact_decode_enabled and bool(self.decode_views),
            "buckets": sorted(self.decode_views),
            "fallback_batch": self.batch_size,
            "selection": "smallest_bucket_covering_nonempty_contiguous_active_prefix_else_owner",
            "sampler_batch": self.batch_size,
            "sampler_owner": "original_generator",
            "position_advance": "original_owner_positions_and_active_mask_once",
            "views": {str(batch): view.runtime_summary() for batch, view in self.decode_views.items()},
        }

    @property
    def record_history(self):
        """Construction-time sampling graph choice, immutable after capture."""
        return self._record_history

    def event(self, kind, **fields):
        if self.event_path:
            with open(self.event_path, "a") as stream:
                stream.write(
                    json.dumps(dict(event=kind, time=time.perf_counter(), pid=os.getpid(), **fields), default=str)
                    + "\n"
                )

    def memory_event(self, kind, **fields):
        if self.event_path:
            memory = {}
            for label, buffer_type in (("dram", ttnn.BufferType.DRAM), ("trace", ttnn.BufferType.TRACE)):
                view = ttnn.get_memory_view(self.mesh, buffer_type)
                memory[label] = {
                    key: getattr(view, key)
                    for key in (
                        "num_banks",
                        "total_bytes_per_bank",
                        "total_bytes_allocated_per_bank",
                        "total_bytes_free_per_bank",
                        "largest_contiguous_bytes_free_per_bank",
                    )
                }
            self.event(kind, memory=memory, **fields)

    def _upload(self, value, dtype):
        return self.model.ops.upload(value, dtype, ttnn.ROW_MAJOR_LAYOUT)

    def _copy(self, value, tensor):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, tensor)

    def _positions(self, positions):
        host = torch.zeros(self.batch_size, 32, dtype=torch.int32)
        values = torch.as_tensor(positions, dtype=torch.int32).reshape(self.batch_size)
        if torch.any(values < 0) or torch.any(values > self.model.context):
            raise ValueError("Positions must lie within the configured context")
        host[:, 0] = values
        self._copy(host, self.tt_pos)
        self.counters["position_refreshes"] += 1

    def _request_slots(self, slots):
        slots = list(self._request_slot_host if slots is None else slots)
        if (
            len(slots) != self.batch_size
            or len(set(slots)) != len(slots)
            or any(not 0 <= s < self.model.max_requests for s in slots)
        ):
            raise ValueError("Request slots must be unique and within max_requests")
        self._request_slot_host = slots.copy()
        return slots

    def _request_pages(self, page_table):
        if page_table is None:
            # The owned snapshot is already validated. Scratch remapping below
            # needs one mutable copy, never a replacement of the public owner.
            return self._request_page_host.clone()
        pages = torch.as_tensor(page_table, dtype=torch.int32)
        if pages.shape != self.page_table.shape:
            raise ValueError("Page table must match the configured physical cache capacity")
        if torch.any(pages < 0) or torch.any(pages >= self.model.scratch_page_base):
            raise ValueError("Public page table references an invalid or private scratch block")
        self._request_page_host = pages.clone()
        return pages.clone()

    def _slots(self, slots):
        slots = list(slots)
        if slots == self._slot_host:
            return
        host = torch.zeros(self.batch_size, 32, dtype=torch.int32)
        host[:, 0] = torch.as_tensor(slots, dtype=torch.int32)
        self._copy(host, self.tt_slots)
        self._slot_host = slots
        self.counters["slot_refreshes"] += 1

    def _pages(self, page_table):
        page_table = torch.as_tensor(page_table, dtype=torch.int32)
        if page_table.shape != self._page_host.shape:
            raise ValueError("Page table must match the configured physical cache capacity")
        if torch.any(page_table < 0) or torch.any(page_table >= self.model.scratch_page_base + self.batch_size * 16):
            raise ValueError("Page table exceeds physical pool plus private scratch")
        if not torch.equal(page_table, self._page_host):
            self._copy(page_table, self.tt_page)
            self._page_host = page_table.clone()
            self.counters["page_table_refreshes"] += 1

    def _active_rows(self, active):
        if len(active) != self.batch_size:
            raise ValueError("active_rows must contain one flag per fixed batch row")
        active = list(active)
        if active == self._active:
            return
        flags = torch.tensor(active, dtype=torch.int32)
        self._copy(flags[:, None].expand(-1, 32).contiguous(), self.tt_active)
        self._copy(flags.float().reshape(1, 1, self.batch_size, 1), self.tt_logit_mask)
        self._active = active
        self.counters["activity_refreshes"] += 1

    def _bind_cache(self, cache):
        if not self.prepared and cache is not self.kv_cache:
            raise ValueError("Call prepare(kv_cache=cache) before populating an external cache")
        if self.prepared and cache is not self.kv_cache:
            raise ValueError("Bind external cache before prepare(); live trace buffer bindings are immutable")
        self.kv_cache = cache
        self.model.bind_cache(cache)

    def _store_last_logits(self, logits, n):
        last = ttnn.reshape(logits[:, :, n - 1 : n, :], self.tt_logits.shape)
        if self.batch_size > 1:
            last = ttnn.where(self.tt_logit_mask, last, self.tt_logits)
        ttnn.copy(last, self.tt_logits)

    def _model_decode(self):
        logits = self.model.forward(
            ttnn.reshape(self.tt_token, (self.batch_size, 1)),
            positions=self.tt_pos,
            slots=self.tt_slots,
            page_table=self.tt_page,
            kv_cache=self.kv_cache,
            decode=True,
        )
        ttnn.copy(logits, self.tt_logits)
        advance_positions(self.tt_pos, self.tt_active)

    def _sample(self, mode):
        params = {} if mode == "argmax" else dict(zip(("k", "p", "temp"), self.sample_params[mode]))
        seeds = self.seeds
        if mode == "sample":
            prepare_sample_seeds(self.seed_init, self.tt_pos, self.sample_seeds)
            seeds = self.sample_seeds
        result = self.sampler.decode_forward(self.tt_logits, **params, seeds=seeds, tt_out_tok=self.tt_token)[0]
        if self.record_history:
            record_tokens(self.tt_token, self.tt_pos, self.tt_history)
        return result

    def _prefill_chunk(self, n, *, last_token_only=False):
        return self.model.forward(
            self.prefill_inputs[n],
            positions=self.tt_pos,
            slots=self.tt_slots,
            page_table=self.tt_page,
            kv_cache=self.kv_cache,
            last_token_only=last_token_only,
        )

    def _prefill_token_out(self, n):
        logits = self._prefill_chunk(n, last_token_only=self.last_token_only)
        self._store_last_logits(logits, 1 if self.last_token_only else n)

    def prepare(self, *, kv_cache=None):
        """Compile and capture once, then clear request state.

        External owners must bind their empty cache here before populating it.
        After preparation, cache addresses remain stable for all requests.
        """
        if self.prepared:
            if kv_cache is not None:
                self._bind_cache(kv_cache)
            return
        if kv_cache is not None:
            self.model.bind_cache(kv_cache)
            self.kv_cache = kv_cache
        if self.model.serving_pool and self.batch_size > 1:
            from .prefill_view import SerialPrefillView

            self.prefill_view = SerialPrefillView(self.model, self.tt_logits)
            self.prefill_view.warm()
        if self.model.serving_pool and self.batch_size == 32:
            from .decode_view import DECODE_BUCKETS, CompactDecodeView

            pool = {}
            self.decode_views = {batch: CompactDecodeView(self, batch, pool) for batch in DECODE_BUCKETS}
        # Retain and warm the fixed-batch full-logit diagnostic route as well.
        # No new shape/program may first appear after a live trace is captured.
        for n in self.buckets:
            print("PREPARE_PREFILL", n, flush=True)
            tick = time.perf_counter()
            self._positions([0] * self.batch_size)
            logits = self._prefill_chunk(n)
            self._store_last_logits(logits, n)
            # Host-logit compatibility paths are warmed before traces exist.
            ttnn.untilize(logits, use_multicore=True)
            del logits
            if self.last_token_only:
                self._positions([0] * self.batch_size)
                self._prefill_token_out(n)
            self.event(
                "preparation",
                signature=f"prefill_{n}",
                elapsed_s=time.perf_counter() - tick,
                scope="warmup host dispatch including cache lookup/compilation; capture follows",
            )
        self.model.reset_cache(self.kv_cache)
        self._positions([0] * self.batch_size)
        tick = time.perf_counter()
        self._model_decode()
        self.event(
            "preparation",
            signature="decode",
            elapsed_s=time.perf_counter() - tick,
            scope="warmup host dispatch including cache lookup/compilation; capture follows",
        )
        for batch, view in self.decode_views.items():
            print("PREPARE_DECODE_VIEW", batch, flush=True)
            tick = time.perf_counter()
            self._positions([0] * self.batch_size)
            view.forward()
            self.event(
                "preparation",
                signature=f"decode_compact_{batch}",
                elapsed_s=time.perf_counter() - tick,
                scope="warmup host dispatch including cache lookup/compilation; capture follows",
            )
        for mode in ("greedy", "sample", "argmax"):
            tick = time.perf_counter()
            self._sample(mode)
            self.event(
                "preparation",
                signature=mode,
                elapsed_s=time.perf_counter() - tick,
                scope="warmup host dispatch including cache lookup/compilation; capture follows",
            )
        self._reset_rng()
        self._positions([0] * self.batch_size)
        self._copy(torch.zeros(1, 1, 1, self.batch_size, dtype=torch.int32), self.tt_token)
        ttnn.synchronize_device(self.mesh)
        self.counters["synchronizations"] += 1
        self.memory_event("preparation_memory")
        self.prepared = True
        if self.prefill_view is not None:
            self.prefill_view.capture(self.traces, self.event)
            self.counters["captures"] += len(self.prefill_view.buckets)
        for name, fn in (
            [(f"decode_compact_{batch}", view.forward) for batch, view in self.decode_views.items()]
            + [("decode", self._model_decode)]
            + [(mode, lambda m=mode: self._sample(m)) for mode in ("greedy", "sample", "argmax")]
        ):
            print("CAPTURE", name, flush=True)
            tick = time.perf_counter()
            if name == "decode" or name.startswith("decode_compact_"):
                # Capturing a smaller graph advances all active owner rows;
                # newly included PLE rows in the next graph need position0.
                # This setup upload is deliberately outside the live graph.
                self._positions([0] * self.batch_size)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces[name] = tid
            self.counters["captures"] += 1
            self.memory_event("capture", signature=name, trace_id=tid, elapsed_s=time.perf_counter() - tick)
        for n in [] if self.prefill_view is not None else self.prefill_trace_buckets:
            name = f"prefill_{n}"
            print("CAPTURE", name, flush=True)
            tick = time.perf_counter()
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._prefill_token_out(n)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces[name] = tid
            self.counters["captures"] += 1
            self.memory_event("capture", signature=name, trace_id=tid, elapsed_s=time.perf_counter() - tick)
        if self.trace_cache_reset:
            print("CAPTURE reset_cache", flush=True)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.model.reset_cache(self.kv_cache)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces["reset_cache"] = tid
            self.counters["captures"] += 1
            self.event("capture", signature="reset_cache", trace_id=tid)
        self.reset()
        self.event(
            "prepared", traces=self.traces, counters=dict(self.counters), batch=self.batch_size, buckets=self.buckets
        )

    def _reset_rng(self):
        ttnn.manual_seed(
            seeds=self.seed_init, user_ids=self.sampler._user_ids, sub_core_grids=self.sampler._sampling_sub_core_grids
        )

    def set_sampling_params(self, *, top_k=32, top_p=0.9, temperature=0.8, seed=1):
        if not 1 <= top_k <= 32 or not 0 <= top_p <= 1 or temperature <= 0:
            raise ValueError("Expected top_k1..32, top_p0..1, temperature>0")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for value, tensor in zip((top_k, top_p, 1.0 / temperature), self.sample_params["sample"]):
            dtype = torch.int32 if tensor.dtype == ttnn.uint32 else torch.float32
            self._copy(torch.full((self.batch_size,), value, dtype=dtype), tensor)
        self._copy(torch.arange(seed, seed + 32, dtype=torch.int64) & 0xFFFFFFFF, self.seed_init)
        self._reset_rng()

    def set_serving_sampling_params(self, params):
        """Translate per-row scheduler parameters into the canonical sampler.

        Both greedy and mixed stochastic batches use the already prepared
        full-model sampling traces. No additional sampler or feedback path.
        """
        from models.common.modules.sampling.params import prepare_sampling_params

        prepared = prepare_sampling_params(params, self.batch_size, max_device_top_k=32, allow_force_argmax=False)
        if prepared.penalties_enabled or any(prepared.enable_log_probs):
            raise ValueError("Penalties and logprobs require explicit host compatibility")
        if all(k == 1 for k in prepared.top_k):
            return "greedy"
        for values_, tensor in zip(
            (prepared.top_k, prepared.top_p, prepared.temperature), self.sample_params["sample"]
        ):
            dtype = torch.int32 if tensor.dtype == ttnn.uint32 else torch.float32
            self._copy(torch.tensor(values_, dtype=dtype), tensor)
        seeds = prepared.seeds
        seeds = [secrets.randbits(32) if seed is None else int(seed) for seed in seeds]
        if any(not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("Device seed must be uint32")
        seeds += [1] * (32 - len(seeds))
        self._copy(torch.tensor(seeds, dtype=torch.int64), self.seed_init)
        self._reset_rng()
        return "sample"

    def _replay(self, name):
        ttnn.execute_trace(self.mesh, self.traces[name], cq_id=0, blocking=False)
        self.counters[name + "_replays"] += 1
        if name.startswith("decode_compact_"):
            self.counters["decode_replays"] += 1
        self.event("replay", signature=name, trace_id=self.traces[name])

    def _read_tokens(self):
        self.counters["token_readbacks"] += 1
        values = ttnn.to_torch(ttnn.get_device_tensors(self.tt_token)[0]).reshape(-1).long().tolist()
        return [v if self._active[i] else -1 for i, v in enumerate(values)]

    def _read_logits(self, tensor):
        """Only explicit logits/test/host-sampling boundaries call this."""
        self.counters["logits_readbacks"] += 1
        shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]
        return torch.cat(shards, -1).reshape(self.batch_size, -1, self.model.cfg.vocab_size)

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        start_pos=None,
        slots=None,
        active_rows=None,
        read_tokens=True,
        sampling="greedy",
        **kwargs,
    ):
        tokens = torch.as_tensor(tokens, dtype=torch.int32)
        if tokens.ndim != 2 or tokens.shape[0] != self.batch_size:
            raise ValueError("Expected tokens with shape [batch_size, sequence]")
        self._bind_cache(kv_cache)
        self.prepare()
        lengths = list(prompt_lens)
        if len(lengths) != self.batch_size or any(n < 0 or n > tokens.shape[1] for n in lengths):
            raise ValueError("Invalid logical prompt lengths")
        active = [n > 0 for n in lengths] if active_rows is None else list(active_rows)
        if any(active[i] and not lengths[i] for i in range(self.batch_size)):
            raise ValueError("Active prompt rows require at least one token")
        real_slots = self._request_slots(slots)
        start = (
            torch.zeros(self.batch_size, dtype=torch.int32)
            if start_pos is None
            else torch.as_tensor(start_pos, dtype=torch.int32)
        )
        if tuple(start.shape) != (self.batch_size,) or torch.any(start < 0):
            raise ValueError("Expected nonnegative start_pos with one position per row")
        if any(int(p) + lengths[i] > self.model.context for i, p in enumerate(start)):
            raise ValueError("Prompt exceeds configured physical cache capacity")
        pages = self._request_pages(page_table)
        offset = 0
        outputs = [[] for _ in lengths]
        use_view = self.prefill_view is not None and not return_all_logits
        if use_view:
            self.prefill_view.forward(
                tokens, lengths, start, real_slots, pages, active, replay=self._replay, event=self.event
            )
        while not use_view and any(active[i] and offset < lengths[i] for i in range(self.batch_size)):
            current = [active[i] and offset < lengths[i] for i in range(self.batch_size)]
            remaining = min(lengths[i] - offset for i in range(self.batch_size) if current[i])
            n = max(b for b in self.buckets if b <= remaining)
            chunk = torch.zeros(self.batch_size, n, dtype=torch.int32)
            chunk_pages = pages.clone()
            chunk_slots = real_slots.copy()
            positions = start + offset
            for i in range(self.batch_size):
                if current[i]:
                    chunk[i] = tokens[i, offset : offset + n]
                else:
                    positions[i] = 0
                    chunk_slots[i] = self.model.max_requests + i
                    chunk_pages[i, :16] = torch.arange(16) + self.model.scratch_page_base + i * 16
            self._active_rows(current)
            self._slots(chunk_slots)
            self._pages(chunk_pages)
            self._copy(chunk, self.prefill_inputs[n])
            self._positions(positions)
            self.event(
                "prefill_chunk",
                logical_lengths=lengths,
                chunk_length=n,
                start_positions=positions.tolist(),
                slots=chunk_slots,
                active=current,
            )
            if return_all_logits:
                logits = self._prefill_chunk(n)
                self._store_last_logits(logits, n)
                host = self._read_logits(logits)
                for i in range(self.batch_size):
                    if current[i]:
                        outputs[i].append(host[i])
                del logits
            elif n in self.prefill_trace_buckets:
                self._replay(f"prefill_{n}")
            else:
                self._prefill_token_out(n)
            offset += n
        self._active_rows(active)
        final_positions = start + torch.tensor(lengths)
        for i in range(self.batch_size):
            if not active[i]:
                real_slots[i] = self.model.max_requests + i
                pages[i, :16] = torch.arange(16) + self.model.scratch_page_base + i * 16
                final_positions[i] = 0
        self._slots(real_slots)
        self._pages(pages)
        self._positions(final_positions)
        if return_all_logits:
            result = torch.zeros(self.batch_size, max(lengths), self.model.cfg.vocab_size, dtype=torch.bfloat16)
            for i, parts in enumerate(outputs):
                if parts:
                    result[i, : lengths[i]] = torch.cat(parts, dim=0)
            return result
        if self.sampling_mode == "host":
            return self._read_logits(self.tt_logits)
        self._replay(sampling)
        return torch.tensor(self._read_tokens()) if read_tokens else self.tt_token

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        slots=None,
        active_rows=None,
        sampling="greedy",
        read_tokens=True,
        **kwargs,
    ):
        """Replay with optional scheduler updates; None retains device state.

        read_tokens=False returns the persistent device token output and avoids
        synchronization. Its contents are consumed directly by the next replay.
        """
        if not read_tokens and self.sampling_mode == "host":
            raise ValueError("Host sampling compatibility requires host logits")
        if start_pos is not None and torch.any(torch.as_tensor(start_pos) >= self.model.context):
            raise ValueError("A decode input position must be below the context limit")
        self._bind_cache(kv_cache)
        self.prepare()
        if active_rows is not None or slots is not None or page_table is not None:
            if active_rows is not None and start_pos is None:
                raise ValueError("Scheduler activity changes require explicit positions")
            if active_rows is not None:
                self._active_rows(active_rows)
            page_table = self._request_pages(page_table)
            # Page-only growth must retain token/position/activity and physical
            # slot residency, including the private slots of inactive rows.
            update_slots = active_rows is not None or slots is not None
            slots = self._request_slots(slots) if update_slots else None
            if start_pos is not None:
                start_pos = torch.as_tensor(start_pos, dtype=torch.int32).clone()
            for i, active in enumerate(self._active):
                if not active:
                    if slots is not None:
                        slots[i] = self.model.max_requests + i
                    page_table[i, :16] = torch.arange(16) + self.model.scratch_page_base + i * 16
                    if start_pos is not None:
                        start_pos[i] = 0
        if page_table is not None:
            self._pages(page_table)
        if slots is not None:
            self._slots(slots)
        if tokens is not None:
            self._copy(torch.as_tensor(tokens, dtype=torch.int32).reshape(1, 1, 1, self.batch_size), self.tt_token)
            self.counters["token_refreshes"] += 1
        if start_pos is not None:
            self._positions(start_pos)
        from .decode_view import select_decode_batch

        batch = select_decode_batch(self._active, enabled=self.compact_decode_enabled and bool(self.decode_views))
        self.last_decode_execution_batch = batch
        self._replay(f"decode_compact_{batch}" if batch in self.decode_views else "decode")
        if self.sampling_mode == "host":
            return self._read_logits(self.tt_logits)[:, 0, :]
        self._replay(sampling)
        return torch.tensor(self._read_tokens()) if read_tokens else self.tt_token

    def prefill_logits(self, prompt_token_ids):
        tokens, lengths = self._single_prompt(prompt_token_ids)
        self.reset()
        return self.prefill_forward(
            tokens, page_table=self.page_table, kv_cache=self.kv_cache, prompt_lens=lengths, return_all_logits=True
        )[:1]

    def _single_prompt(self, prompt_token_ids):
        if not prompt_token_ids:
            raise ValueError("A prompt must contain at least one token")
        tokens = torch.zeros(self.batch_size, len(prompt_token_ids), dtype=torch.int32)
        tokens[0] = torch.as_tensor(prompt_token_ids, dtype=torch.int32)
        return tokens, [len(prompt_token_ids)] + [0] * (self.batch_size - 1)

    def generate(self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True, **kwargs):
        if not self.record_history:
            raise ValueError("Standalone generate requires record_history=True")
        if not enable_trace:
            raise ValueError("This generator requires traced decode")
        if max_new_tokens < 1:
            return []
        if len(prompt_token_ids) + max_new_tokens - 1 > self.model.context:
            raise ValueError("Requested generation exceeds configured context")
        self.prepare()
        start = time.perf_counter()
        self.reset()
        device_loop = next_input is None and self.sampling_mode == "device"
        mode = kwargs.get("sampling", "greedy")
        tokens, lengths = self._single_prompt(prompt_token_ids)
        out = self.prefill_forward(
            tokens,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=lengths,
            read_tokens=not device_loop,
            sampling=mode,
        )
        if device_loop:
            # TTFT requires observing the first output; subsequent sampled
            # feedback and position advance remain entirely in the traces.
            first = self._read_tokens()[0]
            ttft = time.perf_counter() - start
            tick = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                self.decode_forward(
                    None, None, page_table=None, kv_cache=self.kv_cache, sampling=mode, read_tokens=False
                )
            host = ttnn.to_torch(ttnn.get_device_tensors(self.tt_history)[0])
            self.counters["history_readbacks"] += 1
            result = host[0, len(prompt_token_ids) : len(prompt_token_ids) + max_new_tokens].long().tolist()
            assert result[0] == first
            elapsed = time.perf_counter() - tick
            self.last_perf = dict(
                prompt_len=len(prompt_token_ids),
                gen_len=max_new_tokens,
                ttft_s=ttft,
                decode_s=elapsed,
                decode_tps=(max_new_tokens - 1) / elapsed,
                sampling_mode="device",
                sampling=mode,
            )
            return self._trim_eos(result) if kwargs.get("stop_on_eos", False) else result
        pred = int(out[0].argmax(-1).item()) if self.sampling_mode == "host" else int(out[0].item())
        ttft = time.perf_counter() - start
        result = [pred]
        tick = time.perf_counter()
        for step in range(max_new_tokens - 1):
            forced = next_input(step, pred) if next_input is not None else None
            if self.sampling_mode == "host" and forced is None:
                forced = pred
            forced_tokens = None
            if forced is not None:
                forced_tokens = torch.zeros(self.batch_size, 1, dtype=torch.int32)
                forced_tokens[0, 0] = forced
            out = self.decode_forward(
                forced_tokens, None, page_table=None, kv_cache=self.kv_cache, sampling=kwargs.get("sampling", "greedy")
            )
            pred = int(out[0].argmax(-1).item()) if self.sampling_mode == "host" else int(out[0].item())
            result.append(pred)
        if next_input is not None:
            next_input(max_new_tokens - 1, pred)
        elapsed = time.perf_counter() - tick
        self.last_perf = dict(
            prompt_len=len(prompt_token_ids),
            gen_len=max_new_tokens,
            ttft_s=ttft,
            decode_s=elapsed,
            decode_tps=(max_new_tokens - 1) / elapsed,
            sampling_mode=self.sampling_mode,
        )
        return self._trim_eos(result) if kwargs.get("stop_on_eos", False) else result

    def _trim_eos(self, tokens):
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for i, token in enumerate(tokens):
            if token in eos:
                return tokens[: i + 1]
        return tokens

    def reset(self):
        if self.trace_cache_reset and "reset_cache" in self.traces:
            self._replay("reset_cache")
        else:
            self.model.reset_cache(self.kv_cache)
        self._reset_rng()
        self._active_rows([True] * self.batch_size)
        self._positions([0] * self.batch_size)
        self._request_page_host = self.page_table.clone()
        self._request_slot_host = list(range(self.batch_size))
        self._pages(self.page_table)
        self._slots(list(range(self.batch_size)))
        self.counters["resets"] += 1

    def teardown(self):
        """Release traces when called directly by standard readiness runners."""
        self.close()

    def close(self):
        for name, tid in self.traces.items():
            ttnn.release_trace(self.mesh, tid)
            self.counters["releases"] += 1
            self.event("shutdown_release", signature=name, trace_id=tid)
        self.traces.clear()


def build_generator(model_dir, mesh_device, **kwargs):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    checkpoint = kwargs.pop("checkpoint", None) or snapshot_download(
        "Qwen/Qwen3.8-Flash-Next", revision="de4b8e4d43b917e7706784d8bb445c9af86a3540", local_files_only=True
    )
    count = kwargs.pop("override_num_layers", None)
    indices = kwargs.pop("layer_indices", None)
    if count is not None:
        indices = list(range(count))
    model = QwenFlashNextModel(
        checkpoint,
        mesh_device,
        precision_config=kwargs.pop("precision_config", None),
        batch_size=kwargs.pop("batch_size", 1),
        max_requests=kwargs.pop("max_requests", None),
        layer_indices=indices,
        max_seq_len=kwargs.pop("max_seq_len", None),
        cache_dir=kwargs.pop(
            "cache_dir", "/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"
        ),
    )
    return QwenFlashNextGenerator(
        model,
        AutoTokenizer.from_pretrained(checkpoint, local_files_only=True),
        sampling_mode=kwargs.pop("sampling_mode", "device"),
        prefill_trace_max=kwargs.pop("prefill_trace_max", None),
        last_token_only=kwargs.pop("last_token_only", True),
        trace_cache_reset=kwargs.pop("trace_cache_reset", True),
    )

# SPDX-License-Identifier: Apache-2.0
"""Readiness generator with persistent model and common-sampler split traces."""

import math
import time
from collections import Counter
from dataclasses import dataclass

import torch

try:
    from readiness_check.contract import Generator
except ImportError:  # Serving images (TTI/CI) do not ship the bring-up runtime.
    from models.autoports.aleph_alpha_kolibri_1_bf16.tt.readiness_contract import Generator

import ttnn
from models.autoports.aleph_alpha_kolibri_1_bf16.tt.checkpoint import CONTEXT
from models.autoports.aleph_alpha_kolibri_1_bf16.tt.model import DRAM, KolibriModel
from models.common.modules.sampling.sampling_1d import Sampling1D


@dataclass
class CacheState:
    layers: list
    page_tables: dict
    host_page_tables: dict
    capacity: int
    batch: int


class SamplingCCL:
    """Common sampler interface over the decoder's full-grid semaphore owner."""

    def __init__(self, workspace):
        self.ccl = workspace.ccl

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        return self.ccl.get_ag_ping_pong_semaphore()

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return self.ccl.get_barrier_semaphore()

    def line_all_gather(self, tensor, *, dim, cluster_axis=None, memory_config=DRAM, num_links=1, buffer_key=None):
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            memory_config=memory_config,
        )


class KolibriGenerator(Generator):
    def __init__(self, model, *, batch_size=1, capacity=CONTEXT, cache_state=None, trace_prefill=True):
        self.model, self.mesh_device = model, model.device
        if not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be in 1..32")
        if not 1 <= capacity <= model.rope_capacity:
            raise ValueError("capacity exceeds configured absolute position storage")
        self.batch_size = batch_size
        self.trace_prefill = trace_prefill
        self.logical_capacity = capacity
        self.capacity = math.ceil(capacity / 512) * 512
        self.counters = Counter()
        self.event_sink = None
        self.state = self.allocate_cache() if cache_state is None else cache_state
        if self.state.batch != batch_size or self.state.capacity != self.capacity:
            raise ValueError("External cache configuration differs from generator configuration")
        expected_cache_dtype = getattr(ttnn, model.runtime_precision["kv_cache_dtype"])
        if len(self.state.layers) != len(model.layers) or any(
            t.dtype != expected_cache_dtype for cache in self.state.layers for t in cache
        ):
            raise ValueError("External cache dtype/layer count differs from selected precision policy")
        self.owns_cache = cache_state is None
        self.initial_pages = {key: value.clone() for key, value in self.state.host_page_tables.items()}
        # Private snapshots prevent a caller's in-place host-table mutation
        # from looking unchanged merely because both references alias it.
        self.copied_pages = {key: value.clone() for key, value in self.initial_pages.items()}
        self.tokens = self.tt(
            torch.zeros(1, 1, 1, batch_size, dtype=torch.int32),
            getattr(ttnn, model.runtime_precision["sampling_index_dtype"]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.positions = self.tt(torch.zeros(batch_size, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.rope_positions = self.tt(torch.zeros(batch_size, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.logits = ttnn.empty(
            (1, 1, batch_size, 32000),
            dtype=getattr(ttnn, model.runtime_precision["sampling_logits_dtype"]),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=DRAM,
        )
        self.sampler = Sampling1D(
            128000,
            self.mesh_device,
            tt_ccl=SamplingCCL(model.workspace),
            max_batch_size=batch_size,
            max_top_k=32,
            allow_force_argmax=True,
            pad_to_power_of_2=True,
        )
        self.sampler.load_device_buffers()
        self.seed_increments = self.tt(torch.ones(batch_size, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.k = self.tt(
            torch.ones(batch_size, dtype=torch.int32),
            getattr(ttnn, model.runtime_precision["sampling_index_dtype"]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.p = self.tt(
            torch.zeros(batch_size),
            getattr(ttnn, model.runtime_precision["sampling_parameter_dtype"]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.temperature = self.tt(
            torch.ones(batch_size),
            getattr(ttnn, model.runtime_precision["sampling_parameter_dtype"]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        # A precision policy may need smaller prefill temporaries while retaining
        # the same logical context and KV allocation. Every layer must accept
        # the generator's physical bucket, including layer-specific policies.
        chunk_limit = min(layer.policy.prefill_chunk_size for layer in model.layers)
        if chunk_limit < 32:
            raise ValueError("prefill_chunk_size must allow at least one 32-row tile")
        self.buckets = tuple(n for n in (32, 128, 512, 2048, 8192) if n <= min(self.capacity, chunk_limit))
        self.prefill_buffers = {}
        for n in self.buckets:
            self.prefill_buffers[n] = dict(
                tokens=self.tt(
                    torch.zeros(1, n, dtype=torch.int32),
                    getattr(ttnn, model.runtime_precision["sampling_index_dtype"]),
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                cos=self.tt(torch.ones(1, 1, n, 128)),
                sin=self.tt(torch.zeros(1, 1, n, 128)),
                chunk_start=self.tt(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                page_tables={
                    key: self.tt(value[:1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                    for key, value in self.state.host_page_tables.items()
                },
                chunk_page_tables={
                    key: self.tt(value[:1, : n // 32], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                    for key, value in self.state.host_page_tables.items()
                },
            )
        self.prefill_outputs = (
            {
                n: ttnn.empty(
                    (1, 1, n, 2560),
                    dtype=getattr(ttnn, model.runtime_precision["residual_dtype"]),
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=DRAM,
                )
                for n in self.buckets
            }
            if trace_prefill
            else {}
        )
        self.traces = {}
        self.prepared = False
        # Each physical bucket owns stable inputs allocated before capture.
        # Keep only its most recent host values, independent of caller storage.
        # Decode and other buckets never write these input allocations.
        self.prefill_input_snapshots = {}
        self.prefill_rope_positions = {}

    def tt(self, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return self.model.tensor(value, dtype, layout)

    def copy(self, host, target, counter=None):
        source = ttnn.from_torch(host.contiguous(), dtype=target.dtype, layout=target.layout)
        ttnn.copy_host_to_device_tensor(source, target)
        if counter:
            self.counters[counter] += 1

    def allocate_cache(self):
        b, cap = self.batch_size, self.capacity
        full_pages = cap // 32
        ring_pages = min(cap, 8704) // 32
        pages = {
            "full": torch.arange(b * full_pages, dtype=torch.int32).reshape(b, full_pages),
            "sliding": torch.stack(
                [torch.arange(full_pages, dtype=torch.int32) % ring_pages + slot * ring_pages for slot in range(b)]
            ),
        }
        cache = []
        for layer in self.model.layers:
            count = ring_pages if layer.sliding else full_pages
            cache.append(
                tuple(
                    ttnn.zeros(
                        (b * count, 1, 32, 128),
                        dtype=getattr(ttnn, self.model.runtime_precision["kv_cache_dtype"]),
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=DRAM,
                    )
                    for _ in range(2)
                )
            )
        tables = {key: self.tt(value, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for key, value in pages.items()}
        return CacheState(cache, tables, pages, cap, b)

    def _decode(self):
        value = self.model.decode_forward(
            self.tokens,
            current_pos=self.positions,
            page_tables=self.state.page_tables,
            kv_cache=self.state.layers,
            rope_positions=self.rope_positions,
        )
        ttnn.copy(value, self.logits)
        ttnn.plus_one(self.positions, skip_negative_entries=True)
        ttnn.plus_one(self.rope_positions, skip_negative_entries=True)

    def _sample(self, mode="split"):
        args = {} if mode == "argmax" else dict(k=self.k, p=self.p, temp=self.temperature)
        self.sampler.decode_forward(self.logits, tt_out_tok=self.tokens, **args)
        updated_seeds = ttnn.add(self.sampler._seeds, self.seed_increments)
        ttnn.copy(updated_seeds, self.sampler._seeds)

    def _prefill_bucket(self, n, bound, alignment=32):
        return self.model.prefill_chunk_forward(
            **self.prefill_buffers[n],
            kv_cache=self.state.layers,
            context_bound=bound,
            chunk_start_alignment=alignment,
            return_hidden=True,
        )

    def _recordable_prefill(self, n):
        value = self._prefill_bucket(n, min(65536, self.capacity), 256)
        ttnn.copy(value, self.prefill_outputs[n])

    def prepare(self):
        """Compile all configured physical variants, then record without late setup."""
        if self.prepared:
            return
        for n in self.buckets:
            compile_started = time.monotonic()
            for bound in [65536, self.capacity] if self.capacity > 65536 else [self.capacity]:
                for alignment in (32, 64, 128, 256):
                    hidden = self._prefill_bucket(n, bound, alignment)
                    if alignment != 256:
                        del hidden
                # Readiness logits use a bounded 32-row terminal projection.
                # Warm every tile-aligned extraction before any live trace.
                for offset in range(0, n, 32):
                    piece = ttnn.slice(hidden, (0, 0, offset, 0), (1, 1, offset + 32, 2560))
                    if offset == 0:
                        terminal = self.model.terminal(piece)
                        del terminal
                    del piece
                del hidden
            print(f"PREPARED_PREFILL_BUCKET {n}", flush=True)
            if self.event_sink:
                self.event_sink("compile", key=f"prefill_{n}", seconds=time.monotonic() - compile_started)
        if self.trace_prefill:
            for n in self.buckets:
                self._recordable_prefill(n)
        compile_started = time.monotonic()
        self._decode()
        if self.event_sink:
            self.event_sink("compile", key="decode", seconds=time.monotonic() - compile_started)
        for mode in ("split", "argmax"):
            compile_started = time.monotonic()
            self._sample(mode)
            if self.event_sink:
                self.event_sink("compile", key=mode, seconds=time.monotonic() - compile_started)
        self.reset()
        ttnn.synchronize_device(self.mesh_device)
        self.counters["synchronizations"] += 1
        for key, call in (
            ("decode", self._decode),
            ("split", self._sample),
            ("argmax", lambda: self._sample("argmax")),
        ):
            capture_started = time.monotonic()
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            call()
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            self.traces[key] = tid
            self.counters["captures"] += 1
            if self.event_sink:
                self.event_sink(
                    "capture",
                    key=key,
                    trace_id=tid,
                    seconds=time.monotonic() - capture_started,
                    batch=self.batch_size,
                    capacity=self.capacity,
                )
        if self.trace_prefill:
            for n in self.buckets:
                capture_started = time.monotonic()
                tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                self._recordable_prefill(n)
                ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
                self.traces[f"prefill_{n}"] = tid
                self.counters["captures"] += 1
                if self.event_sink:
                    self.event_sink(
                        "capture",
                        key=f"prefill_{n}",
                        trace_id=tid,
                        seconds=time.monotonic() - capture_started,
                        physical=n,
                        capacity=self.capacity,
                        bound=min(65536, self.capacity),
                        alignment=256,
                    )
        self.prepared = True
        self.reset()
        print("PREPARED_DECODE_AND_SAMPLING_TRACES", flush=True)

    def reset(self):
        for cache in self.state.layers:
            for tensor in cache:
                ttnn.multiply(tensor, 0.0, output_tensor=tensor)
        self.copy(torch.zeros(self.batch_size, dtype=torch.int32), self.positions)
        self.copy(torch.zeros(self.batch_size, dtype=torch.int32), self.rope_positions)
        self.copy(torch.zeros(1, 1, 1, self.batch_size, dtype=torch.int32), self.tokens)
        self.copy(torch.arange(self.batch_size, dtype=torch.int32), self.sampler._seeds)
        self.copy(torch.ones(self.batch_size, dtype=torch.int32), self.seed_increments)
        self.refresh_page_tables(self.initial_pages)
        self.counters["resets"] += 1

    def refresh_page_tables(self, page_table):
        if isinstance(page_table, torch.Tensor):
            page_table = {"full": page_table, "sliding": self.state.host_page_tables["sliding"]}
        for key, pages in page_table.items():
            if pages.shape != self.state.host_page_tables[key].shape:
                raise ValueError("Page table must retain the configured physical shape")
            physical_pages = (
                self.state.layers[key][0].shape[0]
                if isinstance(key, int)
                else self.batch_size * (min(self.capacity, 8704) if key == "sliding" else self.capacity) // 32
            )
            if pages.dtype != torch.int32 or (pages < 0).any() or (pages >= physical_pages).any():
                raise ValueError("Invalid physical page mapping")
            if not torch.equal(pages, self.copied_pages[key]):
                self.copy(pages, self.state.page_tables[key], "page_table_refreshes")
                self.state.host_page_tables[key] = pages.clone()
                self.copied_pages[key] = pages.clone()

    def set_sampling(self, *, top_k=1, top_p=0.0, temperature=1.0, seed=0, seed_offset=0):
        """Update persistent per-slot parameters at a request/scheduler boundary."""

        def rows(value, dtype):
            value = torch.as_tensor(value, dtype=dtype).flatten()
            if value.numel() == 1:
                value = value.repeat(self.batch_size)
            if value.numel() != self.batch_size:
                raise ValueError("Sampling parameters must be scalar or one per fixed slot")
            return value

        k = rows(top_k, torch.int32)
        p = rows(top_p, torch.float32)
        temp = rows(temperature, torch.float32)
        # UINT32_MAX tells manual_seed to preserve prior RNG state. Normalize
        # once at request setup, with room for every device-side increment in
        # the supported context, so no user seed can select that sentinel.
        seeds = torch.remainder(rows(seed, torch.int64), 1000000) + 1
        offsets = rows(seed_offset, torch.int64)
        if (offsets < 0).any() or (offsets > self.logical_capacity).any():
            raise ValueError(f"Sampling seed offsets outside cache capacity: {offsets.tolist()}")
        seeds += offsets
        if (
            not torch.isfinite(temp).all()
            or not torch.isfinite(p).all()
            or (temp < 0).any()
            or (p < 0).any()
            or (p > 1).any()
        ):
            raise ValueError("Invalid temperature or top_p")
        k = torch.where(temp == 0, 1, k)
        temp = torch.where(temp == 0, 1.0, temp)
        if (k < 1).any() or (k > 32).any():
            raise ValueError("The configured common sampler supports top_k in 1..32")
        self.copy(k, self.k)
        self.copy(p, self.p)
        self.copy(temp.reciprocal(), self.temperature)
        self.copy(seeds, self.sampler._seeds)
        self.counters["sampling_parameter_refreshes"] += 1

    def bind(self, tokens, positions, page_table=None):
        token_values = torch.as_tensor(tokens)
        position_values = torch.as_tensor(positions)
        if (token_values < 0).any() or (token_values >= 128000).any():
            raise ValueError("Token ID outside the checkpoint vocabulary")
        if (position_values < -1).any() or (position_values >= self.logical_capacity).any():
            raise ValueError("Decode position outside the configured cache")
        self.copy(
            torch.as_tensor(tokens, dtype=torch.int32).reshape(1, 1, 1, self.batch_size), self.tokens, "token_refreshes"
        )
        self.copy(
            torch.as_tensor(positions, dtype=torch.int32).reshape(self.batch_size), self.positions, "position_refreshes"
        )
        self.copy(
            torch.as_tensor(positions, dtype=torch.int32).reshape(self.batch_size),
            self.rope_positions,
            "rope_refreshes",
        )
        self.copy((position_values.flatten() >= 0).to(torch.int32), self.seed_increments)
        if page_table is not None:
            self.refresh_page_tables(page_table)

    def replay(self, *, sample=True, mode="split"):
        if not self.prepared:
            raise RuntimeError("prepare() must complete before replay")
        for key in ("decode", mode) if sample else ("decode",):
            ttnn.execute_trace(self.mesh_device, self.traces[key], cq_id=0, blocking=False)
            self.counters[key + "_replays"] += 1

    def read_tokens(self):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).flatten().to(torch.long)

    def read_logits(self, tensor=None):
        self.counters["logit_readbacks"] += 1
        return torch.cat(
            [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(self.logits if tensor is None else tensor)], -1
        )

    def _copy_prefill_input(self, bucket, key, host, target):
        signature = (bucket, key)
        previous = self.prefill_input_snapshots.get(signature)
        if previous is not None and previous.dtype == host.dtype and torch.equal(previous, host):
            self.counters["prefill_input_copies_skipped"] += 1
            return
        self.copy(host, target, "prefill_input_refreshes")
        self.prefill_input_snapshots[signature] = host.clone()

    def _prefill(self, values, *, slot=0, start_pos=0, return_all_logits=False):
        """Pad physical chunks internally; causal attention excludes future padding."""
        outputs = []
        offset = 0
        while offset < len(values):
            pos = start_pos + offset
            if pos % 32:
                # An unaligned continuation uses the inherited optimized decode path.
                row_tokens = torch.zeros(self.batch_size, dtype=torch.int32)
                row_pos = torch.full((self.batch_size,), -1, dtype=torch.int32)
                row_tokens[slot], row_pos[slot] = values[offset], pos
                self.bind(row_tokens, row_pos)
                self.replay(sample=False)
                if return_all_logits:
                    outputs.append(self.read_logits()[0, 0, slot : slot + 1])
                offset += 1
                continue
            allowed = max(n for n in self.buckets if n <= self.capacity - pos)
            count = min(allowed, len(values) - offset)
            if not self.owns_cache:
                # vLLM allocates only logical pages. Never pad a short request
                # into a larger bucket and overwrite unallocated page entries.
                allocated_extent = math.ceil(count / 32) * 32
                count = min(count, max(n for n in self.buckets if n <= allocated_extent))
            physical = min(n for n in self.buckets if n >= count)
            if self.event_sink:
                self.event_sink(
                    "prefill_chunk",
                    slot=slot,
                    start=pos,
                    logical=count,
                    physical=physical,
                    traces=self.traces,
                    counters=dict(self.counters),
                )
            buf = self.prefill_buffers[physical]
            ids = torch.zeros(1, physical, dtype=torch.int32)
            ids[0, :count] = torch.as_tensor(values[offset : offset + count])
            self._copy_prefill_input(physical, "tokens", ids, buf["tokens"])
            if self.prefill_rope_positions.get(physical) != pos:
                self.copy(torch.tensor([pos], dtype=torch.int32), buf["chunk_start"], "prefill_position_refreshes")
                phase = torch.arange(pos, pos + physical).float()[:, None] / (
                    10000.0 ** (torch.arange(0, 128, 2).float() / 128)
                )
                phase = torch.cat([phase, phase], -1)[None, None]
                self.copy(phase.cos().bfloat16(), buf["cos"], "prefill_rope_refreshes")
                self.copy(phase.sin().bfloat16(), buf["sin"], "prefill_rope_refreshes")
                self.prefill_rope_positions[physical] = pos
            for key, pages in self.state.host_page_tables.items():
                self._copy_prefill_input(physical, ("pages", key), pages[slot : slot + 1], buf["page_tables"][key])
                self._copy_prefill_input(
                    physical,
                    ("chunk_pages", key),
                    pages[slot : slot + 1, pos // 32 : (pos + physical) // 32],
                    buf["chunk_page_tables"][key],
                )
            alignment = max(a for a in (32, 64, 128, 256) if pos % a == 0)
            if self.trace_prefill and alignment == 256 and pos + physical <= 65536:
                ttnn.execute_trace(self.mesh_device, self.traces[f"prefill_{physical}"], cq_id=0, blocking=False)
                self.counters["prefill_replays"] += 1
                hidden = self.prefill_outputs[physical]
            else:
                hidden = self._prefill_bucket(physical, pos + physical, alignment)
            if return_all_logits:
                for tile_start in range(0, count, 32):
                    piece = ttnn.slice(hidden, (0, 0, tile_start, 0), (1, 1, tile_start + 32, 2560))
                    logits = self.model.terminal(piece)
                    outputs.append(self.read_logits(logits)[0, 0, : min(32, count - tile_start)])
                    del logits, piece
            del hidden
            offset += count
        return torch.cat(outputs, 0) if outputs else None

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        slots=None,
        output_mask=None,
        return_all_logits=False,
        sample_on_device=True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported prefill options: {sorted(kwargs)}")
        if tokens.ndim != 2:
            raise ValueError("Prefill tokens must have shape [requests, padded_prompt_length]")
        if kv_cache is not self.state and kv_cache is not self.state.layers:
            raise ValueError("External cache must be bound before trace preparation")
        self.prepare()
        self.refresh_page_tables(page_table)
        lens = torch.as_tensor(prompt_lens).tolist()
        if len(lens) != tokens.shape[0]:
            raise ValueError("Each request must have a logical prompt length")
        slots = list(range(len(lens))) if slots is None else list(slots)
        starts = [0] * len(lens) if start_pos is None else torch.as_tensor(start_pos).tolist()
        if len(slots) != len(lens) or len(starts) != len(lens) or len(set(slots)) != len(slots):
            raise ValueError("Prompt lengths, starts and unique fixed slots must match")
        if any(slot < 0 or slot >= self.batch_size for slot in slots):
            raise ValueError("Invalid fixed slot")
        if any(length < 0 or length > tokens.shape[1] for length in lens) or any(start < 0 for start in starts):
            raise ValueError("Invalid prompt extent")
        for row, length in enumerate(lens):
            self.validate_tokens(tokens[row, :length])
        if any(start + length > self.logical_capacity for start, length in zip(starts, lens)):
            raise ValueError("Prompt exceeds configured cache capacity")
        output_mask = [True] * len(lens) if output_mask is None else list(output_mask)
        if len(output_mask) != len(lens):
            raise ValueError("Each prefill row requires an output-mask entry")
        for row, (length, slot, start) in enumerate(zip(lens, slots, starts)):
            if not output_mask[row] and length:
                self._prefill(tokens[row, :length].tolist(), slot=slot, start_pos=start)
        if not return_all_logits:
            if not any(output_mask):
                return (
                    torch.zeros(self.batch_size, dtype=torch.long)
                    if sample_on_device
                    else torch.zeros(len(lens), 1, 128000)
                )
            last_tokens = torch.zeros(self.batch_size, dtype=torch.int32)
            last_positions = torch.full((self.batch_size,), -1, dtype=torch.int32)
            for row, (length, slot, start) in enumerate(zip(lens, slots, starts)):
                if length <= 0 or not output_mask[row]:
                    continue
                if start + length > self.logical_capacity:
                    raise ValueError("Prompt exceeds configured cache capacity")
                self._prefill(tokens[row, : length - 1].tolist(), slot=slot, start_pos=start)
                last_tokens[slot] = tokens[row, length - 1]
                last_positions[slot] = start + length - 1
            self.bind(last_tokens, last_positions)
            self.replay(sample=sample_on_device)
            if sample_on_device:
                return self.read_tokens()
            return self.read_logits()[0, 0, slots].unsqueeze(1)
        result = []
        for row, (length, slot, start) in enumerate(zip(lens, slots, starts)):
            if length <= 0 or not output_mask[row]:
                result.append(None)
                continue
            if start + length > self.logical_capacity:
                raise ValueError("Prompt exceeds configured cache capacity")
            value = self._prefill(tokens[row, :length].tolist(), slot=slot, start_pos=start, return_all_logits=True)
            result.append(value if return_all_logits else value[-1:])
        if return_all_logits:
            output = torch.zeros(len(lens), max(lens), 128000)
            for row, value in enumerate(result):
                if value is not None:
                    output[row, : len(value)] = value
            return output
        return torch.stack([v if v is not None else torch.zeros(1, 128000) for v in result])

    def decode_forward(
        self, tokens, start_pos, *, page_table, kv_cache, sample_on_device=True, read_from_device=True, **kwargs
    ):
        """Explicit scheduler state, or tokens/start_pos=None for device feedback."""
        if kwargs:
            raise TypeError(f"Unsupported decode options: {sorted(kwargs)}")
        if kv_cache is not self.state and kv_cache is not self.state.layers:
            raise ValueError("External cache must be bound before trace preparation")
        self.prepare()
        if tokens is None and start_pos is None:
            if page_table is not None:
                self.refresh_page_tables(page_table)
        elif tokens is None or start_pos is None:
            raise ValueError("Supply both token and position state, or retain both on device")
        else:
            self.bind(tokens, start_pos, page_table)
        self.replay(sample=sample_on_device)
        if not read_from_device:
            return self.tokens if sample_on_device else self.logits
        return self.read_tokens() if sample_on_device else self.read_logits()[0, 0]

    def prefill_logits(self, prompt_token_ids):
        self.validate_tokens(prompt_token_ids)
        self.prepare()
        self.reset()
        if not 0 < len(prompt_token_ids) <= self.logical_capacity:
            raise ValueError("Invalid prompt length")
        return self._prefill(prompt_token_ids, return_all_logits=True)[None]

    @staticmethod
    def validate_tokens(tokens):
        values = torch.as_tensor(tokens)
        if (values < 0).any() or (values >= 128000).any() or (values != values.long()).any():
            raise ValueError("Token IDs must be integers inside the checkpoint vocabulary")

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=False,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=0,
        stop_on_eos=True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported generation options: {sorted(kwargs)}")
        self.validate_tokens(prompt_token_ids)
        if host_sampling and not (temperature == 0 or top_k == 1):
            raise ValueError(
                "Host-sampling compatibility supports greedy argmax only; use traced sampling for top-k/top-p"
            )
        if not enable_trace:
            raise ValueError("Kolibri generation requires traced decode")
        if (
            max_new_tokens < 0
            or not prompt_token_ids
            or len(prompt_token_ids) + max(max_new_tokens - 1, 0) > self.logical_capacity
        ):
            raise ValueError("Generation exceeds configured cache capacity")
        self.prepare()
        started = time.monotonic()
        first_token = None
        initial_counters = self.counters.copy()
        self.reset()
        self.set_sampling(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
        if max_new_tokens <= 0:
            return []
        if host_sampling:
            self._prefill(prompt_token_ids[:-1])
            self.bind(
                [prompt_token_ids[-1]] + [0] * (self.batch_size - 1),
                [len(prompt_token_ids) - 1] + [-1] * (self.batch_size - 1),
            )
        else:
            initial = self.prefill_forward(
                torch.tensor([prompt_token_ids]),
                page_table=self.state.host_page_tables,
                kv_cache=self.state,
                prompt_lens=[len(prompt_token_ids)],
                slots=[0],
            )
        result = []
        for step in range(max_new_tokens):
            if step == 0 and not host_sampling:
                pred = int(initial[0])
            else:
                output = self.decode_forward(
                    None, None, page_table=None, kv_cache=self.state, sample_on_device=not host_sampling
                )
                pred = int(output[0].argmax()) if host_sampling else int(output[0])
            if first_token is None:
                first_token = time.monotonic()
            result.append(pred)
            if stop_on_eos and next_input is None and pred == self.model.config.eos_token_id:
                break
            if next_input is not None or host_sampling:
                value = next_input(step, pred) if next_input is not None else pred
                if step + 1 < max_new_tokens:
                    host = torch.zeros(1, 1, 1, self.batch_size, dtype=torch.int32)
                    host[0, 0, 0, 0] = value
                    self.copy(host, self.tokens, "token_refreshes")
        finished = time.monotonic()
        self.last_generation_metrics = dict(
            prompt_len=len(prompt_token_ids),
            gen_len=len(result),
            batch=1,
            ttft_ms=(first_token - started) * 1000,
            decode_ms_per_token=(finished - first_token) * 1000 / (len(result) - 1) if len(result) > 1 else None,
            teacher_forcing=next_input is not None,
            host_sampling=host_sampling,
            counters=dict(self.counters - initial_counters),
        )
        return result

    def close(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh_device, tid)
            self.counters["teardown_releases"] += 1
            if self.event_sink:
                self.event_sink("shutdown_release", trace_id=tid)
        self.traces.clear()

    def teardown(self):
        self.close()


def build_generator(model_dir, mesh_device, **kwargs):
    from models.autoports.aleph_alpha_kolibri_1_bf16.tt.precision import load_precision_config

    precision_config = load_precision_config(kwargs.pop("precision_config", None))
    capacity = kwargs.pop("capacity", precision_config["runtime"]["max_context"])
    batch_size = kwargs.pop("batch_size", 1)
    layer_indices = kwargs.pop("layer_indices", None)
    cache_state = kwargs.pop("cache_state", None)
    trace_prefill = kwargs.pop("trace_prefill", True)
    sharded_terminal_norm = kwargs.pop("sharded_terminal_norm", True)
    if kwargs:
        raise TypeError(f"Unsupported construction options: {sorted(kwargs)}")
    model = KolibriModel(
        mesh_device,
        layer_indices=layer_indices,
        rope_capacity=capacity,
        sharded_terminal_norm=sharded_terminal_norm,
        precision_config=precision_config,
    )
    return KolibriGenerator(
        model, batch_size=batch_size, capacity=capacity, cache_state=cache_state, trace_prefill=trace_prefill
    )

"""Persistent paged generator. Host planning is confined to request boundaries."""

import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

import ttnn
from agentic_models.models.nanbeige_4_3_33b.vllm.tt.cache import (
    advance_active_seeds,
    fill_partial,
    remap_counts,
    remap_vector,
    store_rows,
)
from agentic_models.models.nanbeige_4_3_33b.vllm.tt.model import Model
from agentic_models.models.nanbeige_4_3_33b.vllm.tt.prefill_plan import prefill_dependencies, ready_prefill_rows
from models.common.sampling.tt_penalties import TTPenalties
from models.common.sampling.tt_sampling import TTSampling


@dataclass
class DecodeState:
    tokens: object
    positions: object
    indices: object
    active: object
    table: object
    logits: object
    sampler: object
    trace: int = None
    sample_trace: int = None


@dataclass
class DecodeOutput:
    tensor: object
    rows: int
    is_tokens: bool


class Generator:
    # Seven q128 chunks per head occupy 98 cores without causal pair grouping.
    SINGLE_PREFILL_ROWS = 896

    def __init__(
        self,
        model_dir,
        mesh_device,
        *,
        max_batch_size=16,
        max_seq_len=131072,
        num_layers=None,
        num_pages=None,
        prepare=True,
        kv_cache=None,
    ):
        self.init_started = time.perf_counter()
        self.mesh = mesh_device
        self.slots = max_batch_size
        if max_batch_size not in (1, 2, 4, 8, 16):
            raise ValueError("Slot capacity must be a supported bucket")
        self.max_seq_len = max_seq_len
        self.buckets = [b for b in (1, 2, 4, 8, 16) if b <= self.slots]
        self.model = Model(
            model_dir,
            mesh_device,
            max_model_len=max_seq_len,
            num_layers=num_layers,
            num_pages=num_pages,
            kv_cache=kv_cache,
        )
        self.vocab_size = self.model.config.vocab_size
        self.page_columns = ((max_seq_len + 1023) // 1024 + 1) * 32
        self.stats = dict(
            prefill_q32_chunks=0,
            prefill_q128_chunks=0,
            decode_trace_replays=0,
            sampling_trace_replays=0,
            token_refreshes=0,
            position_refreshes=0,
            page_table_refreshes=0,
            host_sampled_steps=0,
            syncs=0,
            token_readbacks=0,
            logits_readbacks=0,
            captures=0,
            retirements=0,
        )
        self.prefill_inputs = {}
        self.decode_states = {}
        self.batch_prefill_inputs = {}
        self.terminal_hidden = self.model.put(torch.zeros(1, 1, 32, self.model.config.hidden_size))
        self.zero_kv_tile = self.model.put(torch.zeros(1, 2, 32, 128))
        self.terminal_map = self.ints(torch.full((32,), -1, dtype=torch.int32))
        self.slot_map = self.ints(torch.arange(32, dtype=torch.int32))
        self.current_bucket = None
        self.next_page = 64  # startup uses only these reserved scratch pages
        for size in (32, 64, 128, 256, 512, self.SINGLE_PREFILL_ROWS, 1024):
            self.prefill_inputs[size] = dict(
                tokens=self.ints(torch.zeros(1, size), unsigned=True),
                indices=self.ints(torch.arange(size).reshape(1, size), unsigned=True),
                start=self.ints([0]),
                metadata=self.ints([0, 0, size]),
                table=self.ints(torch.arange(self.page_columns).reshape(1, -1) % 32 + 1),
                select=self.ints(torch.zeros(1, 32), unsigned=True),
            )
        for b in self.buckets[1:]:
            for size in (32, 64, 128, 256, 512):
                if size * b > 1024:
                    continue
                self.batch_prefill_inputs[b, size] = dict(
                    tokens=self.ints(torch.zeros(1, b * size), unsigned=True),
                    indices=self.ints(torch.arange(size).repeat(b).reshape(1, -1), unsigned=True),
                    start=self.ints(torch.zeros(b)),
                    metadata=self.ints([0, 0, size] * b),
                    table=self.ints(
                        torch.arange(self.page_columns).reshape(1, -1).repeat(b, 1) % (size // 32)
                        + torch.arange(b).reshape(-1, 1) * (size // 32)
                        + 1
                    ),
                    select=self.ints(torch.zeros(1, 32), unsigned=True),
                    batch=b,
                )
        args = SimpleNamespace(
            vocab_size=self.vocab_size,
            padded_vocab_size=self.vocab_size,
            cluster_shape=(1, 4),
            max_batch_size=32,
            max_top_k=64,
            pad_logits_to_power_of_2=True,
        )
        self.penalties = {mode: TTPenalties(self.mesh, args) for mode in (0, 1)}
        self.penalty_map = self.ints(torch.arange(32).reshape(1, 32), unsigned=True)
        self.penalty_active = [1] * 32
        for p in self.penalties.values():
            p.reset_params([0.0] * 32, [0.0] * 32, [1.0] * 32)
        for b in [0] + self.buckets:
            physical = max(1, b)
            sampler = TTSampling(
                args=args,
                mesh_device=self.mesh,
                tt_ccl=None,
                k=torch.full((32,), 64),
                p=torch.full((32,), 0.95),
                temp=torch.ones(32),
            )
            self.decode_states[b] = DecodeState(
                self.ints(torch.arange(32).reshape(1, 1, 1, 32) + 100, unsigned=True),
                self.ints(torch.zeros(physical)),
                self.ints(torch.zeros(1, physical), unsigned=True),
                self.ints(torch.ones(physical)),
                self.ints(torch.arange(physical).reshape(-1, 1).expand(physical, self.page_columns) + 33),
                self.model.put(torch.zeros(1, 1, 32, self.vocab_size), dim=-1),
                sampler,
            )
        self.prepared = False
        if prepare:
            self.prepare()

    def ints(self, value, unsigned=False):
        return self.model.put(
            torch.as_tensor(value, dtype=torch.int64 if unsigned else torch.int32),
            dtype=ttnn.uint32 if unsigned else ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    @staticmethod
    def refresh(value, target):
        h = ttnn.from_torch(torch.as_tensor(value).contiguous(), dtype=target.dtype, layout=target.layout)
        ttnn.copy_host_to_device_tensor(h, target)

    @staticmethod
    def prefill_program(width, starts):
        # Query origins, not logical request lengths, must satisfy SDPA's
        # query-chunk alignment. Retain the general path for arbitrary resumes.
        return (128, 512) if width >= 128 and all(start % 128 == 0 for start in starts) else (32, 256)

    def _prefill(self, e, sdpa_chunks=(32, 256)):
        self.stats[f"prefill_q{sdpa_chunks[0]}_chunks"] += 1
        return self.model.prefill(
            e["tokens"],
            e["indices"],
            e["start"],
            e["metadata"],
            e["table"],
            batch=e.get("batch", 1),
            sdpa_chunks=sdpa_chunks,
        )

    def _select_hidden(self, hidden, selection):
        # Normalize the FP32 residual before selecting BF16 terminal rows.
        hidden = self.model.normalize(hidden)
        n = hidden.shape[2]
        weights = ttnn.reshape(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), [n, self.model.config.hidden_size])
        chosen = ttnn.embedding(selection, weights, layout=ttnn.TILE_LAYOUT)
        chosen = ttnn.reshape(chosen, [1, 1, 32, self.model.config.hidden_size])
        return chosen

    def _head(self, hidden, selection):
        return self.model.project(self._select_hidden(hidden, selection))

    def _decode(self, b):
        e = self.decode_states[b]
        tok = ttnn.reshape(e.tokens, [1, 32])
        y = self.model.decode(tok, e.positions, e.indices, e.active, e.table, b)
        if b != 32:
            y = ttnn.pad(y, [(0, 0), (0, 0), (0, 32 - b), (0, 0)], value=0.0)
        ttnn.copy(y, e.logits)
        ttnn.plus_one(e.positions, skip_negative_entries=True)
        ttnn.plus_one(e.indices)

    def _sample(self, b):
        e = self.decode_states[b]
        penalty = self.penalties[int(b != 0)]
        penalized = penalty.apply(e.logits)
        e.sampler(penalized, tt_out_tok=e.tokens)
        if b:
            penalty.update_output_tokens(e.tokens)
            advance_active_seeds(e.sampler.seeds_tt_tensor, e.active)

    def prepare(self):
        if self.prepared:
            return
        start = time.perf_counter()
        # Every program and persistent allocation precedes every capture.
        for size, e in self.prefill_inputs.items():
            for chunks in [(32, 256)] + ([(128, 512)] if size >= 128 else []):
                h = self._prefill(e, chunks)
                chosen = self._select_hidden(h, e["select"])
                store_rows(chosen, self.terminal_hidden, self.terminal_map)
                logits = self.model.project(chosen)
                del h, chosen, logits
            print("WARM_PREFILL", size, flush=True)
        for key, e in self.batch_prefill_inputs.items():
            for chunks in [(32, 256)] + ([(128, 512)] if key[1] >= 128 else []):
                h = self._prefill(e, chunks)
                chosen = self._select_hidden(h, e["select"])
                store_rows(chosen, self.terminal_hidden, self.terminal_map)
                logits = self.model.project(self.terminal_hidden)
                del h, chosen, logits
            print("WARM_BATCH_PREFILL", key, flush=True)
        for b in self.buckets:
            self._decode(b)
            self._sample(b)
            print("WARM_DECODE", b, flush=True)
        self._sample(0)
        for old in self.buckets:
            for new in self.buckets:
                self._remap(old, new)
        self._remap_penalties(list(range(32)))
        for penalty in self.penalties.values():
            penalty.reset_output_tokens()
        ttnn.synchronize_device(self.mesh)
        self.programs_before_capture = self.mesh.num_program_cache_entries()
        for b, e in self.decode_states.items():
            # Warmup wrote only reserved pages, never an allocated request page.
            self.refresh(torch.arange(32).reshape(1, 1, 1, 32) + 100, e.tokens)
            self.refresh(torch.zeros(max(1, b), dtype=torch.int32), e.positions)
            self.refresh(torch.zeros(1, max(1, b), dtype=torch.int64), e.indices)
            self.refresh(torch.arange(32), e.sampler.seeds_tt_tensor)
            if b:
                e.trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._decode(b)
                ttnn.end_trace_capture(self.mesh, e.trace, cq_id=0)
            e.sample_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._sample(b)
            ttnn.end_trace_capture(self.mesh, e.sample_trace, cq_id=0)
            self.stats["captures"] += 2 if b else 1
        ttnn.execute_trace(self.mesh, self.decode_states[0].sample_trace, cq_id=0, blocking=False)
        self.stats["sampling_trace_replays"] += 1
        for b in self.buckets:
            self.step(b)
        ttnn.synchronize_device(self.mesh)
        for penalty in self.penalties.values():
            penalty.reset_output_tokens()
        self.prepared = True
        self.prepare_seconds = time.perf_counter() - start
        self.startup_seconds = time.perf_counter() - self.init_started
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        self.model.memory_budget["free_after_prepare_bytes"] = int(
            (view.total_bytes_per_bank - view.total_bytes_allocated_per_bank) * view.num_banks
        )
        print("PREPARED", self.startup_seconds, self.stats, flush=True)

    def allocate_pages(self, lengths):
        table = torch.zeros(len(lengths), self.page_columns, dtype=torch.int32)
        for row, length in enumerate(lengths):
            if not 0 < int(length) <= self.max_seq_len:
                raise ValueError("Request exceeds the configured context")
            n = (int(length) + 31) // 32
            if n > self.page_columns or self.next_page + n > self.model.num_pages:
                raise MemoryError(f"KV request needs {n} pages; available {self.model.num_pages-self.next_page}")
            table[row, :n] = torch.arange(self.next_page, self.next_page + n)
            self.next_page += n
        return table

    def reset(self):
        # Standalone ownership only: caller guarantees no live request remains.
        self.next_page = 64

    def read_logits(self, logits):
        self.stats["logits_readbacks"] += 1
        return torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(logits)], dim=-1)

    def prefill_sequence(self, tokens, table, *, start=0, positions=None, last_logits=False):
        """Fixed physical chunks; absolute positions may start inside a page."""
        end = len(tokens)
        requested = [] if positions is None else list(positions)
        results = {}
        while start < end:
            skip = start % 32
            n = min(self.SINGLE_PREFILL_ROWS - skip, end - start)
            # A fixed dense workspace preserves reduction geometry across cache hits.
            size = self.SINGLE_PREFILL_ROWS
            e = self.prefill_inputs[size]
            host = torch.zeros(1, size, dtype=torch.int64)
            host[0, skip : skip + n] = torch.as_tensor(tokens[start : start + n])
            self.refresh(host, e["tokens"])
            self.refresh(torch.arange(start - skip, start - skip + size).reshape(1, -1), e["indices"])
            self.refresh(torch.tensor([start - skip], dtype=torch.int32), e["start"])
            self.refresh(torch.tensor([start - skip, skip, n], dtype=torch.int32), e["metadata"])
            self.refresh(table, e["table"])
            hidden = self._prefill(e, self.prefill_program(size, [start - skip]))
            selected = [p for p in requested if start < p <= start + n]
            if last_logits and start + n == end and end not in selected:
                selected.append(end)
            for off in range(0, len(selected), 32):
                group = selected[off : off + 32]
                idx = [p - 1 - start + skip for p in group]
                idx += [idx[-1]] * (32 - len(idx))
                self.refresh(torch.tensor([idx]), e["select"])
                logits = self._head(hidden, e["select"])
                out = self.read_logits(logits)[0, 0]
                for i, p in enumerate(group):
                    results[p] = out[i].numpy().copy()
                del logits, out
            del hidden
            start += n
        return np.stack([results[p] for p in requested]) if requested else results

    def prefill_logits(self, tokens, positions=None, *, resume_at=None):
        self.reset()
        table = self.allocate_pages([len(tokens)])
        if resume_at:
            self.prefill_sequence(tokens[:resume_at], table)
        all_rows = positions is None
        positions = list(range(1, len(tokens) + 1)) if positions is None else list(positions)
        result = self.prefill_sequence(tokens, table, start=resume_at or 0, positions=positions)
        return torch.from_numpy(result)[None] if all_rows else result

    def prefill_logits_continuation(self, prompt_token_ids, start, *, positions=None, prefill_prefix=True):
        # Readiness uses output indices; toolkit uses predicted-token positions.
        positions = list(range(start, len(prompt_token_ids))) if positions is None else list(positions)
        self.reset()
        table = self.allocate_pages([len(prompt_token_ids)])
        if prefill_prefix and start:
            self.prefill_sequence(prompt_token_ids[:start], table)
        elif start:
            # Page zero is permanently zero and is excluded from the allocator and warmup.
            # Give the continuation a fresh zero tail page even at an unaligned start.
            table[0, : start // 32] = 0
            if start % 32:
                for layer in self.model.layers:
                    e = self.prefill_inputs[32]
                    self.refresh(torch.tensor([start - start % 32, 0, start % 32], dtype=torch.int32), e["metadata"])
                    self.refresh(table, e["table"])
                    fill_partial(layer.key, self.zero_kv_tile, e["table"], e["metadata"])
                    fill_partial(layer.value, self.zero_kv_tile, e["table"], e["metadata"])
        arr = self.prefill_sequence(prompt_token_ids, table, start=start, positions=[p + 1 for p in positions])
        return torch.from_numpy(arr)[None]

    def setup_decode(
        self,
        tokens,
        positions,
        table,
        *,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        seeds=None,
        reset_seeds=True,
        update_sampling=True,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    ):
        n = len(tokens)
        b = next(b for b in self.buckets if b >= n)
        e = self.decode_states[b]
        padded = list(tokens) + [tokens[-1]] * (32 - n)
        pos = list(positions) + [-1] * (b - n)
        pages = torch.zeros(b, self.page_columns, dtype=torch.int32)
        pages[:n] = table[:n]
        if n < b:
            pages[n:] = table[n - 1]
        self.refresh(torch.tensor(padded).reshape(1, 1, 1, 32), e.tokens)
        self.refresh(torch.tensor(pos, dtype=torch.int32), e.positions)
        self.refresh(torch.tensor([max(0, p) for p in pos]).reshape(1, b), e.indices)
        self.refresh(torch.tensor([int(p >= 0) for p in pos], dtype=torch.int32), e.active)
        self.refresh(pages, e.table)
        self.stats["token_refreshes"] += 1
        self.stats["position_refreshes"] += 1
        self.stats["page_table_refreshes"] += 1
        self.set_sampling(
            b,
            n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=seeds,
            seed_offsets=[max(0, int(p) + 1) for p in positions],
            reset_seeds=reset_seeds,
            update_params=update_sampling,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )
        active = [int(p >= 0) for p in pos] + [0] * (32 - b)
        self.penalty_active = active
        self.refresh(torch.tensor(active, dtype=torch.int32).reshape(32, 1), self.penalties[1].decode_src)
        self.current_bucket = b
        return b

    def set_sampling(
        self,
        b,
        n,
        *,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        seeds=None,
        seed_offsets=None,
        reset_seeds=False,
        update_params=True,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    ):
        def rows(value, default):
            if isinstance(value, (list, tuple, torch.Tensor)):
                value = torch.as_tensor(value).reshape(-1).tolist()[:n]
            else:
                value = [value] * n
            if len(value) != n:
                raise ValueError("Sampling parameter row count differs from batch")
            return value + [default] * (32 - n)

        temperatures = rows(temperature, 1.0)
        ks = rows(top_k, 64)
        ps = rows(top_p, 0.95)
        ks = [1 if t == 0 else (64 if k < 1 else int(k)) for t, k in zip(temperatures, ks)]
        if any(k > 64 for k in ks):
            raise NotImplementedError("Device sampler supports top_k <=64 or UNSET")
        ps = [0.0 if t == 0 else p for t, p in zip(temperatures, ps)]
        # The native sampler multiplies logits by this buffer; the API supplies T.
        temperatures = [1.0 if t == 0 else 1.0 / t for t in temperatures]
        if update_params:
            self.penalties[int(b != 0)].reset_params(
                rows(presence_penalty, 0.0), rows(frequency_penalty, 0.0), rows(repetition_penalty, 1.0)
            )
            self.decode_states[b].sampler.reset_params(
                k=torch.tensor(ks), p=torch.tensor(ps), temp=torch.tensor(temperatures)
            )
        if reset_seeds:
            seed_rows = [None] * n if seeds is None else list(seeds)[:n]
            if len(seed_rows) != n:
                raise ValueError("Seed row count differs from batch")
            seed_rows = [int(torch.randint(0, 2**31, ()).item()) if s is None else int(s) for s in seed_rows]
            # Explicit seeds identify a request's stream. Its counter is the
            # absolute position of the token being predicted, so rebuilding
            # after scheduler transitions matches uninterrupted decode.
            if seed_offsets is not None:
                if len(seed_offsets) != n:
                    raise ValueError("Seed offset row count differs from batch")
                seed_rows = [s + int(offset) for s, offset in zip(seed_rows, seed_offsets)]
            # manual_seed reserves UINT32_MAX as "do not reseed".
            seed_rows = [s % (2**32 - 1) for s in seed_rows]
            self.refresh(torch.tensor(seed_rows + [0] * (32 - n)), self.decode_states[b].sampler.seeds_tt_tensor)

    @staticmethod
    def sampling_kwargs(params):
        if isinstance(params, dict):
            values = dict(params)
            for name, neutral in [
                ("min_p", 0.0),
                ("min_tokens", 0),
            ]:
                value = values.pop(name, neutral)
                if torch.as_tensor(value).ne(neutral).any():
                    raise NotImplementedError(f"Non-neutral {name} is not enabled on device")
            if set(values) - {
                "temperature",
                "top_p",
                "top_k",
                "seeds",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
            }:
                raise NotImplementedError(
                    f"Unsupported sampling fields: {sorted(set(values) - {'temperature', 'top_p', 'top_k', 'seeds'})}"
                )
            return values
        for name, neutral in [
            ("min_p", 0.0),
            ("min_tokens", 0),
        ]:
            values = getattr(params, name, None)
            if values is not None and torch.as_tensor(values).ne(neutral).any():
                raise NotImplementedError(f"Non-neutral {name} is not enabled on device")
        if any(getattr(params, "enable_log_probs", []) or []):
            raise NotImplementedError("Device logprob reporting is not enabled")
        return dict(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            seeds=params.seed,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            repetition_penalty=params.repetition_penalty,
        )

    def _remap_penalties(self, remap, live_rows=None):
        indices = [int(i) for i in remap] + list(range(len(remap), 32))
        if len(indices) != 32 or any(i < 0 or i >= 32 for i in indices):
            raise ValueError("Invalid penalty-state slot mapping")
        self.refresh(torch.tensor([indices]), self.penalty_map)
        p = self.penalties[1]
        for tensor in (p.prompt_mask, p.output_mask, p.output_counts, p.output_counts_gathered):
            remap_counts(tensor, self.penalty_map)
        if live_rows is not None:
            self.penalty_active = [self.penalty_active[i] if row < live_rows else 0 for row, i in enumerate(indices)]
            self.refresh(torch.tensor(self.penalty_active, dtype=torch.int32).reshape(32, 1), p.decode_src)

    def _remap(self, old, new):
        src = self.decode_states[old]
        dst = self.decode_states[new]
        for x, y in [
            (src.tokens, dst.tokens),
            (src.positions, dst.positions),
            (src.indices, dst.indices),
            (src.active, dst.active),
            (src.sampler.seeds_tt_tensor, dst.sampler.seeds_tt_tensor),
        ]:
            remap_vector(x, y, self.slot_map)

    def _remap_slots(self, new, slot_remap, live_rows=None):
        # Internal partial move: decode_forward reloads tables and policies before replay.
        n = new if live_rows is None else live_rows
        mapping = list(slot_remap)
        if self.current_bucket is None:
            raise ValueError("Cannot remap before request setup")
        if len(mapping) < n or len(mapping) > self.slots or any(i < 0 or i >= self.slots for i in mapping):
            raise ValueError("Invalid source slot")
        # Full-capacity permutations also carry dormant RNG counters between buckets.
        mapping += list(range(len(mapping), self.slots))
        self.refresh(torch.tensor(mapping + [-1] * (32 - self.slots), dtype=torch.int32), self.slot_map)
        self._remap(self.current_bucket, new)
        self.refresh(torch.tensor([1] * n + [0] * (new - n), dtype=torch.int32), self.decode_states[new].active)
        self.current_bucket = new

    def step(self, bucket, *, sample=True, blocking=False):
        e = self.decode_states[bucket]
        ttnn.execute_trace(self.mesh, e.trace, cq_id=0, blocking=False)
        self.stats["decode_trace_replays"] += 1
        if sample:
            ttnn.execute_trace(self.mesh, e.sample_trace, cq_id=0, blocking=blocking)
            self.stats["sampling_trace_replays"] += 1
        elif blocking:
            ttnn.synchronize_device(self.mesh)
            self.stats["syncs"] += 1
        return e.tokens if sample else e.logits

    def decode_logits(
        self, tokens, start, steps, batch_rows=1, filler=None, rows=None, target_row=0, filler_starts=None
    ):
        self.reset()
        filler = filler or []
        filler_starts = filler_starts or [start] * len(filler)
        seqs = list(filler)
        starts = list(filler_starts)
        seqs.insert(target_row, tokens)
        starts.insert(target_row, start)
        if len(seqs) != batch_rows:
            raise ValueError("Each active row needs its own teacher-forcing sequence")
        table = self.allocate_pages([s + steps for s in starts])
        history = torch.zeros(batch_rows, max(starts), dtype=torch.int64)
        for i, (seq, length) in enumerate(zip(seqs, starts)):
            history[i, :length] = torch.as_tensor(seq[:length])
        self.prefill_forward(history, page_table=table, prompt_lens=starts)
        b = self.setup_decode([s[p] for s, p in zip(seqs, starts)], starts, table)
        selected = set(range(steps) if rows is None else rows)
        result = {}
        began = time.perf_counter()
        for j in range(steps):
            if j and j % 1024 == 0:
                print("DECODE_PROGRESS", j, steps, round(time.perf_counter() - began, 1), flush=True)
            if j:
                values = [seq[p + j] for seq, p in zip(seqs, starts)]
                values += [values[-1]] * (32 - len(values))
                self.refresh(torch.tensor(values).reshape(1, 1, 1, 32), self.decode_states[b].tokens)
                self.stats["token_refreshes"] += 1  # explicit teacher-forcing, never token-out
            logits = self.step(b, sample=False)
            if j in selected:
                result[j] = self.read_logits(logits)[0, 0, target_row].numpy().copy()
        return np.stack([result[j] for j in (range(steps) if rows is None else rows)])

    @property
    def kv_cache(self):
        return [(layer.key, layer.value) for layer in self.model.layers]

    def _check_cache(self, cache):
        if cache is not None and (
            len(cache) != len(self.model.layers)
            or any(k is not layer.key or v is not layer.value for (k, v), layer in zip(cache, self.model.layers))
        ):
            raise ValueError("KV storage must be bound before generator preparation")

    def _pad_table(self, table):
        table = torch.as_tensor(table, dtype=torch.int32)
        if table.ndim != 2 or (table < 0).any() or (table >= self.model.num_pages).any():
            raise ValueError("Invalid physical KV page table")
        if table.shape[1] > self.page_columns:
            raise ValueError("Page table exceeds configured context")
        return torch.nn.functional.pad(table, (0, self.page_columns - table.shape[1]))

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache=None,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        return_all_logits=False,
        sampling_mask=None,
        original_prompt_lens=None,
        **kwargs,
    ):
        self._check_cache(kv_cache)
        tokens = torch.as_tensor(tokens)
        nrows = len(prompt_lens)
        if not 0 < nrows <= self.slots:
            raise ValueError("Prefill exceeds slot capacity")
        ends = [int(x) for x in prompt_lens]
        starts = [0] * nrows if start_pos is None else [int(x) for x in start_pos]
        if len(starts) != nrows or tokens.shape[0] < nrows or tokens.shape[1] < max(ends):
            raise ValueError("Prefill tokens/lengths/start rows do not agree")
        if any(s < 0 or n <= s or n > self.max_seq_len for s, n in zip(starts, ends)):
            raise ValueError("Invalid prefill intervals")
        table = self._pad_table(page_table)[:nrows]
        if table.shape[0] != nrows:
            raise ValueError("Page table has fewer rows than the prefill")
        dependencies = prefill_dependencies(
            tokens,
            table,
            starts,
            ends,
            first_page=0 if self.model.memory_budget["ownership"] == "external" else 1,
            num_pages=self.model.num_pages,
        )
        if return_all_logits:
            if nrows != 1:
                raise NotImplementedError("All-position logits are a single-request diagnostic")
            arr = self.prefill_sequence(
                tokens[0, : ends[0]].tolist(), table, start=starts[0], positions=list(range(starts[0] + 1, ends[0] + 1))
            )
            return torch.from_numpy(arr)[None]
        while any(s < n for s, n in zip(starts, ends)):
            live = ready_prefill_rows(starts, ends, dependencies)
            if not live:
                raise RuntimeError("Prefill dependencies made no progress")
            b = next(b for b in self.buckets if b >= len(live))
            limit = self.SINGLE_PREFILL_ROWS if b == 1 else 1024 // b
            lengths = [min(limit - starts[i] % 32, ends[i] - starts[i]) for i in live]
            width = limit
            e = self.prefill_inputs[width] if b == 1 else self.batch_prefill_inputs[b, width]
            host = torch.zeros(b, width, dtype=torch.int64)
            idx = torch.zeros(b, width, dtype=torch.int64)
            pt = torch.zeros(b, self.page_columns, dtype=torch.int32)
            meta = []
            aligned = []
            select = []
            mapping = []
            for row in range(b):
                if row < len(live):
                    i = live[row]
                    length = lengths[row]
                    start = starts[i]
                    skip = start % 32
                    host[row, skip : skip + length] = tokens[i, start : start + length]
                    origin = start - skip
                    idx[row] = torch.arange(origin, origin + width)
                    pt[row] = table[i]
                    meta.extend([origin, skip, length])
                    aligned.append(origin)
                    if start + length == ends[i]:
                        select.append(row * width + skip + length - 1)
                        mapping.append(i)
                else:
                    meta.extend([0, 0, 0])
                    aligned.append(0)
            self.refresh(host.reshape(1, -1), e["tokens"])
            self.refresh(idx.reshape(1, -1), e["indices"])
            self.refresh(torch.tensor(meta, dtype=torch.int32), e["metadata"])
            self.refresh(torch.tensor(aligned, dtype=torch.int32), e["start"])
            self.refresh(pt, e["table"])
            hidden = self._prefill(e, self.prefill_program(width, aligned))
            if select:
                self.refresh(torch.tensor([select + [select[-1]] * (32 - len(select))]), e["select"])
                self.refresh(torch.tensor(mapping + [-1] * (32 - len(mapping)), dtype=torch.int32), self.terminal_map)
                chosen = self._select_hidden(hidden, e["select"])
                store_rows(chosen, self.terminal_hidden, self.terminal_map)
                del chosen
            del hidden
            for i, length in zip(live, lengths):
                starts[i] += length
        # Intermediate chunks publish no token and must not consume RNG state.
        # The prefill sampler uses independently reset per-row seeds and never
        # advances resident decode RNG. Masked rows use deterministic placeholders.
        if sampling_params is not None and sampling_mask is not None and not torch.as_tensor(sampling_mask).any():
            return torch.zeros(nrows, 1, dtype=torch.int64)
        logits = self.model.project(self.terminal_hidden)
        if sampling_params is None:
            out = self.read_logits(logits)[0, 0, :nrows].unsqueeze(1)
            del logits
        else:
            b = 0
            original = ends if original_prompt_lens is None else original_prompt_lens
            prompt_history = torch.full_like(tokens, -1)
            output_history = torch.full_like(tokens, -1)
            for row, (length, end) in enumerate(zip(original, ends)):
                prompt_history[row, : min(int(length), end)] = tokens[row, : min(int(length), end)]
                output_history[row, int(length) : end] = tokens[row, int(length) : end]
            self.penalties[0].reset_prompt_tokens(prompt_history)
            self.penalties[0].reset_output_tokens(output_history)
            params = self.sampling_kwargs(sampling_params)
            if sampling_mask is not None:
                mask = torch.as_tensor(sampling_mask, dtype=torch.bool)
                temperatures = torch.as_tensor(params.get("temperature", 1.0)).expand(nrows).clone()
                temperatures[~mask] = 0
                params["temperature"] = temperatures.tolist()
            self.set_sampling(b, nrows, **params, seed_offsets=ends, reset_seeds=True)
            e = self.decode_states[0]
            ttnn.copy(logits, e.logits)
            del logits
            ttnn.execute_trace(self.mesh, e.sample_trace, cq_id=0, blocking=False)
            self.stats["sampling_trace_replays"] += 1
            out = self.read_tokens(b, nrows)
        return out

    def read_tokens(self, bucket, rows):
        self.stats["token_readbacks"] += 1
        return (
            ttnn.to_torch(ttnn.get_device_tensors(self.decode_states[bucket].tokens)[0])
            .reshape(-1)[:rows]
            .long()
            .reshape(rows, 1)
        )

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache=None,
        enable_trace=True,
        reload_inputs=True,
        reload_page_table=False,
        sampling_params=None,
        read_from_device=True,
        slot_remap=None,
        reload_sampling_params=False,
        reset_sampling_state=False,
        prompt_tokens=None,
        output_tokens=None,
        **kwargs,
    ):
        self._check_cache(kv_cache)
        if "reset_batch" in kwargs:
            raise ValueError("Legacy reset_batch is incompatible with the version-1 reload contract")
        if not enable_trace:
            raise NotImplementedError("Production decode requires trace replay")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Resetting sampling state requires authoritative input positions")
        n = torch.as_tensor(tokens).shape[0]
        b = next(b for b in self.buckets if b >= n)
        params = self.sampling_kwargs(sampling_params) if sampling_params is not None else {}
        first_setup = self.current_bucket is None
        if first_setup and not reload_inputs:
            raise ValueError("First decode requires authoritative forward inputs")
        if not reload_inputs and slot_remap is not None and any(i >= self.current_bucket for i in list(slot_remap)[:n]):
            raise ValueError("Bringing dormant rows into decode requires authoritative forward inputs")
        layout_changed = not first_setup and (
            b != self.current_bucket or (slot_remap is not None and list(slot_remap)[:n] != list(range(n)))
        )
        # Tokens/positions/RNG move on device; tables and sampling policies must
        # arrive in the new request order. Reject before changing any state.
        if layout_changed and not (reload_inputs or reload_page_table):
            raise ValueError("Slot layout changes require authoritative page tables")
        if layout_changed and sampling_params is not None and not reload_sampling_params:
            raise ValueError("Slot layout changes require authoritative sampling parameters")
        if slot_remap is not None and not first_setup:
            self._remap_slots(b, slot_remap, n)
            self._remap_penalties(list(slot_remap), n)
        elif not first_setup and b != self.current_bucket:
            if not reload_inputs:
                raise ValueError("A bucket change with resident inputs requires a slot remap")
            self._remap_slots(b, list(range(n)), n)
        if reload_inputs:
            b = self.setup_decode(
                torch.as_tensor(tokens).reshape(-1).tolist(),
                torch.as_tensor(start_pos).tolist(),
                self._pad_table(page_table),
                **params,
                reset_seeds=first_setup or reset_sampling_state,
                update_sampling=first_setup or reload_sampling_params,
            )
        elif reload_page_table:
            pages = torch.zeros(b, self.page_columns, dtype=torch.int32)
            pages[:n] = self._pad_table(page_table)[:n]
            self.refresh(pages, self.decode_states[b].table)
            self.stats["page_table_refreshes"] += 1
        if not reload_inputs and (reload_sampling_params or reset_sampling_state):
            self.set_sampling(b, n, **params, reset_seeds=reset_sampling_state, update_params=reload_sampling_params)
        if reset_sampling_state or first_setup:
            history = torch.full((n, 1), -1, dtype=torch.int64)
            self.penalties[1].reset_prompt_tokens(history if prompt_tokens is None else prompt_tokens)
            self.penalties[1].reset_output_tokens(history if output_tokens is None else output_tokens)
        result = self.step(b, sample=sampling_params is not None)
        if not read_from_device:
            return DecodeOutput(result, n, sampling_params is not None)
        if sampling_params is not None:
            return self.read_tokens(b, n)
        return self.read_logits(result)[0, 0, :n].unsqueeze(1)

    def read_decode_output(self, output, async_read=True):
        # Enqueue the copy before another step may overwrite the resident output.
        host = ttnn.from_device(output.tensor, blocking=not async_read)
        events = [ttnn.record_event(self.mesh, 0)] if async_read else []
        key = "token_readbacks" if output.is_tokens else "logits_readbacks"
        self.stats[key] += 1
        return DecodeOutput(host, output.rows, output.is_tokens), events

    def process_decode_output_host(self, output, is_tokens=None):
        shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output.tensor)]
        if output.is_tokens:
            return shards[0].reshape(-1)[: output.rows].long().reshape(-1, 1)
        return torch.cat(shards, dim=-1).float()[0, 0, : output.rows].unsqueeze(1)

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        temperature=0.0,
        top_p=0.95,
        top_k=-1,
        seed=0,
        **kwargs,
    ):
        if not enable_trace:
            raise NotImplementedError("Production generation requires trace replay")
        if max_new_tokens < 1:
            return []
        self.reset()
        table = self.allocate_pages([len(prompt_token_ids) + max_new_tokens])
        first = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=table,
            prompt_lens=[len(prompt_token_ids)],
            sampling_params=dict(temperature=temperature, top_p=top_p, top_k=top_k, seeds=[seed]),
        )
        self.setup_decode(
            [int(first[0, 0])],
            [len(prompt_token_ids)],
            table,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=[seed],
        )
        result = [int(first[0, 0])]
        eos = self.model.config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for step in range(max_new_tokens - 1):
            if next_input is None and result[-1] in eos:
                break
            if next_input is not None:
                forced = next_input(step, result[-1])
                self.refresh(torch.full((1, 1, 1, 32), forced), self.decode_states[1].tokens)
                self.stats["token_refreshes"] += 1
            self.step(1)
            result.append(int(self.read_tokens(1, 1)[0, 0]))
        return result

    def close(self):
        for e in self.decode_states.values():
            for trace in (e.trace, e.sample_trace):
                if trace is not None:
                    ttnn.release_trace(self.mesh, trace)
            e.trace = e.sample_trace = None

    teardown = close


def build_generator(model_dir, mesh_device, **kwargs):
    import json
    from pathlib import Path

    path = Path(model_dir)
    if (path / "doc/targets.json").exists():
        path = Path(json.loads((path / "doc/targets.json").read_text())["model"]["local_dir"])
    gen = Generator(path, mesh_device, **kwargs)
    from transformers import AutoTokenizer

    gen.tokenizer = AutoTokenizer.from_pretrained(path)
    return gen

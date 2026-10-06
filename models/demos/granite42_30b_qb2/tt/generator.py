# SPDX-License-Identifier: Apache-2.0
"""Paged-cache generator with persistent split-sampling traces.

Host logits support plugin sampling and standalone reference checks.
Autoregressive feedback and position advance remain entirely on device.
"""

import time
from collections import Counter
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

import ttnn
from models.common.modules.sampling.sampling_1d import Sampling1D

from .model import MODEL_ID, REVISION, GraniteModel
from .token_history import append_tokens


@dataclass
class DecodeState:
    batch: int
    tokens: object
    position: object
    rope: object
    page_table: object
    logits: object
    seeds: object
    history: object = None
    history_cursor: object = None
    page_table_host: object = None
    model_trace: int | None = None
    sample_trace: int | None = None
    argmax_trace: int | None = None


class GraniteGenerator:
    def __init__(
        self,
        mesh_device,
        *,
        override_num_layers=None,
        max_seq_len=131072,
        cache_pages=None,
        kv_cache=None,
        batch_buckets=(1, 8, 16),
        clear_cache_on_request=False,
        precision_config=None,
        model=None,
        trace_prefill=False,
        **kwargs,
    ):
        self.mesh = mesh_device
        self.model = (
            model
            if model is not None
            else GraniteModel(mesh_device, override_num_layers=override_num_layers, precision_config=precision_config)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        if not 1 <= max_seq_len <= self.model.precision["supported_context"]:
            raise ValueError("Invalid configured context")
        self.max_seq_len = max_seq_len
        # Reserve the attention rounded read window, including physical final chunks.
        self.pages_per_row = ((max_seq_len + 1023) // 1024) * 32
        self.cache_pages = cache_pages or (int(kv_cache[0][0].shape[0]) if kv_cache is not None else self.pages_per_row)
        self.kv_cache = kv_cache if kv_cache is not None else self.model.allocate_cache(self.cache_pages)
        if (
            self.cache_pages < 32
            or len(self.kv_cache) != len(self.model.layers)
            or any(
                len(pair) != 2
                or any(
                    tuple(c.shape) != (self.cache_pages, 2, 32, 128) or c.dtype != self.model.cache_dtype for c in pair
                )
                for pair in self.kv_cache
            )
        ):
            raise ValueError(
                "KV cache must contain one policy-matching K/V pair per layer, [pages>=32,2,32,128] per rank"
            )
        self.owns_cache = kv_cache is None
        self.clear_cache_on_request = clear_cache_on_request
        self.buckets = tuple(batch_buckets)
        if any(b not in (1, 8, 16) for b in self.buckets):
            raise ValueError("Physical buckets are 1/8/16")
        self.counters = Counter(model_captures=0, sampling_captures=0, teardown_releases=0)
        self.prepared = False
        self.states = {}
        self.entries = {}
        self.prefill_tokens = {}
        self.prefill_valid = {}
        self.prefill_traces = {}
        self.trace_prefill = trace_prefill
        # Allocated before every capture; each prefill trace writes this buffer.
        # Consume/copy it before replaying another prefill bucket.
        self.prefill_output = (
            self.model.put(torch.zeros(1, 1, 1, self.model.config.vocab_size), shard=3, dtype=self.model.logits_dtype)
            if trace_prefill
            else None
        )
        self.last_index = self.put_int(torch.zeros(1, 1), unsigned=True)
        self.sampler = Sampling1D(
            self.model.config.vocab_size,
            self.mesh,
            max_batch_size=32,
            max_top_k=32,
            pad_to_power_of_2=True,
            num_gather_links=2,
            allow_force_argmax=True,
        )
        self.sampler.load_device_buffers()
        self.params = {
            b: dict(
                k=self.put_int(torch.ones(b), unsigned=True),
                p=self.model.put(torch.zeros(b), layout=ttnn.ROW_MAJOR_LAYOUT),
                temp=self.model.put(torch.ones(b), layout=ttnn.ROW_MAJOR_LAYOUT),
            )
            for b in self.buckets
        }
        for b in self.buckets:
            self.states[b] = DecodeState(
                b,
                self.put_int(torch.zeros(1, 1, 1, b), unsigned=True),
                self.put_int(torch.full((b,), -1)),
                self.put_int(torch.zeros(1, b), unsigned=True),
                self.put_int(torch.zeros(b, self.pages_per_row)),
                self.model.put(
                    torch.zeros(1, 1, b, self.model.config.vocab_size),
                    shard=3,
                    dtype=getattr(ttnn, self.model.precision["sampling_dtype"]),
                ),
                self.put_int(torch.arange(32)),
            )
        for b, state in self.states.items():
            state.history = self.put_int(torch.zeros(max_seq_len, 1, 1, b), unsigned=True)
            state.history_cursor = self.put_int(torch.zeros(1), unsigned=True)
        self.prefill_rows = [
            self.model.put(torch.zeros(1, 1, 1, self.model.config.vocab_size), shard=3, dtype=self.model.logits_dtype)
            for _ in range(max(self.buckets))
        ]
        table = torch.arange(self.pages_per_row, dtype=torch.int32).reshape(1, -1) % self.cache_pages
        for n in (128, 256, 512, 1024):
            self.entries[n] = self.model.layers[0]._prepare_entry(torch.zeros(1, n, 4096), table, 0, token=False)
            self.prefill_tokens[n] = self.put_int(torch.zeros(1, n), unsigned=True)
            self.prefill_valid[n] = self.put_int(torch.tensor([n]))

    def put_int(self, x, unsigned=False):
        return self.model.put(
            x.int(),
            dtype=getattr(ttnn, self.model.precision["token_dtype"]) if unsigned else ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def refresh(self, target, value):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=target.dtype,
            layout=target.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, target)

    def read_tokens(self, state):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(state.tokens)[0]).reshape(-1).long()

    def read_history(self, state, count):
        self.counters["history_readbacks"] += 1
        # One completion boundary; slicing happens after all replay commands finish.
        return ttnn.to_torch(ttnn.get_device_tensors(state.history)[0]).reshape(-1, state.batch)[:count].long()

    def read_logits(self, logits):
        self.counters["full_logits_readbacks"] += 1
        return torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(logits)], dim=-1)

    def _model_step(self, s):
        logits = self.model.decode_forward(
            s.tokens, current_pos=s.position, rope_indices=s.rope, page_table=s.page_table, kv_cache=self.kv_cache
        )
        ttnn.copy(logits, s.logits)

    def _prefill_step(self, n):
        logits = self.model.prefill_forward(
            self.prefill_tokens[n],
            entry=self.entries[n],
            kv_cache=self.kv_cache,
            last_index=self.last_index,
            valid_seq_len=self.prefill_valid[n],
        )
        ttnn.copy(logits, self.prefill_output)

    def _sample_step(self, s, argmax=False):
        self.sampler(
            s.logits,
            tt_out_tok=s.tokens,
            seeds=ttnn.typecast(s.seeds, ttnn.uint32),
            **({} if argmax else self.params[s.batch]),
        )
        # The next non-greedy draw must use new device RNG state without host reseeding.
        ttnn.plus_one(s.seeds, skip_negative_entries=True)
        append_tokens(s.tokens, s.history, s.history_cursor)

    def _assemble_prefill(self, state):
        joined = self.prefill_rows[0] if state.batch == 1 else ttnn.concat(self.prefill_rows[: state.batch], dim=2)
        ttnn.copy(joined, state.logits)

    def _active_sampling_rows(self, state, active):
        # Request-boundary seed masking preserves paused slots. Negative encoding
        # keeps their counters unchanged under traced plus_one(skip_negative_entries).
        seeds = ttnn.to_torch(ttnn.get_device_tensors(state.seeds)[0]).int()
        seeds = torch.where(seeds < 0, -seeds - 1, seeds)
        mask = torch.zeros(32, dtype=torch.bool)
        mask[: len(active)] = torch.as_tensor(active, dtype=torch.bool)
        self.refresh(state.seeds, torch.where(mask, seeds, -seeds - 1))
        self.counters["seed_boundary_readbacks"] += 1
        self.counters["seed_boundary_refreshes"] += 1

    def prepare(self):
        """Compile all configured prefill/decode/sampling variants before capture."""
        if self.prepared:
            return
        # Bulk warmup writes only physical pages 0..31. Preserve externally owned
        # cache contents as whole BFP8 tiles; no quantization or host readback.
        saved = []
        if not self.owns_cache:
            for pair in self.kv_cache:
                for cache in pair:
                    prefix = ttnn.slice(cache, [0, 0, 0, 0], [32, 2, 32, 128])
                    saved.append(ttnn.reshape(ttnn.permute(prefix, (1, 0, 2, 3)), [1, 2, 1024, 128]))
        for n, entry in self.entries.items():
            for last in (None, self.last_index):
                y = self.model.prefill_forward(
                    self.prefill_tokens[n],
                    entry=entry,
                    kv_cache=self.kv_cache,
                    last_index=last,
                    valid_seq_len=self.prefill_valid[n],
                )
                del y
            if self.trace_prefill:
                self._prefill_step(n)
        for b, s in self.states.items():
            self._assemble_prefill(s)
            # Inactive cache updates avoid overwriting a caller's real requests.
            self._model_step(s)
            for argmax in (False, True):
                self._sample_step(s, argmax)
        self._clear_owned_cache()
        if saved:
            for cache, backup in zip((c for pair in self.kv_cache for c in pair), saved):
                ttnn.experimental.paged_fill_cache(cache, backup, self.entries[1024].write_page_table, batch_idx=0)
            del backup, saved, prefix
        for s in self.states.values():
            self.refresh(s.tokens, torch.zeros(1, 1, 1, s.batch).int())
            self.refresh(s.position, torch.full((s.batch,), -1).int())
            self.refresh(s.rope, torch.zeros(1, s.batch).int())
        for state in self.states.values():
            self.refresh(state.seeds, torch.arange(32).int())
        ttnn.synchronize_device(self.mesh)
        self.counters["setup_synchronizations"] += 1
        for n in self.entries if self.trace_prefill else ():
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._prefill_step(n)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.prefill_traces[n] = tid
            self.counters["prefill_captures"] += 1
        for s in self.states.values():
            s.model_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._model_step(s)
            ttnn.end_trace_capture(self.mesh, s.model_trace, cq_id=0)
            self.counters["model_captures"] += 1
            for name, argmax in (("sample_trace", False), ("argmax_trace", True)):
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._sample_step(s, argmax)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                setattr(s, name, tid)
                self.counters["sampling_captures"] += 1
        self.prepared = True
        self.reset()

    def _clear_owned_cache(self):
        if self.owns_cache:
            for pair in self.kv_cache:
                for cache in pair:
                    ttnn.multiply(cache, 0.0, output_tensor=cache)

    def reset(self, *, clear_cache=True):
        if clear_cache:
            self._clear_owned_cache()
        for state in self.states.values():
            self.refresh(state.history_cursor, torch.zeros(1, dtype=torch.int32))
        for s in self.states.values():
            self.refresh(s.tokens, torch.zeros(1, 1, 1, s.batch).int())
            self.refresh(s.position, torch.full((s.batch,), -1).int())
            self.refresh(s.rope, torch.zeros(1, s.batch).int())
            self.refresh(s.page_table, torch.zeros(s.batch, self.pages_per_row).int())
            s.page_table_host = None
        for state in self.states.values():
            self.refresh(state.seeds, torch.arange(32).int())
        self.counters["request_resets"] += 1

    def set_sampling_params(
        self,
        *,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        enable_log_probs=False,
        reset_seed=True,
    ):
        if presence_penalty != 0 or frequency_penalty != 0 or repetition_penalty != 1:
            raise NotImplementedError("This sampler contract does not implement penalty history")
        if enable_log_probs:
            raise NotImplementedError("Common Sampling1D logprobs do not support TP4")

        def rows(value, dtype):
            x = torch.as_tensor(value, dtype=dtype).reshape(-1)
            if x.numel() == 1:
                x = x.expand(32).clone()
            if x.numel() != 32:
                raise ValueError("Parameters must be scalar or have 32 physical sampling rows")
            return x

        k, p, temp = rows(top_k, torch.int32), rows(top_p, torch.bfloat16), rows(temperature, torch.bfloat16)
        if (k < 1).any() or (k > 32).any() or (p < 0).any() or (p > 1).any() or (temp <= 0).any():
            raise ValueError("Requires 1<=top_k<=32, 0<=top_p<=1, temperature>0")
        for b, params in self.params.items():
            self.refresh(params["k"], k[:b])
            self.refresh(params["p"], p[:b])
            self.refresh(params["temp"], temp[:b].float().reciprocal().bfloat16())
        if reset_seed:
            seeds = rows(seed, torch.int64)
            for state in self.states.values():
                self.refresh(state.seeds, (seeds % 1000000 + 1).int())
        self.counters["sampling_param_refreshes"] += 1

    def reset_sampling_state(self, seeds, positions):
        """Reconstruct a serving request's RNG counter from authoritative positions."""
        state = self.states[self._bucket(len(positions))]
        values = torch.zeros(32, dtype=torch.int32)
        count = len(positions)
        values[:count] = (torch.as_tensor(seeds[:count]) % 1000000 + 1 + torch.as_tensor(positions).clamp_min(0)).int()
        self.refresh(state.seeds, values)
        self.counters["sampling_state_resets"] += 1

    def _validate_cache(self, cache):
        if (
            len(cache) != len(self.kv_cache)
            or any(len(p) != 2 for p in cache)
            or any(a is not b for pa, pb in zip(cache, self.kv_cache) for a, b in zip(pa, pb))
        ):
            raise ValueError("External cache must be bound at generator construction and retained for trace lifetime")

    def _page_table(self, s, table):
        table = table.int().contiguous()
        self._validate_pages(table)
        if table.shape != (s.batch, self.pages_per_row):
            raise ValueError("Page table must match configured physical bucket and column capacity")
        if s.page_table_host is None or not torch.equal(table, s.page_table_host):
            self.refresh(s.page_table, table)
            s.page_table_host = table.clone()
            self.counters["page_table_refreshes"] += 1

    def _validate_pages(self, table):
        if table.numel() and (int(table.min()) < 0 or int(table.max()) >= self.cache_pages):
            raise ValueError("Page table contains a physical page outside the bound KV cache")

    def _bucket(self, batch):
        return next(b for b in self.buckets if b >= batch)

    def _setup_decode(self, tokens, positions, table):
        b = self._bucket(len(positions))
        s = self.states[b]
        pos = torch.full((b,), -1, dtype=torch.int32)
        pos[: len(positions)] = torch.as_tensor(positions, dtype=torch.int32)
        ids = torch.zeros(1, 1, 1, b, dtype=torch.int32)
        ids.reshape(-1)[: tokens.numel()] = tokens.reshape(-1).int()
        self.refresh(s.tokens, ids)
        self.refresh(s.position, pos)
        self.refresh(s.rope, pos.clamp_min(0).reshape(1, b))
        self.counters["token_refreshes"] += 1
        self.counters["position_refreshes"] += 1
        self.counters["rope_refreshes"] += 1
        self._page_table(s, table)
        self._active_sampling_rows(s, pos >= 0)
        return s

    def replay(self, state, *, sampling_mode="device", strategy="split", read_from_device=True):
        ttnn.execute_trace(self.mesh, state.model_trace, cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        if sampling_mode == "logits":
            return state.logits
        if sampling_mode == "host":
            logits = self.read_logits(state.logits).reshape(state.batch, -1)
            return logits
        if sampling_mode != "device":
            raise ValueError("sampling_mode must be device or explicit host compatibility")
        tid = state.argmax_trace if strategy == "argmax" else state.sample_trace
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        self.counters["sampling_replays"] += 1
        return self.read_tokens(state) if read_from_device else state.tokens

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        sampling_mode="device",
        reload_inputs=True,
        reload_page_table=True,
        read_from_device=True,
        **kwargs,
    ):
        self._validate_cache(kv_cache)
        self.prepare()
        b = self._bucket(len(start_pos))
        s = self.states[b]
        if reload_inputs:
            s = self._setup_decode(tokens, start_pos, page_table)
        elif reload_page_table:
            self._page_table(s, page_table)
        return self.replay(s, sampling_mode=sampling_mode, read_from_device=read_from_device)

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        start_pos=None,
        sampling_mode="device",
        sample_mask=None,
        **kwargs,
    ):
        self._validate_cache(kv_cache)
        self.prepare()
        self._validate_pages(page_table)
        batch = len(prompt_lens)
        starts = [0] * batch if start_pos is None else list(start_pos)
        results = []
        for slot, (length, start) in enumerate(zip(prompt_lens, starts)):
            if length == 0:
                results.append(None)
                continue
            if not 0 <= start < self.max_seq_len or start + length > self.max_seq_len:
                raise ValueError("Prompt exceeds configured context")
            chunks = []
            offset = 0
            # Align to the smallest prepared bucket, then bridge with aligned buckets.
            # This bounds exact-token fringe work to 127 steps for any scheduler offset.
            while offset < length and (start + offset) % min(self.entries):
                table = page_table[slot : slot + 1]
                fringe_mode = "host" if return_all_logits or sampling_mode == "host" else "logits"
                value = self.decode_forward(
                    tokens[slot : slot + 1, offset : offset + 1],
                    torch.tensor([start + offset]),
                    page_table=table,
                    kv_cache=kv_cache,
                    sampling_mode=fringe_mode,
                )
                if fringe_mode == "logits":
                    if offset + 1 == length:
                        ttnn.copy(value, self.prefill_rows[slot])
                elif return_all_logits or offset + 1 == length:
                    chunks.append(value.reshape(1, 1, -1))
                offset += 1
            while offset < length:
                aligned = [n for n in self.entries if (start + offset) % n == 0]
                valid = min(max(aligned), length - offset)
                n = next(n for n in aligned if n >= valid)
                e = self.entries[n]
                # Request-boundary preparation is allowed on the host; no hidden-state arithmetic.
                e = self.model.layers[0].refresh_prefill_entry(
                    e, logical_length=valid, page_table=page_table, start_pos=start + offset, slot=slot
                )
                ids = torch.zeros(1, n, dtype=torch.int32)
                ids[0, :valid] = tokens[slot, offset : offset + valid].int()
                self.refresh(self.prefill_tokens[n], ids)
                self.refresh(self.last_index, torch.tensor([[valid - 1]], dtype=torch.int32))
                self.refresh(self.prefill_valid[n], torch.tensor([valid], dtype=torch.int32))
                last = None if return_all_logits else self.last_index
                if return_all_logits or not self.trace_prefill:
                    logits = self.model.prefill_forward(
                        self.prefill_tokens[n],
                        entry=e,
                        kv_cache=kv_cache,
                        last_index=last,
                        valid_seq_len=self.prefill_valid[n],
                    )
                else:
                    ttnn.execute_trace(self.mesh, self.prefill_traces[n], cq_id=0, blocking=False)
                    self.counters["prefill_replays"] += 1
                    logits = self.prefill_output
                if return_all_logits:
                    chunks.append(self.read_logits(logits).reshape(1, n, -1)[:, :valid])
                elif offset + valid == length:
                    if sampling_mode == "host":
                        chunks.append(self.read_logits(logits).reshape(1, 1, -1))
                    else:
                        ttnn.copy(logits, self.prefill_rows[slot])
                logits = None
                offset += valid
            results.append(torch.cat(chunks, dim=1) if return_all_logits else (chunks[-1] if chunks else None))
        if return_all_logits:
            output = torch.zeros(batch, max(prompt_lens), self.model.config.vocab_size)
            for i, r in enumerate(results):
                if r is not None:
                    output[i, : prompt_lens[i]] = r[0]
            return output
        if sampling_mode == "device":
            active = [length > 0 and (sample_mask is None or sample_mask[i]) for i, length in enumerate(prompt_lens)]
            if not any(active):
                return torch.zeros(batch, dtype=torch.long)
            state = self.states[self._bucket(batch)]
            self._assemble_prefill(state)
            self._active_sampling_rows(state, active)
            ttnn.execute_trace(self.mesh, state.sample_trace, cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
            return self.read_tokens(state)[:batch]
        empty = (
            torch.zeros(1, 1, self.model.config.vocab_size)
            if sampling_mode == "host"
            else torch.zeros(1, dtype=torch.long)
        )
        return torch.cat([r if r is not None else empty for r in results], dim=0)

    def _standalone_table(self):
        return (torch.arange(self.pages_per_row, dtype=torch.int32) % self.cache_pages).reshape(1, -1)

    def _check_standalone_capacity(self, length):
        if ((length + 1023) // 1024) * 32 > self.cache_pages:
            raise ValueError("Standalone request exceeds physical shared KV pool; configure cache_pages")

    def prefill_logits(self, prompt_token_ids):
        self._check_standalone_capacity(len(prompt_token_ids))
        self.prepare()
        self.reset(clear_cache=self.clear_cache_on_request)
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
        sampling_mode="device",
        stop_on_eos=False,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        enable_log_probs=False,
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("Production generation requires traced decode")
        if len(prompt_token_ids) + max(0, max_new_tokens - 1) > self.max_seq_len:
            raise ValueError("Prompt plus output exceeds configured context")
        if max_new_tokens <= 0:
            return []
        self._check_standalone_capacity(len(prompt_token_ids) + max_new_tokens - 1)
        if sampling_mode == "host" and (top_k != 1 or top_p != 0 or temperature != 1):
            raise ValueError("High-level host compatibility is greedy; use low-level logits for external sampling")
        self.prepare()
        begin = time.perf_counter()
        self.reset(clear_cache=self.clear_cache_on_request)
        self.set_sampling_params(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            enable_log_probs=enable_log_probs,
        )
        table = self._standalone_table()
        result = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            sampling_mode=sampling_mode,
        )
        first = int(result.reshape(-1)[0]) if sampling_mode == "device" else int(result.reshape(-1).argmax())
        self.last_ttft = time.perf_counter() - begin
        generated = [first]
        s = self.states[1]
        # Prefill sampling already wrote the persistent token tensor. Only teacher forcing
        # or the explicitly requested host-sampling mode replaces that token on the host.
        self.refresh(s.position, torch.tensor([len(prompt_token_ids)], dtype=torch.int32))
        self.refresh(s.rope, torch.tensor([[len(prompt_token_ids)]], dtype=torch.int32))
        self.counters["position_refreshes"] += 1
        self.counters["rope_refreshes"] += 1
        self._page_table(s, table)
        decode_start = time.perf_counter()
        if next_input is None and sampling_mode == "device" and not stop_on_eos:
            for _ in range(max_new_tokens - 1):
                self.replay(s, read_from_device=False)
            generated = self.read_history(s, max_new_tokens)[:, 0].tolist()
            self.last_decode_seconds = time.perf_counter() - decode_start
            return generated
        for i in range(max_new_tokens - 1):
            if stop_on_eos and generated[-1] == self.tokenizer.eos_token_id:
                break
            if next_input is not None or sampling_mode == "host":
                forced = next_input(i, generated[-1]) if next_input else generated[-1]
                self.refresh(s.tokens, torch.tensor([[[[forced]]]], dtype=torch.int32))
                self.counters["token_refreshes"] += 1
            result = self.replay(s, sampling_mode=sampling_mode)
            generated.append(
                int(result.reshape(-1)[0]) if sampling_mode == "device" else int(result.reshape(-1).argmax())
            )
        if next_input is not None:
            next_input(len(generated) - 1, generated[-1])
        self.last_decode_seconds = time.perf_counter() - decode_start
        return generated

    def generate_batch(
        self, prompts, max_new_tokens, *, enable_trace=True, top_k=1, top_p=0.0, temperature=1.0, seed=0
    ):
        """Fixed-slot mixed prompts; queues larger than 16 use successive batches."""
        if not enable_trace:
            raise ValueError("Batch generation requires traced decode")
        if not prompts:
            return []
        if len(prompts) > 16:
            return [
                out
                for start in range(0, len(prompts), 16)
                for out in self.generate_batch(
                    prompts[start : start + 16],
                    max_new_tokens,
                    enable_trace=enable_trace,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=seed,
                )
            ]
        if max_new_tokens <= 0:
            return [[] for _ in prompts]
        lens = [len(p) for p in prompts]
        if min(lens) < 1 or max(lens) + max_new_tokens - 1 > self.max_seq_len:
            raise ValueError("Invalid prompt/output length")
        n, b = len(prompts), self._bucket(len(prompts))
        needed = [((length + max_new_tokens - 1 + 1023) // 1024) * 32 for length in lens]
        if sum(needed) > self.cache_pages:
            raise ValueError(
                f"Physical shared KV pool needs {sum(needed)} pages, has {self.cache_pages}; configure cache_pages at construction"
            )
        table = torch.zeros(b, self.pages_per_row, dtype=torch.int32)
        offset = 0
        for row, count in enumerate(needed):
            table[row, :count] = torch.arange(offset, offset + count, dtype=torch.int32)
            table[row, count:] = offset + count - 1
            offset += count
        padded = torch.zeros(n, max(lens), dtype=torch.int64)
        for row, ids in enumerate(prompts):
            padded[row, : len(ids)] = torch.as_tensor(ids)
        self.prepare()
        begin = time.perf_counter()
        self.reset(clear_cache=self.clear_cache_on_request)
        self.set_sampling_params(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
        first = self.prefill_forward(padded, page_table=table, kv_cache=self.kv_cache, prompt_lens=lens)
        # Initial slot assembly is a request boundary; all subsequent feedback is
        # the sampler's direct write to these same persistent token buffers.
        s = self._setup_decode(first, torch.tensor(lens), table)
        self.last_batch_ttft = time.perf_counter() - begin
        outputs = [[int(t)] for t in first]
        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            self.replay(s, read_from_device=False)
        outputs = self.read_history(s, max_new_tokens)[:, :n].T.tolist()
        self.last_batch_decode_seconds = time.perf_counter() - decode_start
        return outputs

    def close(self):
        for tid in self.prefill_traces.values():
            ttnn.release_trace(self.mesh, tid)
            self.counters["teardown_releases"] += 1
        self.prefill_traces.clear()
        for s in self.states.values():
            for tid in (s.model_trace, s.sample_trace, s.argmax_trace):
                if tid is not None:
                    ttnn.release_trace(self.mesh, tid)
                    self.counters["teardown_releases"] += 1
            s.model_trace = s.sample_trace = s.argmax_trace = None

    teardown = close


def build_generator(model_dir, mesh_device, **kwargs):
    return GraniteGenerator(mesh_device, **kwargs)

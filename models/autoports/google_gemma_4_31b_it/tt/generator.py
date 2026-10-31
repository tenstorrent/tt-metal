# SPDX-License-Identifier: Apache-2.0
"""Metal readiness generator with persistent split model/sampling traces."""

import math
from numbers import Integral, Real
from types import SimpleNamespace

import torch
from readiness_check.contract import Generator
from transformers import AutoTokenizer, GenerationConfig

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.model import Gemma4Model
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION
from models.autoports.google_gemma_4_31b_it.tt.prefill import PreparedPrefill
from models.common.sampling.generator import SamplingParams, format_sampling_params
from models.common.sampling.tt_sampling import TTSampling


class Gemma4Generator(Generator):
    def __init__(
        self,
        mesh_device,
        *,
        layer_indices=None,
        batch_size=1,
        max_seq_len=None,
        kv_cache=None,
        page_table=None,
        host_page_tables=None,
        pool_tokens=None,
        output_capacity=None,
        head_mode="dram",
        precision_config=None,
    ):
        self.model = Gemma4Model(
            mesh_device, layer_indices=layer_indices, head_mode=head_mode, precision_config=precision_config
        )
        self.mesh = mesh_device
        self.batch = batch_size
        if max_seq_len is None:
            max_seq_len = self.model.precision_config["max_seq_len"]
        self.context = max_seq_len
        if max_seq_len > self.model.precision_config["max_seq_len"]:
            raise ValueError("Requested context exceeds the precision capability contract")
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
        generation = GenerationConfig.from_pretrained(HF_MODEL, revision=HF_REVISION)
        eos = generation.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])
        self.owns_cache = kv_cache is None
        self.pool_tokens = (
            min(batch_size * max_seq_len, max(max_seq_len, batch_size * 32768)) if pool_tokens is None else pool_tokens
        )
        if kv_cache is None:
            if page_table is not None or host_page_tables is not None:
                raise ValueError("Supply cache, device tables and host ownership maps together")
            self.kv_cache, self.page_table = self.model.allocate_cache(
                batch=batch_size, context=max_seq_len, pool_tokens=self.pool_tokens
            )
            host_page_tables = self.model.host_page_tables
        else:
            if page_table is None or host_page_tables is None:
                raise ValueError("External cache requires stable device tables and host ownership maps")
            self.kv_cache, self.page_table = kv_cache, page_table
        self._validate_page_maps(host_page_tables)
        self.page_revision = None
        self.seed_values = torch.arange(32, dtype=torch.int32)
        self.tokens = self.model.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.positions = self.model.upload(
            torch.zeros(1, batch_size, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.sampler = TTSampling(
            mesh_device,
            self.model.ccl,
            SimpleNamespace(
                vocab_size=self.model.config.vocab_size,
                max_batch_size=32,
                max_top_k=32,
                cluster_shape=tuple(mesh_device.shape),
            ),
        )
        if output_capacity is None:
            output_capacity = min(1024, max_seq_len)
        if not isinstance(output_capacity, int) or not 1 <= output_capacity <= max_seq_len:
            raise ValueError("Output capacity must be within configured context")
        self.output_capacity = output_capacity
        self.output_history = self.model.upload(
            torch.zeros(output_capacity, 1, 1, 32, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.output_index = self.model.upload(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.buffered_sample_trace = None
        self.model_trace = None
        self.sample_trace = None
        self.logits = self.model.upload(
            torch.zeros(1, 1, self.batch, self.model.config.vocab_size),
            dtype=getattr(ttnn, self.model.precision_config["sampling_dtype"]),
            shard=3,
        )
        self.inactive_logits = self.model.upload(
            torch.zeros(1, 1, 1, self.model.config.vocab_size),
            dtype=getattr(ttnn, self.model.precision_config["sampling_dtype"]),
            shard=3,
        )
        self.prefill = PreparedPrefill(self.model, self.context, host_page_tables)
        self.prepared = False
        self.counters = dict(
            model_replays=0,
            sample_replays=0,
            token_refreshes=0,
            position_refreshes=0,
            page_refreshes=0,
            readbacks=0,
            captures=0,
        )

    def _copy(self, value, target):
        host = ttnn.from_torch(
            value, dtype=target.dtype, layout=target.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, target)

    def _token_view(self):
        # Sampling requires rank4 storage; embedding consumes a rank2 logical view.
        return ttnn.reshape(ttnn.slice(self.tokens, (0, 0, 0, 0), (1, 1, 1, self.batch)), (1, self.batch))

    def _decode(self):
        logits = self.model.decode(
            self._token_view(), self.positions, kv_cache=self.kv_cache, page_tables=self.page_table
        )
        ttnn.plus_one(self.positions, skip_negative_entries=True)
        ttnn.copy(logits, self.logits)
        return self.logits

    def _sample(self, *, record=False):
        logits = self.logits
        # Common sampler parameter/index buffers own32 slots. Singleton rows
        # broadcast; intermediate logical batches must expose the padded rows.
        if 1 < self.batch < 32:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - self.batch), (0, 0)], value=0.0)
        result = self.sampler(logits, tt_out_tok=self.tokens)
        ttnn.plus_one(self.sampler.seeds_tt_tensor)
        if record:
            history = ttnn.indexed_fill(self.output_index, self.output_history, self.tokens)
            ttnn.copy(history, self.output_history)
            ttnn.plus_one(self.output_index)
        return result

    def prepare_decode(self):
        if self.model_trace is not None:
            return
        # Prepare every physical tail, terminal mode and cache reset before any
        # trace lives. Absolute offsets/page ownership are persistent tensor data.
        self.reset()
        for size in self.prefill.buckets:
            print("WARM_PREFILL", size, flush=True)
            self.prefill.stage(torch.zeros(size, dtype=torch.int32), slot=0, start=0, size=size, valid=size)
            hidden = self.prefill.hidden(size=size, start=0, kv_cache=self.kv_cache)
            all_logits = self.model.terminal(hidden)
            selected = self.model.terminal(self.prefill.last_row(hidden))
            rows = selected if self.batch == 1 else ttnn.concat([selected] * self.batch, dim=2)
            ttnn.copy(rows, self.logits)
            del hidden, all_logits, selected, rows
        self._decode()
        self._sample()
        self._sample(record=True)
        self.reset()
        self._copy(torch.zeros(1, 1, 1, 32, dtype=torch.int32), self.tokens)
        self._copy(self.seed_values, self.sampler.seeds_tt_tensor)
        ttnn.synchronize_device(self.mesh)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._decode()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.model_trace = tid
        sid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        sampled, logprobs = self._sample()
        ttnn.end_trace_capture(self.mesh, sid, cq_id=0)
        if logprobs is not None:
            raise RuntimeError("Unexpected logprob allocation in token-only trace")
        self.sample_trace = sid
        bid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._sample(record=True)
        ttnn.end_trace_capture(self.mesh, bid, cq_id=0)
        self.buffered_sample_trace = bid
        self.counters["captures"] += 3
        self.prepared = True
        self.reset()

    def _read_tokens(self):
        self.counters["readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).reshape(-1)[: self.batch].tolist()

    def _host_logits(self, logits):
        self.counters["readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()

    def prefill_forward(
        self, tokens, *, page_table, kv_cache, prompt_lens, return_all_logits=False, buffered=False, **kwargs
    ):
        if kv_cache is not self.kv_cache or page_table is not self.page_table:
            raise ValueError("External cache must be bound before preparation")
        if len(prompt_lens) != tokens.shape[0] or len(prompt_lens) != self.batch:
            raise ValueError("Each configured slot needs an explicit logical length")
        if any(not 0 <= length <= min(self.context, tokens.shape[1]) for length in prompt_lens):
            raise ValueError("Prompt length outside token or cache capacity")
        self.prepare_decode()
        results = []
        selected_rows = []
        for slot, length in enumerate(prompt_lens):
            if length == 0:
                if return_all_logits:
                    results.append(torch.zeros(1, 0, self.model.config.vocab_size))
                else:
                    selected_rows.append(self.inactive_logits)
                continue
            outputs = []
            for start in range(0, length, 1024):
                valid = min(1024, length - start)
                size = ((valid + 31) // 32) * 32
                self.prefill.stage(tokens[slot, start : start + valid], slot=slot, start=start, size=size, valid=valid)
                hidden = self.prefill.hidden(size=size, start=start, kv_cache=kv_cache)
                if return_all_logits:
                    logits = self.model.terminal(hidden)
                    outputs.append(self._host_logits(logits).reshape(1, size, -1)[:, :valid])
                    del logits
                elif start + valid == length:
                    selected_rows.append(self.model.terminal(self.prefill.last_row(hidden)))
                del hidden
            if return_all_logits:
                results.append(torch.cat(outputs, dim=1))
        if return_all_logits:
            padded = torch.zeros(self.batch, max(prompt_lens), self.model.config.vocab_size)
            for i, result in enumerate(results):
                padded[i, : prompt_lens[i]] = result[0]
            return padded
        rows = selected_rows[0] if self.batch == 1 else ttnn.concat(selected_rows, dim=2)
        ttnn.copy(rows, self.logits)
        del rows, selected_rows
        ttnn.execute_trace(
            self.mesh, self.buffered_sample_trace if buffered else self.sample_trace, cq_id=0, blocking=False
        )
        self.counters["sample_replays"] += 1
        return self.tokens if buffered else self._read_tokens()

    def set_sampling(self, *, temperature=0.0, top_k=1, top_p=1.0, seed=0):
        def per_slot(value, name):
            values = list(value) if isinstance(value, (list, tuple)) else [value] * self.batch
            if len(values) != self.batch:
                raise ValueError(f"{name} requires a scalar or one value per configured slot")
            return values

        temperatures = per_slot(temperature, "temperature")
        ks, ps = per_slot(top_k, "top_k"), per_slot(top_p, "top_p")
        for temp, k, p in zip(temperatures, ks, ps):
            if not isinstance(temp, Real) or not math.isfinite(temp) or temp < 0:
                raise ValueError("Temperature must be finite and nonnegative")
            if not isinstance(p, Real) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("top_p must be within [0, 1]")
            if not isinstance(k, Integral) or (temp > 0 and not 1 <= k <= 32):
                raise ValueError("Common device sampling supports stochastic top_k in [1, 32]")
        if not isinstance(seed, Integral) or not 0 <= seed <= 2**31 - 33:
            raise ValueError("Seed must leave room for the 32 signed per-slot seeds")
        # The common formatter treats scalar temperature as one active lane.
        # Describe all configured slots explicitly, then let it pad unused lanes.
        params = format_sampling_params(SamplingParams(temperature=temperatures, top_k=ks, top_p=ps), 32)
        self.sampler.reset_params(params.top_k, params.top_p, params.temperature)
        self.seed_values = torch.arange(32, dtype=torch.int32) + int(seed)
        self._copy(self.seed_values, self.sampler.seeds_tt_tensor)

    def _reserve_requests(self, lengths, generated):
        if not self.owns_cache:
            return
        counts = [((n + generated + 127) // 128) * 4 if n else 0 for n in lengths]
        if sum(counts) > self.model.pool_blocks:
            raise ValueError(f"Requests need {sum(counts)*32} pooled tokens; capacity is {self.model.pool_blocks*32}")
        tables = []
        for layer, old in zip(self.model.layers, self.prefill.host_pages):
            if layer.sliding_window:
                tables.append(old)
                continue
            table = torch.empty_like(old)
            offset = 0
            for slot, count in enumerate(counts):
                table[slot] = offset if count else 0
                table[slot, :count] = torch.arange(offset, offset + count, dtype=torch.int32)
                offset += count
            tables.append(table)
        revision = tuple(counts)
        self.refresh_page_tables(tables, revision=revision)

    def _validate_page_maps(self, maps):
        if len(self.kv_cache) != len(self.model.layers) or len(self.page_table) != len(self.model.layers):
            raise ValueError("Cache and device tables must cover every decoder layer")
        if len(maps) != len(self.model.layers):
            raise ValueError("One page ownership map is required per layer")
        by_kind = {}
        for layer, host, cache, page in zip(self.model.layers, maps, self.kv_cache, self.page_table):
            if len(cache) != 2 or any(
                tuple(t.shape)[1:] != (layer.nkv, 32, layer.head_dim) or t.dtype != layer.cache_dtype for t in cache
            ):
                raise ValueError("External cache must preserve the decoder paged shape and dtype")
            if tuple(cache[0].shape) != tuple(cache[1].shape) or tuple(page.shape) != tuple(host.shape):
                raise ValueError("External K/V capacities and ownership-map shapes must agree")
            if page.dtype != ttnn.int32 or page.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise ValueError("Page tables require signed int32 row-major storage")
            if host.ndim != 2 or host.shape[0] != self.batch:
                raise ValueError("Page ownership maps must contain every configured slot")
            if bool((host < 0).any()) or bool((host >= cache[0].shape[0]).any()):
                raise ValueError("Page ownership map references a block outside its cache")
            previous = by_kind.setdefault(layer.layer_type, host)
            if not torch.equal(previous, host):
                raise ValueError("Prepared prefill requires common page ownership within each attention kind")

    def refresh_page_tables(self, host_page_tables, *, revision):
        """Refresh fixed bindings only on an explicit scheduler mapping change."""
        if revision == self.page_revision:
            return
        self._validate_page_maps(host_page_tables)
        for host, target in zip(host_page_tables, self.page_table):
            if tuple(host.shape) != tuple(target.shape):
                raise ValueError("Page-table capacity is fixed for the trace lifetime")
        for host, target in zip(host_page_tables, self.page_table):
            self._copy(host.to(torch.int32), target)
            self.counters["page_refreshes"] += 1
        self.prefill.host_pages = [p.clone() for p in host_page_tables]
        self.page_revision = revision

    def decode_forward(self, tokens, start_pos, *, page_table, kv_cache, **kwargs):
        if kv_cache is not self.kv_cache or page_table is not self.page_table:
            raise ValueError("External cache binding must be configured before capture")
        token_values = torch.as_tensor(tokens).reshape(-1)
        positions = torch.as_tensor(start_pos).reshape(-1)
        if token_values.numel() != self.batch or positions.numel() != self.batch:
            raise ValueError("Decode requires one token and position per configured slot")
        if any(
            value.is_floating_point() or value.is_complex() or value.dtype == torch.bool
            for value in (token_values, positions)
        ):
            raise ValueError("Decode tokens and positions must be integral")
        if bool(((positions < -1) | (positions >= self.context)).any()):
            raise ValueError("Decode position must be -1 or within the supported context")
        if bool(((token_values < 0) | (token_values >= self.model.config.vocab_size)).any()):
            raise ValueError("Decode token outside pinned vocabulary")
        self.prepare_decode()
        padded = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        padded.reshape(-1)[: self.batch] = token_values
        self._copy(padded, self.tokens)
        self._copy(positions.to(torch.int32).reshape(1, self.batch), self.positions)
        self.counters["token_refreshes"] += 1
        self.counters["position_refreshes"] += 1
        return self._replay()

    def _replay(self, *, host_sampling=False):
        ttnn.execute_trace(self.mesh, self.model_trace, cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        if host_sampling:
            return self._host_logits(self.logits).reshape(self.batch, -1)
        ttnn.execute_trace(self.mesh, self.sample_trace, cq_id=0, blocking=False)
        self.counters["sample_replays"] += 1
        return self._read_tokens()

    def prefill_logits(self, prompt_token_ids):
        self.prepare_decode()
        self.reset()
        self._reserve_requests([len(prompt_token_ids)], 0)
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table,
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
        ignore_eos=False,
        buffered=True,
        sampling_params=None,
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("Use the reduced probe for eager debugging")
        if host_sampling and sampling_params:
            host_params = {}
            for name, default in (("temperature", 0), ("top_k", 1)):
                value = sampling_params.get(name, default)
                if isinstance(value, (list, tuple)):
                    if len(value) != 1:
                        raise ValueError("Single-request host sampling requires one parameter per field")
                    value = value[0]
                host_params[name] = value
            if host_params["temperature"] > 0 and host_params["top_k"] != 1:
                raise ValueError("Host compatibility mode is greedy; stochastic sampling uses the device trace")
        if self.batch != 1:
            raise ValueError("Single-list readiness API requires batch1; low-level API supports batches")
        if len(prompt_token_ids) + max_new_tokens > self.context:
            raise ValueError("Requested generation exceeds cache capacity")
        if max_new_tokens <= 0:
            return []
        if (
            buffered
            and ignore_eos
            and next_input is None
            and not host_sampling
            and max_new_tokens <= self.output_capacity
        ):
            return self.generate_buffered([prompt_token_ids], max_new_tokens, sampling_params=sampling_params)[0]
        self.prepare_decode()
        self.set_sampling(**(sampling_params or {}))
        self.reset()
        self._reserve_requests([len(prompt_token_ids)], max_new_tokens)
        result = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
        )
        predicted = (
            int(self._host_logits(self.logits).reshape(self.batch, -1)[0].argmax()) if host_sampling else int(result[0])
        )
        output = [predicted]
        if next_input is None and not ignore_eos and predicted in self.eos_token_ids:
            return output
        padded = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        if next_input is not None or host_sampling:
            padded[0, 0, 0, 0] = predicted if next_input is None else next_input(0, predicted)
            self._copy(padded, self.tokens)
            self.counters["token_refreshes"] += 1
        self._copy(torch.tensor([[len(prompt_token_ids)]], dtype=torch.int32), self.positions)
        self.counters["position_refreshes"] += 1
        for step in range(1, max_new_tokens):
            result = self._replay(host_sampling=host_sampling)
            predicted = int(result[0].argmax()) if host_sampling else int(result[0])
            output.append(predicted)
            if next_input is None and not ignore_eos and predicted in self.eos_token_ids:
                break
            if next_input is not None or host_sampling:
                forced = predicted if next_input is None else next_input(step, predicted)
                padded[0, 0, 0, 0] = forced
                self._copy(padded, self.tokens)
                self.counters["token_refreshes"] += 1
        return output

    def generate_batch(
        self, prompts, max_new_tokens, *, enable_trace=True, sampling_params=None, ignore_eos=False, buffered=True
    ):
        """Fixed configured slots, with independent prompt lengths and positions."""
        if not enable_trace:
            raise ValueError("Batched generation requires traced token feedback")
        if len(prompts) != self.batch:
            raise ValueError("Supply one token list or None per configured slot")
        lengths = [len(p) if p else 0 for p in prompts]
        if any(n + max_new_tokens > self.context for n in lengths):
            raise ValueError("Generation exceeds configured cache capacity")
        if max_new_tokens <= 0 or not any(lengths):
            return [[] for _ in prompts]
        if buffered and ignore_eos and max_new_tokens <= self.output_capacity:
            return self.generate_buffered(prompts, max_new_tokens, sampling_params=sampling_params)
        self.prepare_decode()
        self.set_sampling(**(sampling_params or {}))
        self.reset()
        self._reserve_requests(lengths, max_new_tokens)
        tokens = torch.zeros(self.batch, max(lengths), dtype=torch.int32)
        for slot, prompt in enumerate(prompts):
            if prompt:
                tokens[slot, : len(prompt)] = torch.tensor(prompt)
        first = self.prefill_forward(tokens, page_table=self.page_table, kv_cache=self.kv_cache, prompt_lens=lengths)
        outputs = [[int(token)] if length else [] for token, length in zip(first, lengths)]
        finished = [
            not length or (not ignore_eos and int(token) in self.eos_token_ids) for token, length in zip(first, lengths)
        ]
        self._copy(torch.tensor([[n if n else -1 for n in lengths]], dtype=torch.int32), self.positions)
        self.counters["position_refreshes"] += 1
        for _ in range(1, max_new_tokens):
            if all(finished):
                break
            for slot, (output, token) in enumerate(zip(outputs, self._replay())):
                if not finished[slot]:
                    output.append(int(token))
                    finished[slot] = not ignore_eos and int(token) in self.eos_token_ids
        return outputs

    def generate_buffered(self, prompts, max_new_tokens, *, sampling_params=None):
        """Fixed-step generation with one output read, returning all tokens per slot.

        EOS is deliberately not a device-loop stop condition. Callers needing
        immediate EOS stopping use generate/generate_batch. Output capacity is
        chosen at construction, before traces, independently of context capacity.
        """
        if len(prompts) != self.batch:
            raise ValueError("Supply one token list or None per configured slot")
        lengths = [len(p) if p else 0 for p in prompts]
        if not 0 <= max_new_tokens <= self.output_capacity:
            raise ValueError("Generation exceeds prepared output capacity")
        if any(n + max_new_tokens > self.context for n in lengths):
            raise ValueError("Generation exceeds configured cache capacity")
        if not max_new_tokens or not any(lengths):
            return [[] for _ in prompts]
        self.prepare_decode()
        self.set_sampling(**(sampling_params or {}))
        self.reset()
        self._reserve_requests(lengths, max_new_tokens)
        tokens = torch.zeros(self.batch, max(lengths), dtype=torch.int32)
        for slot, prompt in enumerate(prompts):
            if prompt:
                tokens[slot, : len(prompt)] = torch.tensor(prompt)
        self.prefill_forward(
            tokens, page_table=self.page_table, kv_cache=self.kv_cache, prompt_lens=lengths, buffered=True
        )
        self._copy(torch.tensor([[n if n else -1 for n in lengths]], dtype=torch.int32), self.positions)
        self.counters["position_refreshes"] += 1
        for _ in range(1, max_new_tokens):
            ttnn.execute_trace(self.mesh, self.model_trace, cq_id=0, blocking=False)
            ttnn.execute_trace(self.mesh, self.buffered_sample_trace, cq_id=0, blocking=False)
            self.counters["model_replays"] += 1
            self.counters["sample_replays"] += 1
        self.counters["readbacks"] += 1
        history = ttnn.to_torch(ttnn.get_device_tensors(self.output_history)[0]).reshape(self.output_capacity, 32)
        return [history[:max_new_tokens, slot].tolist() if length else [] for slot, length in enumerate(lengths)]

    def reset(self):
        self._copy(torch.zeros(1, dtype=torch.int32), self.output_index)
        for cache in self.kv_cache:
            for tensor in cache:
                ttnn.mul(tensor, 0, output_tensor=tensor)
        self._copy(torch.zeros(1, self.batch, dtype=torch.int32), self.positions)
        self._copy(self.seed_values, self.sampler.seeds_tt_tensor)


def build_generator(model_dir, mesh_device, **kwargs):
    return Gemma4Generator(mesh_device, **kwargs)

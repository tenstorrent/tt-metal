# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Explicit-cache generator for Gemma4 TP4.

The optimized loop replays two traces. Only the sampling trace writes the next
token; position advances in the model trace. Host sampling is opt-in compatibility.
"""

import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from loguru import logger
from transformers import AutoTokenizer, GenerationConfig
from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

import ttnn
from models.common.sampling.tt_sampling import TTSampling
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.model import Gemma4Model


@dataclass
class CacheState:
    kv: list
    page_tables: dict
    batch: int
    capacity: int
    device_tables: dict = field(default_factory=dict)
    uploaded_tables: dict = field(default_factory=dict)
    # One explicitly reserved warmup page per fixed slot, independent per kind.
    scratch_pages: dict = field(default_factory=dict)
    # Serving supplies per-layer allocator tables (including null/evicted pages).
    # Standalone mode retains the stricter private cyclic-cache contract.
    vllm_owned: bool = False
    prefill_history: dict = field(default_factory=dict)


class Gemma4Generator:
    def __init__(self, model, *, max_seq_len=2048):
        self.model, self.mesh = model, model.mesh
        self.tokenizer = AutoTokenizer.from_pretrained(model.checkpoint, local_files_only=True)
        generation_config = GenerationConfig.from_pretrained(model.checkpoint, local_files_only=True)
        eos = generation_config.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])
        self.max_seq_len = model.config.max_position_embeddings
        self.initial_capacity = max_seq_len
        self.states = {}
        self.traces = {}
        self.prefill_shapes = set()
        self.counters = {
            k: 0
            for k in (
                "model_replays",
                "sampling_replays",
                "token_refreshes",
                "position_refreshes",
                "page_table_refreshes",
                "synchronizations",
                "token_readbacks",
                "logits_readbacks",
                "seed_refreshes",
                "rope_refreshes",
                "sampling_param_refreshes",
                "prefill_table_uploads",
            )
        }
        args = SimpleNamespace(
            vocab_size=model.config.vocab_size,
            padded_vocab_size=model.config.vocab_size,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=self.mesh.shape,
            model_config={
                "SAMPLING_AG_CONFIG": {"allow_force_argmax": False, "num_links": 2, "topology": ttnn.Topology.Ring}
            },
        )
        self.sampler = TTSampling(mesh_device=self.mesh, tt_ccl=model.ccl, args=args)

    def allocate_cache(self, *, batch=1, capacity=None):
        capacity = self.initial_capacity if capacity is None else capacity
        if not 1 <= batch <= 32 or not 1 <= capacity <= self.max_seq_len:
            raise ValueError("Expected batch1..32 and context1..262144")
        self._retire_traces()
        # All accepted decode SDPA chunks are64/128: one128-token page covers
        # the rounded read window. No power-of-two capacity restriction.
        pages = (capacity + 127) // 128
        tables = {}
        pools = {}
        for kind in {l.kind for l in self.model.layers}:
            pool = min(pages, Decoder.SLIDING_WINDOW_PAGES) if kind == "sliding_attention" else pages
            pools[kind] = pool
            tables[kind] = (torch.arange(batch)[:, None] * pool + torch.arange(pages)[None, :] % pool).int()
        # Private per-slot scratch pages let warmup use the real cache geometry
        # without writing any caller-owned prompt page.
        kv = [l.allocate_cache(physical_pages=batch * pools[l.kind] + max(2, batch)) for l in self.model.layers]
        # Reset must preserve captured traces. Compile/cache its exact in-place
        # operations before capture, while these newly owned caches are empty.
        for pair in kv:
            for tensor in pair:
                ttnn.multiply(tensor, 0.0, output_tensor=tensor)
        scratch = {
            kind: list(range(batch * pools[kind] + max(2, batch) - batch, batch * pools[kind] + max(2, batch)))
            for kind in pools
        }
        return CacheState(kv, tables, batch, pages * 128, scratch_pages=scratch)

    def _owned(self, batch, capacity):
        for (b, c), state in self.states.items():
            if b == batch and c >= capacity:
                return state
        # A larger request replaces this batch's obsolete owned cache. Keeping
        # every historical capacity would invalidate the single-state DRAM bound.
        self._retire_traces()
        for key in [key for key in self.states if key[0] == batch]:
            del self.states[key]
        # Do not retain the loop variable's last obsolete state through allocation.
        state = None
        state = self.allocate_cache(batch=batch, capacity=max(self.initial_capacity, capacity))
        self.states[(batch, state.capacity)] = state
        return state

    def _upload(self, x, dtype=ttnn.int32):
        return self.model.tensor(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _refresh(self, x, dest, counter):
        h = ttnn.from_torch(x.contiguous(), dtype=dest.dtype, layout=dest.layout)
        ttnn.copy_host_to_device_tensor(h, dest)
        self.counters[counter] += 1

    def _tables(self, state, tables):
        self._validate_cache(state, tables)
        if not isinstance(tables, dict):
            raise TypeError("page_table must map layer kind to its independent physical-page table")
        expected = set(range(len(self.model.layers))) if state.vllm_owned else {l.kind for l in self.model.layers}
        if set(tables) != expected:
            raise ValueError("Page tables must cover each loaded layer kind exactly")
        for kind, table in tables.items():
            table = table.int().contiguous()
            if tuple(table.shape) != (state.batch, state.capacity // 128):
                raise ValueError("Page table shape disagrees with cache batch/capacity")
            if kind not in state.device_tables:
                state.device_tables[kind] = self._upload(table)
                state.uploaded_tables[kind] = table.clone()
                self.counters["page_table_refreshes"] += 1
            elif not torch.equal(table, state.uploaded_tables[kind]):
                if tuple(table.shape) != tuple(state.uploaded_tables[kind].shape):
                    raise ValueError("Changing table capacity requires a new CacheState")
                self._refresh(table, state.device_tables[kind], "page_table_refreshes")
                state.uploaded_tables[kind] = table.clone()
        return state.device_tables

    def _validate_cache(self, state, tables):
        if not isinstance(state, CacheState):
            raise TypeError("kv_cache must explicitly describe its CacheState ownership")
        if not 1 <= state.batch <= 32 or not 0 < state.capacity <= self.max_seq_len or state.capacity % 128:
            raise ValueError("Invalid cache batch/capacity")
        if len(state.kv) != len(self.model.layers):
            raise ValueError("Cache must contain exactly one K/V pair per loaded layer")
        if state.vllm_owned:
            self._validate_serving_cache(state, tables)
            return
        kinds = {l.kind for l in self.model.layers}
        if not isinstance(tables, dict) or set(tables) != kinds or set(state.scratch_pages) != kinds:
            raise ValueError("Explicit tables and private scratch pages are required for every layer kind")
        for kind, table in tables.items():
            if not isinstance(table, torch.Tensor) or tuple(table.shape) != (state.batch, state.capacity // 128):
                raise ValueError("Expected a two-dimensional page table")
            period = (
                min(table.shape[1], Decoder.SLIDING_WINDOW_PAGES) if kind == "sliding_attention" else table.shape[1]
            )
            unique_pages = table[:, :period]
            if unique_pages.numel() != torch.unique(unique_pages).numel():
                raise ValueError("Writable cache pages must be distinct across slots and live logical blocks")
            if kind == "sliding_attention" and not torch.equal(
                table, unique_pages[:, torch.arange(table.shape[1]) % period]
            ):
                raise ValueError("Sliding cache requires the accepted private nine-page cyclic mapping")
        buffer_owners = set()
        for layer, pair in zip(self.model.layers, state.kv):
            if len(pair) != 2:
                raise ValueError("Each layer requires exactly two K/V tensors")
            for tensor in pair:
                if (
                    len(tensor.shape) != 4
                    or tuple(tensor.shape)[1:] != (layer.kv_heads, 128, layer.head_dim)
                    or tensor.dtype != ttnn.bfloat8_b
                    or tensor.layout != ttnn.TILE_LAYOUT
                    or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                    or tensor.device() != self.mesh
                ):
                    raise ValueError("Cache geometry, dtype, layout or mesh violates the decoder contract")
                owner = tensor.buffer_unique_id()
                if owner in buffer_owners:
                    raise ValueError("Every writable K/V tensor requires distinct backing storage across all layers")
                buffer_owners.add(owner)
            if tuple(pair[0].shape) != tuple(pair[1].shape):
                raise ValueError("K/V physical page counts must match")
            table = tables[layer.kind]
            if (
                not isinstance(table, torch.Tensor)
                or table.dtype not in (torch.int32, torch.int64)
                or tuple(table.shape) != (state.batch, state.capacity // 128)
            ):
                raise ValueError("Expected integral [batch,logical_pages] page table")
            scratch = state.scratch_pages[layer.kind]
            if len(scratch) != state.batch or len(set(scratch)) != state.batch:
                raise ValueError("Exactly one distinct private warmup page is required per slot")
            physical = pair[0].shape[0]
            if any(not isinstance(p, int) or p < 0 or p >= physical for p in scratch):
                raise ValueError("Invalid private warmup page")
            if bool(((table < 0) | (table >= physical)).any()) or any(bool((table == p).any()) for p in scratch):
                raise ValueError("Page table references an invalid or reserved warmup page")

    def _validate_serving_cache(self, state, tables):
        expected = set(range(len(self.model.layers)))
        if set(tables) != expected or set(state.scratch_pages) != expected:
            raise ValueError("Serving needs one allocator table and reserved warmup map per layer")
        checked_tables = set()
        for i, (layer, pair) in enumerate(zip(self.model.layers, state.kv)):
            table = tables[i]
            if tuple(table.shape) != (state.batch, state.capacity // 128) or table.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("Invalid serving table geometry")
            if len(pair) != 2 or pair[0].buffer_unique_id() == pair[1].buffer_unique_id():
                raise ValueError("Serving K and V must have separate storage")
            for tensor in pair:
                if (
                    tuple(tensor.shape)[1:] != (layer.kv_heads, 128, layer.head_dim)
                    or tensor.dtype != ttnn.bfloat8_b
                    or tensor.layout != ttnn.TILE_LAYOUT
                    or tensor.device() != self.mesh
                    or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                ):
                    raise ValueError("Serving cache violates selected native geometry/precision")
            physical = pair[0].shape[0]
            scratch = state.scratch_pages[i]
            if len(scratch) != state.batch or len(set(scratch)) != state.batch:
                raise ValueError("Serving warmup needs distinct private pages")
            # Keep per-layer cache geometry/ownership checks above. Shared HMA
            # tables with identical physical bounds need one content scan per
            # call, not one scan per layer and per reserved batch slot.
            key = (id(table), physical, tuple(scratch))
            if key not in checked_tables:
                low, high = torch.aminmax(table)
                if scratch == list(range(physical - state.batch, physical)):
                    invalid = low < 0 or high >= physical - state.batch
                else:
                    invalid = (
                        low < 0 or high >= physical or torch.isin(table, torch.tensor(scratch, dtype=table.dtype)).any()
                    )
                if bool(invalid):
                    raise ValueError("Serving table references invalid or private warmup storage")
                checked_tables.add(key)

    def configure_sampling(
        self, *, top_k=1, top_p=0.0, temperature=1.0, seed=0, batch=1, do_sample=True, reset_seed=True
    ):
        """Request-boundary update of stable per-slot common-sampler buffers.

        The sampling kernel supports k1..32. k0/-1/unbounded and checkpoint k64
        are explicitly unsupported, rather than silently clamped to another distribution.
        Inactive rows' sampled tokens/seeds are ignored; activation configures their state.
        """
        if not 1 <= batch <= 32:
            raise ValueError("Expected batch1..32")

        def rows(value):
            values = list(value) if isinstance(value, (list, tuple)) else [value] * batch
            if len(values) != batch:
                raise ValueError("Sampling parameters need one value per fixed slot")
            return values

        ks, ps, ts, seeds = map(rows, (top_k, top_p, temperature, seed))
        for i in range(batch):
            if not math.isfinite(ts[i]) or ts[i] < 0 or not math.isfinite(ps[i]) or not 0 <= ps[i] <= 1:
                raise ValueError("Temperature must be finite/nonnegative and top_p in [0,1]")
            if not do_sample or ts[i] == 0:
                ks[i], ps[i], ts[i] = 1, 0.0, 1.0
            elif not isinstance(ks[i], int) or not 1 <= ks[i] <= 32:
                raise ValueError("Canonical on-device sampling supports top_k1..32; no silent clamp")
            else:
                ts[i] = 1.0 / ts[i]
            if not isinstance(seeds[i], int) or not 0 <= seeds[i] < 2**31:
                raise ValueError("Seed must be an integer in [0,2**31)")
        self.sampler.reset_params(
            k=ks + [1] * (32 - batch), p=ps + [0.0] * (32 - batch), temp=ts + [1.0] * (32 - batch)
        )
        self.counters["sampling_param_refreshes"] += 1
        if reset_seed:
            self._refresh(
                torch.tensor(seeds + [0] * (32 - batch), dtype=torch.int32),
                self.sampler.seeds_tt_tensor,
                "seed_refreshes",
            )

    @staticmethod
    def _check_options(kwargs):
        defaults = {
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "logprobs": None,
            "num_logprobs": None,
            "logits_processor": None,
            "enable_log_probs": False,
        }
        for name, default in defaults.items():
            if name in kwargs and kwargs[name] != default:
                raise ValueError(f"Unsupported generation option: {name}")

    def _validate_tokens(self, tokens):
        if tokens.dtype not in (torch.int32, torch.int64) or bool(
            ((tokens < 0) | (tokens >= self.model.config.vocab_size)).any()
        ):
            raise ValueError("Expected integer token IDs in the model vocabulary")

    def _host_logits(self, logits):
        self.counters["logits_readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3)).float()

    def _read_tokens(self, token, batch):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(token)[0]).reshape(-1)[:batch].long()

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        slots=None,
        return_device=False,
        start_positions=None,
        history_keys=None,
        **kwargs,
    ):
        state = kv_cache
        self._check_options(kwargs)
        if not isinstance(state, CacheState):
            raise TypeError("kv_cache must be a CacheState; caller may supply its kv tensors and tables")
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        if (
            len(slots) != len(prompt_lens)
            or len(set(slots)) != len(slots)
            or any(s < 0 or s >= state.batch for s in slots)
        ):
            raise ValueError("Expected distinct valid fixed slots, one per prompt")
        if tokens.ndim != 2 or tokens.shape[0] != len(prompt_lens) or not prompt_lens:
            raise ValueError("Expected one token row per nonempty prompt")
        self._validate_tokens(tokens)
        if any(not isinstance(n, int) or not 0 < n <= min(state.capacity, tokens.shape[1]) for n in prompt_lens):
            raise ValueError("Invalid logical prompt length")
        starts = [0] * len(prompt_lens) if start_positions is None else list(start_positions)
        if len(starts) != len(prompt_lens) or any(s < 0 or s >= n for s, n in zip(starts, prompt_lens)):
            raise ValueError("Prefill must advance each request from its current position")
        shape_key = (state.batch, state.capacity, tuple(prompt_lens), tuple(starts), bool(return_all_logits))
        if shape_key not in self.prefill_shapes:
            # A new prefill signature may create cached protocol semaphores or
            # norm buffers. Retire old traces before those persistent allocations.
            self._retire_traces()
        self._tables(state, page_table)
        outputs = []
        for row, (slot, length) in enumerate(zip(slots, prompt_lens)):
            if not 0 < length <= state.capacity or length > tokens.shape[1]:
                raise ValueError("Invalid logical prompt length")
            start = starts[row]
            aligned_start = start // Decoder.PAGE_SIZE * Decoder.PAGE_SIZE
            chunk_length = length - aligned_start
            histories = None
            if history_keys is not None:
                key = history_keys[row]
                if start == 0:
                    histories = [None] * len(self.model.layers)
                else:
                    previous = state.prefill_history[key]
                    if previous["end"] != start:
                        raise ValueError("Prefill history does not match the scheduler continuation")
                    rewind = start - aligned_start
                    histories = []
                    for tail in previous["layers"]:
                        if tail is None:
                            histories.append(None)
                            continue
                        valid = tail[0].shape[2] - rewind
                        if valid == 0:
                            histories.append(None)
                            continue
                        begin = max(0, valid - 1024)
                        histories.append(tuple(t[:, :, begin:valid, :] for t in tail))
            ids = torch.zeros(1, (chunk_length + 31) // 32 * 32, dtype=torch.int32)
            ids[0, :chunk_length] = tokens[row, aligned_start:length]
            tt_ids = self._upload(ids, ttnn.uint32)
            tables, uploaded_rows = {}, {}
            for kind, table in page_table.items():
                sentinel = state.vllm_owned and self.model.layers[kind].kind == "sliding_attention"
                key = (id(table), sentinel)
                if key not in uploaded_rows:
                    row_table = table[slot : slot + 1]
                    if sentinel:
                        # Only temporary serving prefill uses the null-page
                        # sentinel. These read-only rows may be shared; mutable
                        # persistent decode tables retain independent storage.
                        row_table = row_table.masked_fill(row_table == 0, -1)
                    uploaded_rows[key] = self._upload(row_table)
                    self.counters["prefill_table_uploads"] += 1
                tables[kind] = uploaded_rows[key]
            out = self.model.prefill_device(
                tt_ids,
                sequence_length=chunk_length,
                page_tables=tables,
                kv_cache=state.kv,
                all_logits=return_all_logits,
                start_pos=aligned_start,
                histories=histories,
            )
            if history_keys is not None:
                state.prefill_history[history_keys[row]] = {"end": length, "layers": histories}
            outputs.append(
                out if return_device else self._host_logits(out).reshape(1, -1, self.model.config.vocab_size)
            )
            # Device-return outputs retain their own owners in outputs. Host
            # mode has consumed out; do not carry it into the next prefill.
            del out, tt_ids, tables, uploaded_rows
        self.prefill_shapes.add(shape_key)
        if return_device:
            return outputs
        width = max(prompt_lens) if return_all_logits else 1
        result = torch.zeros(len(outputs), width, self.model.config.vocab_size)
        for i, out in enumerate(outputs):
            result[i, : out.shape[1]] = out[0]
        return result

    def _model_step(self, entry, state):
        out = self.model.decode_device(
            entry["tokens"],
            entry["positions"],
            page_tables=state.device_tables,
            kv_cache=state.kv,
            batch=state.batch,
            rope_positions=entry["rope_positions"],
        )
        ttnn.copy(out, entry["logits"])
        ttnn.plus_one(entry["positions"], skip_negative_entries=True)
        ttnn.plus_one(entry["rope_positions"])

    def sample_prefill_outputs(self, outputs, state, sampling_params):
        """Consume compact prefill rows with the canonical sampling trace."""
        batch = len(outputs)
        self.configure_sampling(batch=batch, **sampling_params)
        merged = outputs[0] if batch == 1 else ttnn.concat(outputs, dim=2)
        if batch > 1 and batch < 32:
            merged = ttnn.pad(merged, [(0, 0), (0, 0), (0, 32 - batch), (0, 0)], value=0.0)
        entry = self._entry(state)
        shape = ttnn.Shape([1, 1, 32, self.model.config.vocab_size // 4])
        ttnn.copy(ttnn.reshape(merged, shape, shape), entry["logits"])
        # CQ0 consumes the copy before replay. Release temporary prefill owners
        # so a hot trace never replays with unrelated live postcapture buffers.
        outputs.clear()
        del merged
        ttnn.execute_trace(self.mesh, entry["sampling_trace"], cq_id=0, blocking=False)
        self.counters["sampling_replays"] += 1
        return self._read_tokens(entry["tokens"], batch).reshape(-1, 1)

    def _sample_step(self, entry):
        self.sampler(entry["logits"], tt_out_tok=entry["tokens"])
        # Seed state advances on device, including when an explicit seed was set.
        ttnn.plus_one(self.sampler.seeds_tt_tensor)

    def _entry(self, state):
        key = id(state)
        if key in self.traces:
            entry = self.traces[key]
            if entry["cache_ids"] != tuple(id(t) for pair in state.kv for t in pair) or entry["table_ids"] != {
                k: id(v) for k, v in state.device_tables.items()
            }:
                raise ValueError(
                    "Captured cache/table tensors were replaced; retire traces before replacing backing storage"
                )
            return self.traces[key]
        self._retire_traces()
        entry = {
            "tokens": self._upload(
                torch.zeros(1, 1, 1, 32, dtype=torch.int32),
                ttnn.uint32,
            ),
            "positions": self._upload(torch.zeros(state.batch, dtype=torch.int32)),
            "rope_positions": self._upload(torch.zeros(state.batch, dtype=torch.int32)),
            "logits": self.model.tensor(
                torch.zeros(1, 1, 32, self.model.config.vocab_size),
                dim=3,
                dtype=ttnn.bfloat16,
            ),
            "state": state,
            "cache_owners": tuple(t for pair in state.kv for t in pair),
            "table_owners": tuple(state.device_tables.values()),
            "cache_ids": tuple(id(t) for pair in state.kv for t in pair),
            "table_ids": {k: id(v) for k, v in state.device_tables.items()},
            "remaining_steps": 0,
        }
        for kind, table in state.device_tables.items():
            scratch_table = (
                torch.tensor(state.scratch_pages[kind])[:, None]
                .expand(state.batch, state.capacity // 128)
                .int()
                .contiguous()
            )
            self._refresh(scratch_table, table, "page_table_refreshes")
        logger.info("Warming model trace")
        self._model_step(entry, state)
        logger.info("Warming sampling trace")
        # Setup must not consume a caller's random stream. Preserve the actual
        # on-device seed state even if a previous request already advanced it.
        saved_seeds = ttnn.clone(self.sampler.seeds_tt_tensor)
        self._sample_step(entry)
        ttnn.copy(saved_seeds, self.sampler.seeds_tt_tensor)
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(saved_seeds)
        self.counters["synchronizations"] += 1
        for kind, table in state.device_tables.items():
            self._refresh(state.uploaded_tables[kind], table, "page_table_refreshes")
        self._refresh(torch.zeros(state.batch, dtype=torch.int32), entry["positions"], "position_refreshes")
        self._refresh(torch.zeros(state.batch, dtype=torch.int32), entry["rope_positions"], "rope_refreshes")
        self._refresh(torch.zeros(1, 1, 1, 32, dtype=torch.int32), entry["tokens"], "token_refreshes")
        logger.info("Capturing model trace")
        with corruptible_allocation_scope(self.mesh):
            entry["model_trace"] = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._model_step(entry, state)
            ttnn.end_trace_capture(self.mesh, entry["model_trace"], cq_id=0)
        logger.info("Capturing sampling trace")
        with corruptible_allocation_scope(self.mesh):
            entry["sampling_trace"] = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._sample_step(entry)
            ttnn.end_trace_capture(self.mesh, entry["sampling_trace"], cq_id=0)
        self.traces[key] = entry
        return entry

    def replay(self, entry, *, sample=True):
        """Continue a prepared fixed-slot request without rebuilding device inputs."""
        if self.traces.get(id(entry["state"])) is not entry:
            raise ValueError("Decode handle was retired; prepare the request again")
        if entry["remaining_steps"] <= 0:
            raise ValueError("No decode steps remain within the prepared cache/context")
        ttnn.execute_trace(self.mesh, entry["model_trace"], cq_id=0, blocking=False)
        entry["remaining_steps"] -= 1
        self.counters["model_replays"] += 1
        if sample:
            ttnn.execute_trace(self.mesh, entry["sampling_trace"], cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1

    def prepare_decode(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        sampling_params=None,
        **kwargs,
    ):
        """Prepare explicit scheduler state once; replay(handle) owns token feedback.

        Reprepare only when the scheduler changes slots, tokens, positions or
        page tables. The returned handle is valid until trace retirement/reset.
        """
        state = kv_cache
        self._check_options(kwargs)
        if any(k in kwargs for k in ("top_k", "top_p", "temperature", "seed", "do_sample")):
            raise ValueError("Pass per-slot sampling_params or call configure_sampling at the request boundary")
        if tuple(tokens.shape) != (state.batch, 1) or tuple(start_pos.shape) != (state.batch,):
            raise ValueError("Expected tokens[batch,1] and positions[batch]")
        self._validate_tokens(tokens)
        if start_pos.dtype not in (torch.int32, torch.int64) or bool(
            ((start_pos < -1) | (start_pos >= state.capacity)).any()
        ):
            raise ValueError("Positions must be -1 (inactive) or valid cache positions")
        self._tables(state, page_table)
        entry = self._entry(state)
        if sampling_params is not None:
            self.configure_sampling(batch=state.batch, **sampling_params)
        ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        ids.reshape(-1)[: state.batch] = tokens.reshape(-1)
        self._refresh(ids, entry["tokens"], "token_refreshes")
        self._refresh(start_pos.int(), entry["positions"], "position_refreshes")
        self._refresh(start_pos.int().clamp_min(0), entry["rope_positions"], "rope_refreshes")
        active = start_pos[start_pos >= 0]
        entry["remaining_steps"] = state.capacity - int(active.max()) if active.numel() else state.capacity
        return entry

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        read_from_device=True,
        host_sampling=False,
        sampling_params=None,
        **kwargs,
    ):
        entry = self.prepare_decode(
            tokens, start_pos, page_table=page_table, kv_cache=kv_cache, sampling_params=sampling_params, **kwargs
        )
        self.replay(entry, sample=not host_sampling)
        if host_sampling:
            return self._host_logits(entry["logits"])[0, 0, : kv_cache.batch]
        return self.read_decode_tokens(entry) if read_from_device else entry["tokens"]

    def read_decode_tokens(self, entry):
        if self.traces.get(id(entry["state"])) is not entry:
            raise ValueError("Decode handle was retired")
        return self._read_tokens(entry["tokens"], entry["state"].batch)

    def prefill_logits(self, prompt_token_ids):
        state = self._owned(1, len(prompt_token_ids))
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=state.page_tables,
            kv_cache=state,
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
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        seed=0,
        do_sample=True,
        stop_on_eos=True,
        **kwargs,
    ):
        start = time.perf_counter()
        self._check_options(kwargs)
        if not enable_trace:
            raise ValueError("This generator requires traced decode")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        prompt = torch.as_tensor(prompt_token_ids)
        if prompt.ndim != 1 or not 0 < prompt.numel() <= self.max_seq_len:
            raise ValueError("Expected one nonempty prompt within the supported context")
        self._validate_tokens(prompt)
        length = len(prompt_token_ids)
        if length + max(0, max_new_tokens - 1) > self.max_seq_len:
            raise ValueError("Prompt plus consumed decode tokens exceeds the supported context")
        if max_new_tokens == 0:
            return []
        if host_sampling and do_sample and temperature != 0 and top_k != 1:
            raise ValueError("Explicit host compatibility mode currently supports greedy sampling only")
        state = self._owned(1, length + max_new_tokens - 1)
        self._tables(state, state.page_tables)
        self.configure_sampling(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed, do_sample=do_sample)
        out = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=[length],
            return_device=True,
        )[0]
        entry = self._entry(state)
        self._refresh(torch.full((32,), seed, dtype=torch.int32), self.sampler.seeds_tt_tensor, "seed_refreshes")
        shape = ttnn.Shape([1, 1, 32, self.model.config.vocab_size // 4])
        ttnn.copy(ttnn.reshape(out, shape, shape), entry["logits"])
        del out
        self._refresh(torch.tensor([length], dtype=torch.int32), entry["positions"], "position_refreshes")
        self._refresh(torch.tensor([length], dtype=torch.int32), entry["rope_positions"], "rope_refreshes")
        entry["remaining_steps"] = state.capacity - length
        if host_sampling:
            predictions = [int(self._host_logits(entry["logits"])[0, 0, 0].argmax())]
        else:
            ttnn.execute_trace(self.mesh, entry["sampling_trace"], cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
            predictions = [int(self._read_tokens(entry["tokens"], 1)[0])]
        self.last_ttft_s = time.perf_counter() - start
        start = time.perf_counter()
        for step in range(max_new_tokens - 1):
            if next_input is None and stop_on_eos and predictions[-1] in self.eos_token_ids:
                break
            if next_input is not None or host_sampling:
                token = next_input(step, predictions[-1]) if next_input is not None else predictions[-1]
                ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                ids.reshape(-1)[0] = token
                self._refresh(ids, entry["tokens"], "token_refreshes")
            self.replay(entry, sample=not host_sampling)
            if host_sampling:
                predictions.append(int(self._host_logits(entry["logits"])[0, 0, 0].argmax()))
            else:
                predictions.append(int(self._read_tokens(entry["tokens"], 1)[0]))
        if next_input is not None:
            next_input(max_new_tokens - 1, predictions[-1])
        self.last_decode_s = time.perf_counter() - start
        return predictions

    def reset(self):
        for state in self.states.values():
            for cache in state.kv:
                for tensor in cache:
                    ttnn.multiply(tensor, 0.0, output_tensor=tensor)
        for entry in self.traces.values():
            if not any(entry["state"] is state for state in self.states.values()):
                continue  # External cache and scheduling state remain caller-owned.
            entry["remaining_steps"] = 0
            self._refresh(torch.zeros(1, 1, 1, 32, dtype=torch.int32), entry["tokens"], "token_refreshes")
            self._refresh(
                torch.zeros(entry["positions"].shape[0], dtype=torch.int32), entry["positions"], "position_refreshes"
            )
            self._refresh(
                torch.zeros(entry["positions"].shape[0], dtype=torch.int32), entry["rope_positions"], "rope_refreshes"
            )
        self.configure_sampling()

    def _retire_traces(self):
        if self.traces:
            ttnn.synchronize_device(self.mesh)
            self.counters["synchronizations"] += 1
        for entry in self.traces.values():
            for name in ("model_trace", "sampling_trace"):
                ttnn.release_trace(self.mesh, entry[name])
        self.traces.clear()

    def close(self):
        self._retire_traces()


def build_generator(mesh_device, *, checkpoint=None, layer_indices=None, max_seq_len=2048):
    model = Gemma4Model(mesh_device, checkpoint=checkpoint, layer_indices=layer_indices)
    return Gemma4Generator(model, max_seq_len=max_seq_len)

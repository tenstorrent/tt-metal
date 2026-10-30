# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Explicit-cache generator for Gemma4 TP4.

The optimized loop replays two traces. Only the sampling trace writes the next
token; position advances in the model trace. Host sampling is opt-in compatibility.
"""

import gemma_history_reservation as _history_reservation
import json
import math
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from loguru import logger
from transformers import AutoTokenizer, GenerationConfig

import ttnn
from models.common.sampling.tt_sampling import TTSampling
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.model import Gemma4Model
from models.demos.gemma4_31b_qb2.tt.history_bank import HistoryBank
from models.demos.gemma4_31b_qb2.tt.prefill_batch import DeferredPrefillHistory, padded_span, plan_prefill_batches


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
    history_bank: object = None

    def release_history(self, key):
        self.prefill_history.pop(key, None)
        if self.history_bank is not None:
            self.history_bank.release(key)

    def clear_histories(self):
        self.prefill_history.clear()
        if self.history_bank is not None:
            self.history_bank.clear()


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
        # Speculative verify traces, keyed by id(CacheState); see spec_decode().
        self.spec_traces = {}
        # Keep explicit decode/sampler IO across conservative trace retirement.
        # The next capture gets a new entry object, so old handles stay invalid.
        self._entry_storage = None
        self.prefill_shapes = set()
        # Experimental bounded closure gate; never enabled by serving defaults.
        self._prepared_prefill_state = None
        self._serving_prefill_prepared_state = None
        self._serving_prefill_has_real_work = False
        self.runtime_page_selector = None
        if os.environ.get("GEMMA_PREFILL_RUNTIME_PAGES", "0") == "1":
            from .runtime_page_selector import RuntimePageSelector
            self.runtime_page_selector = RuntimePageSelector(self.mesh)
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
        self._entry_storage = None
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
        state = CacheState(kv, tables, batch, pages * 128, scratch_pages=scratch)
        self.setup_history_bank(state)
        return state

    def setup_history_bank(self, state, *, capacity=None):
        """Explicit setup before first capture; experiment remains default off."""
        if os.environ.get("GEMMA_PREFILL_HISTORY_BANK", "0") != "1":
            return
        capacity = state.batch if capacity is None else capacity
        if type(capacity) is not int or not 1 <= capacity <= state.batch:
            raise ValueError("History capacity must be an integer in 1..physical width")
        if state.history_bank is not None:
            if len(state.history_bank.roots) != capacity:
                raise RuntimeError("History capacity cannot change after allocation")
            return
        if self.traces or state.prefill_history:
            raise RuntimeError("History bank setup must precede capture and requests")
        state.history_bank = HistoryBank(self.model, capacity)

    def _bank_history(self, state, key, start, aligned_start, tables, slot, retain):
        previous = state.prefill_history.pop(key, None)
        if start == 0:
            state.history_bank.release(key)
            previous = None
        elif previous is not None and previous["end"] != start:
            raise ValueError("Bank history does not match scheduler continuation")
        restore = None
        if start and previous is None:
            if not state.vllm_owned or start != aligned_start:
                raise ValueError("Cached prefill must start at a whole page in the serving cache")
            restore = lambda i: self._restore_prefill_history(state, tables, slot, start, layer_index=i)[0]
        return state.history_bank.session(
            key, previous=previous["layers"] if previous else None,
            rewind=start-aligned_start, restore=restore, retain=retain,
        )

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
            for tensor, expected_dtype in zip(pair, layer.kv_dtypes):
                if (
                    len(tensor.shape) != 4
                    or tuple(tensor.shape)[1:] != (layer.kv_heads, 128, layer.head_dim)
                    or tensor.dtype != expected_dtype
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
            for tensor, expected_dtype in zip(pair, layer.kv_dtypes):
                if (
                    tuple(tensor.shape)[1:] != (layer.kv_heads, 128, layer.head_dim)
                    or tensor.dtype != expected_dtype
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

    def _restore_prefill_history(self, state, page_tables, slot, start, *, layer_index=None):
        """Gather the preceding sliding window from vLLM's retained KV pages.

        The scheduler owns cache identity and eviction. Only pages in this
        request's current table are read; null pages are never valid history.
        Decode and cached continuation consume the same stored BF8 values.
        """
        histories = []
        indices = range(len(self.model.layers)) if layer_index is None else [layer_index]
        for i in indices:
            layer = self.model.layers[i]
            if not layer.sliding:
                histories.append(None)
                continue
            first = max(0, start - layer.window) // Decoder.PAGE_SIZE
            pages = page_tables[i][slot, first : start // Decoder.PAGE_SIZE].tolist()
            if not pages or any(p <= 0 or p >= state.kv[i][0].shape[0] for p in pages):
                raise ValueError(f"Missing sliding prefix pages for layer {i} at position {start}: {pages}")
            pair = []
            for cache in state.kv[i]:
                parts = [self.runtime_page_selector.select(cache, p) if self.runtime_page_selector is not None
                         else cache[p : p + 1, :, :, :] for p in pages]
                tail = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
                pair.append(ttnn.typecast(tail, ttnn.bfloat16))
            histories.append(tuple(pair))
        return histories

    def prepare_serving_prefill(self, state):
        """Opt-in fresh-cache preparation; capture remains at its normal boundary."""
        if os.environ.get("GEMMA_SERIAL_PREPARED_RETENTION", "0") != "1":
            return
        required = ("GEMMA_PREFILL_HISTORY_BANK", "GEMMA_PREFILL_RUNTIME_POSITION",
                    "GEMMA_PREFILL_RUNTIME_FINAL_TILE", "GEMMA_PREFILL_RUNTIME_PAGES",
                    "GEMMA_PREFILL_HOST_FILL_TABLES", "TT_METAL_TRACE_ALLOC_TRACKING")
        if any(os.environ.get(key, "0") != "1" for key in required):
            raise RuntimeError("Serial prepared retention requires all fixed-owner controls and tracking")
        if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
            raise RuntimeError("Program-cache allocations must remain tracked")
        pool_profile = os.environ.get("GEMMA_SERIAL_PREPARED_POOL_PROFILE", "f606208")
        c32_profiles = {"c32a24_f262144": (24, "262144"), "c32a20_f327680": (20, "327680"), "c32a24_f458752": (24, "458752"), "c32a32_p2_f412672": (32, "412672"), "c32a32_p2_f443392": (32, "443392"), "c32a32_p2_f458752": (32, "458752"), "c32a32_p2_f626688": (32, "626688")}
        c32_profile = pool_profile in c32_profiles
        expected_batch = 32 if c32_profile else 16
        expected_admission = c32_profiles[pool_profile][0] if c32_profile else 16
        expected_history = _history_reservation.serving_history_capacity(
            batch=state.batch, admitted=expected_admission,
            capacity=state.capacity)
        if _history_reservation.bank_mode() is not None and (
                not getattr(state, "vllm_owned", False) or not hasattr(state, "serving_admission_limit")):
            raise RuntimeError("Reduced prepared history requires explicit serving state")
        if c32_profile and (
                getattr(state, "serving_admission_limit", None) != expected_admission
                or os.environ.get("GEMMA_EXPERIMENT_KV_POOL_TOKENS") != c32_profiles[pool_profile][1]
                or os.environ.get("GEMMA_EXPERIMENT_SHARE_BF8_SLIDING_QKV") != "1"
                or os.environ.get("GEMMA_EXPERIMENT_SHARE_BF8_DOWN") != "1"
                or os.environ.get("GEMMA_REPETITION_PRECISION_POLICY") != "bf8"
                or os.environ.get("GEMMA_EXPERIMENT_KV_DTYPE") != "bf8kv"):
            raise RuntimeError("C32 declared preparation requires explicit admission and joint BF8/BF8KV profile")
        if (os.environ.get("GEMMA_BATCH_PREFILL_MAX_BATCH", "1") != "1"
                or state.batch != expected_batch or state.capacity != 262144
                or len(self.model.layers) != 60 or state.history_bank is None
                or self.traces or state.prefill_history or state.history_bank.leases
                or len(state.history_bank.roots) != expected_history
                or self._serving_prefill_prepared_state is not None):
            raise RuntimeError("Experimental preparation requires fresh full60 declared-width serial state")
        pool_profile = os.environ.get("GEMMA_SERIAL_PREPARED_POOL_PROFILE", "f606208")
        pool_rows = {"f606208": 6128, "f262144": 3440, "f311296": 3824, "f393216": 4464, "c32a24_f262144": 3832, "c32a20_f327680": 4164, "c32a24_f458752": 5368, "c32a32_p2_f412672": 5368, "c32a32_p2_f443392": 5608, "c32a32_p2_f458752": 5728, "c32a32_p2_f626688": 7040}
        if pool_profile not in pool_rows:
            raise RuntimeError("Unsupported serial prepared alias pool profile")
        if pool_profile == "f311296" and (
                os.environ.get("GEMMA_EXPERIMENT_KV_POOL_TOKENS") != "311296"
                or os.environ.get("GEMMA_EXPERIMENT_SHARE_BF8_SLIDING_QKV") != "1"
                or os.environ.get("GEMMA_REPETITION_PRECISION_POLICY") != "bf8"
                or os.environ.get("GEMMA_EXPERIMENT_KV_DTYPE") != "bf8kv"):
            raise RuntimeError("F311296 preparation requires declared shared BF8/BF8KV source")
        if pool_profile == "f393216" and (
                os.environ.get("GEMMA_EXPERIMENT_KV_POOL_TOKENS") != "393216"
                or os.environ.get("GEMMA_EXPERIMENT_SHARE_BF8_SLIDING_QKV") != "1"
                or os.environ.get("GEMMA_EXPERIMENT_SHARE_BF8_DOWN") != "1"
                or os.environ.get("GEMMA_REPETITION_PRECISION_POLICY") != "bf8"
                or os.environ.get("GEMMA_EXPERIMENT_KV_DTYPE") != "bf8kv"):
            raise RuntimeError("F393216 requires joint shared BF8/BF8KV source")
        wide_rows = pool_rows[pool_profile]
        if any(int(pair[0].shape[0]) != (wide_rows if layer.sliding else 2 * wide_rows)
               for layer, pair in zip(self.model.layers, state.kv)):
            raise RuntimeError("Experimental preparation requires the declared alias pool")
        from models.demos.gemma4_31b_qb2.tt.serial_staging_preparation import prepare as staging
        from models.demos.gemma4_31b_qb2.tt.serial_compute_preparation import prepare as compute
        from models.demos.gemma4_31b_qb2.tt.serial_terminal_preparation import prepare as terminal
        started = time.perf_counter()
        def record(row):
            logger.info("SERIAL_PREFILL_PREPARATION " + json.dumps(row))
        staging(self.model, self.mesh, 8319, record)
        compute(self, state, record)
        terminal(self, state, record)
        # Do not capture against the adapter's initially all-null page tables.
        # The first normal sampling/decode entry arms retention after capture.
        self._serving_prefill_prepared_state = state
        record({"phase": "ready_for_first_real_capture", "seconds": time.perf_counter() - started})

    def _check_serving_prefill_family(self, tokens, kwargs):
        state = self._serving_prefill_prepared_state
        if state is None:
            return
        lengths = kwargs.get("prompt_lens", ())
        starts = kwargs.get("start_positions")
        starts = [0] * len(lengths) if starts is None else list(starts)
        supported = (kwargs.get("kv_cache") is state
                     and os.environ.get("GEMMA_BATCH_PREFILL_MAX_BATCH", "1") == "1"
                     and not kwargs.get("return_all_logits", False)
                     and 0 < len(lengths) <= state.batch and len(starts) == len(lengths)
                     and all(type(n) is int and type(s) is int
                             and 0 <= s < n <= state.capacity and n - s <= 8192
                             and n - (s // 128 * 128) <= 8319
                             for n, s in zip(lengths, starts)))
        if supported:
            return
        # Retire before any unsupported computation, never after a partial write.
        self._serving_prefill_prepared_state = None
        self._serving_prefill_has_real_work = False
        self._prepared_prefill_state = None
        self.mesh.set_program_cache_misses_allowed(True)
        self._retire_traces()
        logger.info("SERIAL_PREFILL_RETENTION_FALLBACK unsupported_family; conservative retirement restored")

    def enable_prepared_prefill_trace_retention(self, state):
        """Arm a strict diagnostic only after native eager preparation.

        This does not assert preparation is complete. Every subsequent prefill
        forbids native program misses and the allocator tracker still checks
        surviving captures. Unsupported signatures fail instead of recapturing.
        The opt-in serving hook calls this only after successful real prefill.
        """
        if self._prepared_prefill_state is not None:
            raise RuntimeError("Prepared prefill retention is already enabled")
        if state.history_bank is None or any(os.environ.get(key, "0") != "1" for key in (
            "GEMMA_PREFILL_HISTORY_BANK", "GEMMA_PREFILL_RUNTIME_POSITION",
            "GEMMA_PREFILL_RUNTIME_FINAL_TILE",
        )):
            raise RuntimeError("Prepared retention requires fixed histories and runtime coordinates")
        if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1" or os.environ.get(
            "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"
        ) != "0":
            raise RuntimeError("Diagnostic requires complete trace allocation tracking")
        if id(state) not in self.traces:
            raise RuntimeError("Prepare the decode and sampling traces before enabling retention")
        # Existing _entry checks the captured cache/table owners on every use.
        self._entry(state)
        self.mesh.set_program_cache_misses_allowed(False)
        self._prepared_prefill_state = state

    def prefill_forward(self, tokens, **kwargs):
        self._check_serving_prefill_family(tokens, kwargs)
        if self._prepared_prefill_state is None:
            result = self._prefill_forward_impl(tokens, **kwargs)
            if (self._serving_prefill_prepared_state is not None
                    and self._serving_prefill_prepared_state is kwargs.get("kv_cache")):
                self._serving_prefill_has_real_work = True
            return result
        if kwargs.get("kv_cache") is not self._prepared_prefill_state:
            raise RuntimeError("Prepared retention cannot change the cache state")
        if id(self._prepared_prefill_state) not in self.traces:
            raise RuntimeError("Prepared traces were retired; no implicit recapture is permitted")
        self._entry(self._prepared_prefill_state)
        before = self.mesh.num_program_cache_entries()
        result = self._prefill_forward_impl(tokens, **kwargs)
        if self.mesh.num_program_cache_entries() != before:
            raise RuntimeError("Native program cache grew in prepared prefill")
        return result

    def _prefill_forward_impl(self, tokens, **kwargs):
        state = kwargs.get("kv_cache")
        bank = getattr(state, "history_bank", None)
        if os.environ.get("GEMMA_PREFILL_HISTORY_BANK", "0") == "1" and bank is None:
            raise RuntimeError("Call setup_history_bank explicitly before capture or prefill")
        keys = kwargs.get("history_keys")
        if bank is not None and keys is not None and (len(keys) != len(kwargs.get("prompt_lens", ())) or len(set(keys)) != len(keys)):
            raise ValueError("Expected distinct stable history keys, one per prefill row")
        if bank is not None and keys is not None:
            reservation = _history_reservation.context()
            if reservation is not None and reservation.scheduler is not None:
                _history_reservation.sequential_model_preflight(state, keys, kwargs.get("start_positions"), kwargs.get("prompt_lens"), kwargs.get("retain_prefill_history"))
            elif len(set(bank.leases) | set(keys)) > len(bank.roots):
                raise RuntimeError("Live plus incoming history owners exceed configured capacity")
        try:
            maximum = int(os.environ.get("GEMMA_BATCH_PREFILL_MAX_BATCH", "1"))
            if maximum > 1 and len(kwargs.get("prompt_lens", ())) > 1 and not kwargs.get("return_all_logits", False):
                return self._prefill_jobs_forward(tokens, **kwargs)
            return self._prefill_serial_forward(tokens, **kwargs)
        except Exception:
            if bank is not None:
                for key in keys or ():
                    state.release_history(key)
            raise

    def _prefill_jobs_forward(
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
        retain_prefill_history=None,
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
        if retain_prefill_history is not None:
            if (
                history_keys is None
                or len(retain_prefill_history) != len(prompt_lens)
                or any(type(flag) is not bool for flag in retain_prefill_history)
            ):
                raise ValueError("Expected one explicit history retention bool per prefill row")
        starts = [0] * len(prompt_lens) if start_positions is None else list(start_positions)
        if len(starts) != len(prompt_lens) or any(s < 0 or s >= n for s, n in zip(starts, prompt_lens)):
            raise ValueError("Prefill must advance each request from its current position")
        shape_key = (state.batch, state.capacity, tuple(prompt_lens), tuple(starts), bool(return_all_logits))
        if self._prepared_prefill_state is None and (history_keys is not None or shape_key not in self.prefill_shapes):
            # Request histories allocate persistent K/V tails even for a seen
            # signature. Unseen signatures can also allocate native cached state.
            # Both must be created before the next trace capture.
            self._retire_traces()
        self._tables(state, page_table)
        jobs = []
        for row, (slot, length) in enumerate(zip(slots, prompt_lens)):
            if not 0 < length <= state.capacity or length > tokens.shape[1]:
                raise ValueError("Invalid logical prompt length")
            start = starts[row]
            aligned_start = start // Decoder.PAGE_SIZE * Decoder.PAGE_SIZE
            chunk_length = length - aligned_start
            histories = None
            if history_keys is not None and state.history_bank is not None:
                histories = self._bank_history(
                    state, history_keys[row], start, aligned_start, page_table, slot,
                    retain_prefill_history is None or retain_prefill_history[row],
                )
            elif history_keys is not None:
                key = history_keys[row]
                if start == 0:
                    histories = [None] * len(self.model.layers)
                else:
                    previous = state.prefill_history.get(key)
                    if previous is None:
                        if not state.vllm_owned or start != aligned_start:
                            raise ValueError("Cached prefill must start at a whole page in the serving cache")
                        previous = {
                            "end": start,
                            "layers": DeferredPrefillHistory(
                                len(self.model.layers),
                                lambda i, slot=slot, start=start: self._restore_prefill_history(
                                    state, page_table, slot, start, layer_index=i
                                )[0],
                            ),
                        }
                    if previous["end"] != start:
                        raise ValueError(
                            "Prefill history does not match the scheduler continuation: "
                            f"key={key}, saved_end={previous['end']}, start={start}"
                        )
                    rewind = start - aligned_start
                    # Never leave a partially consumed list under its old end
                    # position if a later device operation raises. Successful
                    # prefill publishes the completed history below.
                    state.prefill_history.pop(key, None)
                    # This request owns the list; other requests never borrow it.
                    # Replace each consumed tail as soon as its CQ0 slice has
                    # been enqueued. Bank tensors may have another root owner,
                    # so use normal reference release, never forced deallocation.
                    histories = previous["layers"]
                    for index, tail in enumerate(histories):
                        if tail is None:
                            continue
                        valid = tail[0].shape[2] - rewind
                        if valid == 0:
                            histories[index] = None
                        else:
                            begin = max(0, valid - 1024)
                            histories[index] = tuple(t[:, :, begin:valid, :] for t in tail)
                        del tail
                    del previous
            ids = torch.zeros(1, (chunk_length + 31) // 32 * 32, dtype=torch.int32)
            ids[0, :chunk_length] = tokens[row, aligned_start:length]
            tables, host_rows = {}, {}
            for kind, table in page_table.items():
                sentinel = state.vllm_owned and self.model.layers[kind].kind == "sliding_attention"
                table_key = (id(table), sentinel)
                if table_key not in host_rows:
                    row_table = table[slot : slot + 1]
                    if sentinel:
                        row_table = row_table.masked_fill(row_table == 0, -1)
                    host_rows[table_key] = row_table
                tables[kind] = host_rows[table_key]
            jobs.append(
                dict(
                    row=row,
                    slot=slot,
                    length=length,
                    span=chunk_length,
                    start=aligned_start,
                    ids=ids,
                    tables=tables,
                    histories=histories,
                )
            )
        outputs = [None] * len(jobs)
        maximum = int(os.environ.get("GEMMA_BATCH_PREFILL_MAX_BATCH", "1"))
        for indices in plan_prefill_batches(
            [job["span"] for job in jobs],
            maximum,
            long_spans=os.environ.get("GEMMA_BATCH_PREFILL_LONG", "0") == "1",
            packed_rows=self.model.layers[0].PREFILL_BATCH_ROWS,
        ):
            group = [jobs[index] for index in indices]
            padded = max(padded_span(job["span"]) for job in group)
            if os.environ.get("GEMMA_BATCH_PREFILL_REPORT", "0") == "1":
                logger.info(
                    "PREFILL_REQUEST_BATCH "
                    + json.dumps(
                        {
                            "requests": len(group),
                            "slots": [job["slot"] for job in group],
                            "spans": [job["span"] for job in group],
                            "padded_span": padded,
                            "starts": [job["start"] for job in group],
                            "time": time.time(),
                        }
                    )
                )
            if len(group) == 1:
                job = group[0]
                tt_ids = self._upload(job["ids"], ttnn.uint32)
                tables, uploaded = {}, {}
                for kind, table in job["tables"].items():
                    if id(table) not in uploaded:
                        uploaded[id(table)] = self._upload(table)
                        self.counters["prefill_table_uploads"] += 1
                    tables[kind] = uploaded[id(table)]
                device_outputs = [
                    self.model.prefill_device(
                        tt_ids,
                        sequence_length=job["span"],
                        page_tables=tables,
                        kv_cache=state.kv,
                        start_pos=job["start"],
                        histories=job["histories"],
                    )
                ]
            else:
                packed = torch.zeros(len(group), padded, dtype=torch.int32)
                for index, job in enumerate(group):
                    packed[index, : job["span"]] = job["ids"][0, : job["span"]]
                tt_ids = self._upload(packed, ttnn.uint32)
                tables, fill_tables, uploaded = {}, {}, {}
                for kind in group[0]["tables"]:
                    # Layers sharing immutable host rows also share device tables
                    # within this prefill group. A later scheduler update gets
                    # new uploads; mutable decode tables are never reused here.
                    table_key = tuple(id(job["tables"][kind]) for job in group)
                    if table_key not in uploaded:
                        table = self._upload(torch.cat([job["tables"][kind] for job in group], dim=0))
                        fill = torch.full((len(group), padded // 128), -1, dtype=torch.int32)
                        for index, job in enumerate(group):
                            pages = (job["span"] + 127) // 128
                            begin = job["start"] // 128
                            fill[index, :pages] = job["tables"][kind][0, begin : begin + pages]
                        uploaded[table_key] = (table, self._upload(fill))
                        del table, fill
                        self.counters["prefill_table_uploads"] += 2
                    tables[kind], fill_tables[kind] = uploaded[table_key]
                batch_indices = self._upload(torch.arange(len(group), dtype=torch.int32))
                merged = self.model.prefill_batched_device(
                    tt_ids,
                    batch=len(group),
                    sequence_length=padded,
                    page_tables=tables,
                    kv_cache=state.kv,
                    logical_lengths=[job["span"] for job in group],
                    start_positions=[job["start"] for job in group],
                    histories=[job["histories"] for job in group] if history_keys is not None else None,
                    retain_histories=[retain_prefill_history[job["row"]] for job in group]
                    if retain_prefill_history is not None
                    else None,
                    fill_page_tables=fill_tables,
                    batch_indices=batch_indices,
                )
                del fill_tables, batch_indices
                device_outputs = [merged[:, :, index : index + 1, :] for index in range(len(group))]
                del merged
            for job, out in zip(group, device_outputs):
                row = job["row"]
                if history_keys is not None:
                    if retain_prefill_history is None or retain_prefill_history[row]:
                        state.prefill_history[history_keys[row]] = {"end": job["length"], "layers": job["histories"].publish() if state.history_bank is not None else job["histories"]}
                    else:
                        state.release_history(history_keys[row])
                outputs[row] = (
                    out if return_device else self._host_logits(out).reshape(1, -1, self.model.config.vocab_size)
                )
                job["histories"] = None
            del out, device_outputs, tt_ids, tables, uploaded, group
        self.prefill_shapes.add(shape_key)
        if return_device:
            return outputs
        width = max(prompt_lens) if return_all_logits else 1
        result = torch.zeros(len(outputs), width, self.model.config.vocab_size)
        for i, out in enumerate(outputs):
            result[i, : out.shape[1]] = out[0]
        return result

    def _prefill_serial_forward(
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
        retain_prefill_history=None,
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
        if retain_prefill_history is not None:
            if (
                history_keys is None
                or len(retain_prefill_history) != len(prompt_lens)
                or any(type(flag) is not bool for flag in retain_prefill_history)
            ):
                raise ValueError("Expected one explicit history retention bool per prefill row")
        starts = [0] * len(prompt_lens) if start_positions is None else list(start_positions)
        if len(starts) != len(prompt_lens) or any(s < 0 or s >= n for s, n in zip(starts, prompt_lens)):
            raise ValueError("Prefill must advance each request from its current position")
        shape_key = (state.batch, state.capacity, tuple(prompt_lens), tuple(starts), bool(return_all_logits))
        if self._prepared_prefill_state is None and (history_keys is not None or shape_key not in self.prefill_shapes):
            # Request histories allocate persistent K/V tails even for a seen
            # signature. Unseen signatures can also allocate native cached state.
            # Both must be created before the next trace capture.
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
            if history_keys is not None and state.history_bank is not None:
                histories = self._bank_history(
                    state, history_keys[row], start, aligned_start, page_table, slot,
                    retain_prefill_history is None or retain_prefill_history[row],
                )
            elif history_keys is not None:
                key = history_keys[row]
                if start == 0:
                    histories = [None] * len(self.model.layers)
                else:
                    previous = state.prefill_history.get(key)
                    if previous is None:
                        if not state.vllm_owned or start != aligned_start:
                            raise ValueError("Cached prefill must start at a whole page in the serving cache")
                        previous = {
                            "end": start,
                            "layers": self._restore_prefill_history(state, page_table, slot, start),
                        }
                    if previous["end"] != start:
                        raise ValueError(
                            "Prefill history does not match the scheduler continuation: "
                            f"key={key}, saved_end={previous['end']}, start={start}"
                        )
                    rewind = start - aligned_start
                    # Never leave a partially consumed list under its old end
                    # position if a later device operation raises. Successful
                    # prefill publishes the completed history below.
                    state.prefill_history.pop(key, None)
                    # This request owns the list; other requests never borrow it.
                    # Replace each consumed tail as soon as its CQ0 slice has
                    # been enqueued. Bank tensors may have another root owner,
                    # so use normal reference release, never forced deallocation.
                    histories = previous["layers"]
                    for index, tail in enumerate(histories):
                        if tail is None:
                            continue
                        valid = tail[0].shape[2] - rewind
                        if valid == 0:
                            histories[index] = None
                        else:
                            begin = max(0, valid - 1024)
                            histories[index] = tuple(t[:, :, begin:valid, :] for t in tail)
                        del tail
                    del previous
            ids = torch.zeros(1, (chunk_length + 31) // 32 * 32, dtype=torch.int32)
            ids[0, :chunk_length] = tokens[row, aligned_start:length]
            tt_ids = self._upload(ids, ttnn.uint32)
            tables, uploaded_rows = {}, {}
            serial_tables, serial_shared = {}, {}
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
                if os.environ.get("GEMMA_PREFILL_HOST_FILL_TABLES", "0") == "1":
                    layer = (self.model.layers[kind] if isinstance(kind, int) else
                             next(layer for layer in self.model.layers if layer.kind == kind))
                    selected = {}
                    for begin in range(0, chunk_length, layer.PREFILL_CHUNK):
                        end = min(begin + layer.PREFILL_CHUNK, chunk_length)
                        first = (aligned_start + begin) // 128
                        count = (end - begin + 127) // 128
                        fill_key = (key, first, count)
                        if fill_key not in serial_shared:
                            host_table = table[slot:slot+1, first:first+count]
                            if sentinel:
                                host_table = host_table.masked_fill(host_table == 0, -1)
                            serial_shared[fill_key] = self._upload(host_table.contiguous())
                        selected[begin] = serial_shared[fill_key]
                    serial_tables[kind] = selected
            out = self.model.prefill_device(
                tt_ids,
                sequence_length=chunk_length,
                page_tables=tables,
                kv_cache=state.kv,
                all_logits=return_all_logits,
                start_pos=aligned_start,
                histories=histories,
                serial_fill_tables=serial_tables or None,
            )
            if history_keys is not None:
                if retain_prefill_history is None or retain_prefill_history[row]:
                    state.prefill_history[history_keys[row]] = {"end": length, "layers": histories.publish() if state.history_bank is not None else histories}
                else:
                    state.release_history(history_keys[row])
                # Final prefill rows cannot consume BF16 chunk tails again.
                # Native KV owners persist; CQ0 orders all preceding consumers
                # before later allocations may reuse these tail addresses.
                del histories
            outputs.append(
                out if return_device else self._host_logits(out).reshape(1, -1, self.model.config.vocab_size)
            )
            # Device-return outputs retain their own owners in outputs. Host
            # mode has consumed out; do not carry it into the next prefill.
            del out, tt_ids, tables, uploaded_rows, serial_tables, serial_shared
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
        observer_dir = os.environ.get("GEMMA_REPETITION_OBSERVER_DIR")
        if observer_dir:
            from pathlib import Path
            root = Path(observer_dir)
            self._repetition_prefill_count = getattr(self, "_repetition_prefill_count", 0) + 1
            index = self._repetition_prefill_count
            logits = self._host_logits(entry["logits"])[0, 0, 0].clone()
            sampled = ttnn.to_torch(ttnn.get_device_tensors(entry["tokens"])[0]).reshape(-1).tolist()
            torch.save(logits, root / f"prefill-logits-{index}.pt")
            (root / f"prefill-{index}.json").write_text(json.dumps({"index": index, "sampled_tokens": sampled, "shape": list(logits.shape), "batch": batch}) + "\n")
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
        trace_entry_start = time.perf_counter()
        self._retire_traces()
        cache_ids = tuple(id(t) for pair in state.kv for t in pair)
        table_ids = {k: id(v) for k, v in state.device_tables.items()}
        cache_buffer_ids = tuple(t.buffer_unique_id() for pair in state.kv for t in pair)
        table_buffer_ids = {k: v.buffer_unique_id() for k, v in state.device_tables.items()}
        storage = self._entry_storage
        if (
            storage is None
            or storage["state"] is not state
            or storage["cache_ids"] != cache_ids
            or storage["table_ids"] != table_ids
            or storage["cache_buffer_ids"] != cache_buffer_ids
            or storage["table_buffer_ids"] != table_buffer_ids
        ):
            storage = {
                "state": state,
                "cache_ids": cache_ids,
                "table_ids": table_ids,
                "cache_buffer_ids": cache_buffer_ids,
                "table_buffer_ids": table_buffer_ids,
                "tokens": self._upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32),
                "positions": self._upload(torch.zeros(state.batch, dtype=torch.int32)),
                "rope_positions": self._upload(torch.zeros(state.batch, dtype=torch.int32)),
                "logits": self.model.tensor(
                    torch.zeros(1, 1, 32, self.model.config.vocab_size),
                    dim=3,
                    dtype=ttnn.bfloat16,
                ),
            }
            self._entry_storage = storage
            prepared = False
        else:
            prepared = True
        entry = {
            **storage,
            "state": state,
            "cache_owners": tuple(t for pair in state.kv for t in pair),
            "table_owners": tuple(state.device_tables.values()),
            "remaining_steps": 0,
        }
        if not prepared:
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
        else:
            # Prefill can still be queued after conservative retirement.
            ttnn.synchronize_device(self.mesh)
            self.counters["synchronizations"] += 1
        for kind, table in state.device_tables.items():
            self._refresh(state.uploaded_tables[kind], table, "page_table_refreshes")
        self._refresh(torch.zeros(state.batch, dtype=torch.int32), entry["positions"], "position_refreshes")
        self._refresh(torch.zeros(state.batch, dtype=torch.int32), entry["rope_positions"], "rope_refreshes")
        self._refresh(torch.zeros(1, 1, 1, 32, dtype=torch.int32), entry["tokens"], "token_refreshes")
        programs_before = self.mesh.num_program_cache_entries()
        self.mesh.set_program_cache_misses_allowed(False)
        try:
            logger.info("Capturing model trace")
            entry["model_trace"] = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._model_step(entry, state)
            ttnn.end_trace_capture(self.mesh, entry["model_trace"], cq_id=0)
            logger.info("Capturing sampling trace")
            entry["sampling_trace"] = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._sample_step(entry)
            ttnn.end_trace_capture(self.mesh, entry["sampling_trace"], cq_id=0)
        finally:
            # Unseen eager prefill shapes still need to compile before the
            # existing conservative prefill retirement and next capture.
            self.mesh.set_program_cache_misses_allowed(True)
        programs_after = self.mesh.num_program_cache_entries()
        if programs_before != programs_after:
            raise RuntimeError("Native program cache grew during trace capture")
        self.traces[key] = entry
        if (self._serving_prefill_prepared_state is state and self._serving_prefill_has_real_work
                and self._prepared_prefill_state is None):
            self.enable_prepared_prefill_trace_retention(state)
            logger.info("SERIAL_PREFILL_RETENTION_ARMED first_real_capture")
        logger.info(
            "TRACE_ENTRY_COST "
            + json.dumps(
                {
                    "seconds": time.perf_counter() - trace_entry_start,
                    "prepared": prepared,
                    "programs_before": programs_before,
                    "programs_after": programs_after,
                }
            )
        )
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
        if self._prepared_prefill_state is not None:
            raise RuntimeError("Prepared trace retention forbids implicit retirement; close explicitly")
        if self.traces or self.spec_traces:
            ttnn.synchronize_device(self.mesh)
            self.counters["synchronizations"] += 1
        for entry in list(self.traces.values()) + list(self.spec_traces.values()):
            for name in ("model_trace", "sampling_trace"):
                ttnn.release_trace(self.mesh, entry[name])
        self.traces.clear()
        self.spec_traces.clear()

    # ------------------------------------------------------------------ speculative verify
    # A verify step decodes 32 token rows like a plain step, but several rows can belong to one request: its last
    # committed token (candidate 0) and up to SPEC_MAX_CANDIDATES-1 draft tokens at the following positions. Each
    # row carries a copy of its request's page-table row and its own position, so attention, RoPE and sampling are
    # the plain per-row operations. Only the K/V write is split by candidate index (decoder.decode_forward).
    # Every row is sampled with its request's own top-k/top-p/temperature and a seed of request-seed + position, the
    # same per-position seed a plain decode uses. With deterministic drafts, accepting while the sampled token
    # equals the draft is exactly speculative sampling (acceptance probability p(draft)), so verification is
    # lossless for sampled requests, not only for greedy ones.
    SPEC_MAX_CANDIDATES = int(os.environ.get("GEMMA_SPEC_MAX_CANDIDATES", "8"))

    def _spec_entry(self, state, groups):
        """Capture (once per cache state) the 32-row verify model trace and its sampling trace.

        groups: list of lists of page-table keys (layer index or layer kind) that share one table; the verify
        trace reads one [32, W] row table per group, refreshed per step from the host tables.
        """
        key = id(state)
        entry = self.spec_traces.get(key)
        cache_ids = tuple(id(t) for pair in state.kv for t in pair)
        if entry is not None:
            if entry["cache_ids"] != cache_ids or entry["groups"] != groups:
                raise ValueError("Speculative trace was captured for different cache tensors or table groups")
            return entry
        start = time.perf_counter()
        P = self.SPEC_MAX_CANDIDATES
        width = state.capacity // 128
        zeros32 = torch.zeros(32, dtype=torch.int32)
        entry = {
            "state": state,
            "cache_ids": cache_ids,
            "groups": groups,
            "tokens": self._upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32),
            "positions": self._upload(torch.full((32,), -1, dtype=torch.int32)),
            "rope_positions": self._upload(zeros32),
            "update_positions": [self._upload(torch.full((32,), -1, dtype=torch.int32)) for _ in range(P)],
            "group_tables": [self._upload(torch.zeros(32, width, dtype=torch.int32)) for _ in groups],
            "logits": self.model.tensor(torch.zeros(1, 1, 32, self.model.config.vocab_size), dim=3,
                                        dtype=ttnn.bfloat16),
        }
        tables = {k: entry["group_tables"][g] for g, keys in enumerate(groups) for k in keys}
        entry["tables"] = tables

        def model_step():
            out = self.model.decode_device(
                entry["tokens"], entry["positions"], page_tables=tables, kv_cache=state.kv, batch=32,
                rope_positions=entry["rope_positions"], update_positions=entry["update_positions"])
            ttnn.copy(out, entry["logits"])

        def sample_step():
            self.sampler(entry["logits"], tt_out_tok=entry["tokens"])

        # Warm with every row inactive (position -1: no cache write, no attention row), then capture.
        logger.info("Warming speculative verify trace")
        model_step()
        saved_seeds = ttnn.clone(self.sampler.seeds_tt_tensor)
        sample_step()
        ttnn.copy(saved_seeds, self.sampler.seeds_tt_tensor)
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(saved_seeds)
        programs_before = self.mesh.num_program_cache_entries()
        self.mesh.set_program_cache_misses_allowed(False)
        try:
            entry["model_trace"] = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            model_step()
            ttnn.end_trace_capture(self.mesh, entry["model_trace"], cq_id=0)
            entry["sampling_trace"] = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            sample_step()
            ttnn.end_trace_capture(self.mesh, entry["sampling_trace"], cq_id=0)
        finally:
            self.mesh.set_program_cache_misses_allowed(True)
        if self.mesh.num_program_cache_entries() != programs_before:
            raise RuntimeError("Native program cache grew during speculative trace capture")
        self.spec_traces[key] = entry
        logger.info("SPEC_TRACE_ENTRY_COST " + json.dumps({"seconds": time.perf_counter() - start, "P": P,
                                                             "groups": len(groups)}))
        return entry

    def spec_decode(self, state, *, tokens, positions, slots, candidates, host_tables, groups, sampling_params,
                    sample=True):
        """One synchronous verify step over 32 rows; returns the sampled token of every row (host int64 [32]).

        tokens/positions/slots/candidates: int tensors of length 32. Row r feeds tokens[r] at positions[r]
        (-1 = inactive row) with the page-table row of state slot slots[r]; candidates[r] is its candidate index
        (0 = the request's last committed token, j = its j-th draft). host_tables maps every key of `groups` to the
        current host page table [batch, W] (the same tables a plain decode would upload). sampling_params: per-row
        lists for configure_sampling (top_k/top_p/temperature/seed, length 32). sample=False returns host logits
        rows [32, V] instead (greedy/host checks only).
        """
        entry = self._spec_entry(state, groups)
        P = len(entry["update_positions"])
        positions = positions.to(torch.int32)
        candidates = candidates.to(torch.int32)
        live = positions >= 0
        if tuple(tokens.shape) != (32,) or tuple(positions.shape) != (32,):
            raise ValueError("Speculative verify takes exactly 32 rows")
        if bool((candidates[live] >= P).any()) or bool((candidates[live] < 0).any()):
            raise ValueError(f"Candidate index outside 0..{P - 1}")
        if bool((positions >= state.capacity).any()):
            raise ValueError("Verify position beyond the cache capacity")
        self._validate_tokens(tokens[live])
        rows = slots.to(torch.int64).clamp(0, state.batch - 1)
        ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        ids.reshape(-1)[:] = torch.where(live, tokens.to(torch.int32), torch.zeros_like(positions))
        self._refresh(ids, entry["tokens"], "token_refreshes")
        self._refresh(positions, entry["positions"], "position_refreshes")
        self._refresh(positions.clamp_min(0), entry["rope_positions"], "rope_refreshes")
        for j in range(P):
            upd = torch.where(live & (candidates == j), positions, torch.full_like(positions, -1))
            self._refresh(upd, entry["update_positions"][j], "position_refreshes")
        for g, keys in enumerate(groups):
            table = host_tables[keys[0]].to(torch.int32)
            self._refresh(table[rows].contiguous(), entry["group_tables"][g], "page_table_refreshes")
        if sample:
            self.configure_sampling(batch=32, **sampling_params)
        ttnn.execute_trace(self.mesh, entry["model_trace"], cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        if not sample:
            return self._host_logits(entry["logits"])[0, 0, :32]
        ttnn.execute_trace(self.mesh, entry["sampling_trace"], cq_id=0, blocking=False)
        self.counters["sampling_replays"] += 1
        return self._read_tokens(entry["tokens"], 32).reshape(-1).to(torch.int64)

    def close(self):
        self._serving_prefill_prepared_state = None
        self._serving_prefill_has_real_work = False
        prepared = self._prepared_prefill_state is not None
        self._prepared_prefill_state = None
        self._retire_traces()
        if prepared:
            self.mesh.set_program_cache_misses_allowed(True)
        self._entry_storage = None


def build_generator(mesh_device, *, checkpoint=None, layer_indices=None, max_seq_len=2048):
    model = Gemma4Model(mesh_device, checkpoint=checkpoint, layer_indices=layer_indices)
    return Gemma4Generator(model, max_seq_len=max_seq_len)

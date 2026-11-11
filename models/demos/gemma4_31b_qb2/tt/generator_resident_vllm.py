# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter for Gemma4 31B QB2 with ttnn prefill and the resident decode engine.

ttnn runs prefill into the vLLM-owned KV pools; the resident persistent-kernel engine
(`resident_s3.engine`, contract `bench/resident/M1_INTERFACE.md`) runs every decode step on
the same pools. No ttnn decode trace is ever captured.

KV aliasing: each pool id has one BF8 sliding K and V tensor [N', 4, 128, 256], N' = N + B.
The pool's full-attention layer reads ttnn.experimental.view(..., [2N', 1, 128, 512]) of
them, so full block b lives on full page 2b (the first half of sliding block b) and slot
j's full scratch page is 2(N + j). vLLM never hands one block id to two groups, so a page
is only ever read in the geometry that wrote it.

Never call ttnn.deallocate on a KV tensor or view: its default force=True frees the shared
root buffer under every layer of the pool. Teardown only drops references.
"""

import importlib
import os

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import CacheState, Gemma4Generator
from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM
from models.demos.gemma4_31b_qb2.tt.model import Gemma4Model

RAW_WEIGHT_CACHE = "/home/ttuser/experiments/gemma4/resident_cache_raw"
LOGIT_SOFTCAP = 30.0  # Gemma 4's final_logit_softcapping, the constant of the engine's own cap (engine._cap)


def cap_logits(raw):
    """Gemma's final logit soft cap, 30 tanh(x / 30), in fp32. Strictly monotonic at logit magnitudes, so distinct
    bf16 raw logits stay distinct and equal ones stay equal: the argmax, first index on ties, is the device argmax's
    (the hub keeps the larger bf16 key and, on equal keys, the lower vocab index: kernels/hub_dm.cpp `better`)."""
    return LOGIT_SOFTCAP * torch.tanh(raw.float() / LOGIT_SOFTCAP)
DEFAULT_L1_TOP = 1_330_000


def layer_groups(kinds):
    """vLLM's KV cache group of each layer, in vLLM's group order.

    vLLM 0.31 (`kv_cache_utils._get_kv_cache_groups_uniform_page_size`) buckets layers by
    kind in first-seen order, takes the smallest bucket as the group size and splits each
    bucket as layers[g::n]. For Gemma4 31B this is group i % 6, the full layers being 5.
    Serving re-checks it on every table update (`_group_sources`).
    """
    order = list(dict.fromkeys(kinds))
    members = {kind: [i for i, k in enumerate(kinds) if k == kind] for kind in order}
    size = min(len(m) for m in members.values())
    groups, first = [None] * len(kinds), 0
    for kind in order:
        count = len(members[kind]) // size
        if count * size != len(members[kind]):
            raise ValueError("Layer kinds do not split into equal vLLM groups")
        for rank, i in enumerate(members[kind]):
            groups[i] = first + rank % count
        first += count
    return groups


class _PrefillGenerator(Gemma4Generator):
    """The ttnn generator as a prefill engine: tables are validated but never uploaded whole.

    Serving prefill uploads its own per-row tables; the [batch, pages] device tables exist
    only for the ttnn decode trace, which this adapter never captures.
    """

    def _tables(self, state, tables):
        self._validate_cache(state, tables)
        return {}


class _Readback:
    """One decode step's host copies of the engine tokens and status words."""

    def __init__(self, tokens, status):
        self.tokens, self.status = tokens, status


class Gemma4ResidentForCausalLM(Gemma4ForCausalLM):
    model_capabilities = dict(Gemma4ForCausalLM.model_capabilities)
    decode_input_update_contract = 1
    ROWS = (1,)

    def __init__(self, generator, max_batch_size, max_seq_len, *, vllm_config=None):
        super().__init__(generator, max_batch_size, max_seq_len, vllm_config=vllm_config)
        self.engine = None
        self.groups = layer_groups([layer.kind for layer in generator.model.layers])
        self.blocks = None
        self.mode = None
        self.sampling_trace = None
        self.prefill_tokens = None
        self.positions = [-1] * max_batch_size
        # Set when the device feed no longer matches the host (an engine rebuild re-warms with inert rows).
        self.feed_stale = False
        self.l1_checks = []
        self.sampling_captures = []
        self.embedding_warm_s = None
        self.bucket_warm_s = None

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or max_batch_size != 1:
            raise ValueError("Resident Gemma4 QB2 (M1) serves TP4, DP1, one request")
        if not 128 <= max_seq_len <= 262144 or max_seq_len % 128:
            raise ValueError("Serving context must be 128..262144, aligned to 128-token pages")
        if hf_config.text_config.hidden_size != 5376 or hf_config.text_config.num_hidden_layers != 60:
            raise ValueError("Expected the Gemma4 31B checkpoint")
        torch.set_num_threads(8)
        # Prefill-only ttnn model; prefill padding with length-independent kernels (no JIT per prompt length).
        tile_pad = os.environ.get("GEMMA4_RESIDENT_TILE_PAD", "1") == "1"
        model = Gemma4Model(mesh_device, decode_weights=False, prefill_tile_pad=tile_pad)
        return cls(_PrefillGenerator(model, max_seq_len=max_seq_len), max_batch_size, max_seq_len)

    @classmethod
    def get_max_tokens_all_users(cls, *, max_num_seqs, max_model_len=262144, **kwargs):
        # The ttnn budget plus one lookahead page per request in each sliding group.
        tails = (Decoder.SLIDING_WINDOW_PAGES + 1) * Decoder.PAGE_SIZE * max_num_seqs
        return max_model_len + 5 * (tails + 2 * cls.MAX_PREFILL_TOKENS)

    # ------------------------------------------------------------------ KV pools and tables

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        self.cache = self.allocate_pools(per_layer_specs)
        self.engine = self._build_engine()
        return self.cache

    def allocate_pools(self, per_layer_specs, *, alias=True):
        """The vLLM KV pools as a CacheState. alias=False gives each full layer its own tensors
        (the ttnn layout), for equivalence tests only."""
        layers = self.generator.model.layers
        if len(per_layer_specs) != len(layers):
            raise ValueError("The allocator must supply one spec per layer")
        blocks = {shape[0] for shape, _, _ in per_layer_specs}
        if len(blocks) != 1:
            raise ValueError("Hybrid groups must share one block count")
        self.blocks = blocks.pop()
        pages = self.blocks + self.max_batch_size
        roots = {}
        for layer, (shape, dtype, pool_id) in zip(layers, per_layer_specs):
            if tuple(shape[1:]) != (layer.kv_heads, 128, layer.head_dim) or dtype != torch.bfloat16:
                raise ValueError(f"Unexpected allocator geometry or accounting dtype: {shape}, {dtype}")
            if layer.sliding and pool_id not in roots:
                roots[pool_id] = layer.allocate_cache(physical_pages=pages)
        kv, scratch, unaliased = [], {}, {}
        for i, (layer, (_, _, pool_id)) in enumerate(zip(layers, per_layer_specs)):
            if pool_id not in roots:
                raise ValueError("Every pool must hold a sliding layer for the full layer to alias")
            if layer.sliding:
                kv.append(roots[pool_id])
                scratch[i] = list(range(self.blocks, pages))
                continue
            if not alias:
                if pool_id not in unaliased:
                    unaliased[pool_id] = layer.allocate_cache(physical_pages=pages)
                kv.append(unaliased[pool_id])
                scratch[i] = list(range(self.blocks, pages))
                continue
            sliding = roots[pool_id][0]
            if layer.kv_heads * layer.head_dim * 2 != sliding.shape[1] * sliding.shape[3]:
                raise ValueError("A full page must hold exactly half a sliding block")
            view = [2 * pages, layer.kv_heads, 128, layer.head_dim]
            kv.append(tuple(ttnn.experimental.view(t, view) for t in roots[pool_id]))
            scratch[i] = [2 * p for p in range(self.blocks, pages)]
        tables = {
            i: torch.zeros(self.max_batch_size, self.max_seq_len // 128, dtype=torch.int32) for i in range(len(kv))
        }
        state = CacheState(kv, tables, self.max_batch_size, self.max_seq_len, scratch_pages=scratch, vllm_owned=True)
        # vLLM block b of a full layer is page b * full_page_scale of that layer's tensor.
        state.full_page_scale = 2 if alias else 1
        return state

    def _group_scratch(self):
        """scratch[g][slot] in group g's own page geometry."""
        full = {g for g, layer in zip(self.groups, self.generator.model.layers) if not layer.sliding}
        scale = getattr(self.cache, "full_page_scale", 2)
        return [
            [(scale if g in full else 1) * (self.blocks + j) for j in range(self.max_batch_size)]
            for g in range(max(self.groups) + 1)
        ]

    def _build_engine(self):
        # The engine package: resident_s3 (M1's, verified) unless GEMMA4_RESIDENT_ENGINE_PACKAGE names another with its
        # API (lead, 2026-11-11: the regression's stage 2 runs C at 262K on resident_m3 / m3_1, whose capacity is 262,144
        # against resident_s3's 32,768). The speculative adapter reads the same variable (default resident_m2).
        package = os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE", "resident_s3")
        resident = importlib.import_module(f"models.demos.gemma4_31b_qb2.{package}.engine")

        setup = self.generator.model.shared_setup
        cfg = resident.EngineConfig(
            weight_cache=os.environ.get("GRES_CACHE", RAW_WEIGHT_CACHE),
            rows=self.ROWS,
            max_users=self.max_batch_size,
            capacity=self.max_seq_len,
            tap_layers=(),
            l1_top=int(os.environ.get("GRES_L1_TOP", DEFAULT_L1_TOP)),
            logits=True,
            embedding="resident",
        )
        pools = resident.KVPools(
            kv=list(self.cache.kv),
            layer_group=list(self.groups),
            scratch=self._group_scratch(),
            rope={
                kind: (setup[("rope", kind)]["decode_cos"], setup[("rope", kind)]["decode_sin"])
                for kind in ("sliding_attention", "full_attention")
            },
        )
        stub = os.environ.get("GEMMA4_RESIDENT_ENGINE", "resident") == "stub"
        logger.info("Building the resident {} engine from {}", "stub" if stub else "decode", package)
        return (resident.StubEngine if stub else resident.ResidentEngine)(self.mesh, cfg, pools)

    def _tables(self, cache, page_tables_per_layer, *, slots=None):
        """ttnn prefill tables: vLLM block b of a full layer is its view's page 2b (null 0 stays 0)."""
        result = super()._tables(cache, page_tables_per_layer, slots=slots)
        scale = getattr(cache, "full_page_scale", 2)
        scaled = {}
        for i, layer in enumerate(self.generator.model.layers):
            if not layer.sliding and scale != 1:
                key = id(result[i])
                if key not in scaled:
                    scaled[key] = result[i] * scale
                result[i] = scaled[key]
        return result

    def _group_sources(self, tables):
        """One translated table per vLLM group; the layers of a group must agree."""
        sources = {}
        for i, group in enumerate(self.groups):
            if group not in sources:
                sources[group] = tables[i]
            elif tables[i] is not sources[group] and not torch.equal(tables[i], sources[group]):
                raise ValueError(f"Layer {i} does not share the page table of vLLM group {group}")
        return [sources[g] for g in range(len(sources))]

    def _resident_tables(self, tables):
        """Engine tables [users, groups, pages] from translated ttnn tables.

        The runner zeroes every column past a row's live block count, and vLLM's null block
        is 0. So a full-group 0 is unallocated, and sliding zeros after a row's last live
        block are unallocated; both go to the slot's scratch page, which is write-only.
        Sliding zeros before it are evicted blocks outside the window and stay 0.
        """
        layers = self.generator.model.layers
        full = {g for g, layer in zip(self.groups, layers) if not layer.sliding}
        scratch = torch.tensor(self._group_scratch(), dtype=torch.int32)
        result = []
        for g, table in enumerate(self._group_sources(tables)):
            table = table.to(torch.int32).clamp_min(0)
            live = table != 0
            if g not in full:
                columns = torch.arange(1, table.shape[1] + 1)
                last = (live * columns).amax(dim=1, keepdim=True)
                live = columns <= last
            result.append(torch.where(live, table, scratch[g][:, None]))
        return torch.stack(result, dim=1).contiguous()

    # ------------------------------------------------------------------ prefill

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        *,
        sampling_params=None,
        page_tables_per_layer=None,
        start_pos=None,
        empty_slots=None,
        **kwargs,
    ):
        starts = [0] * len(prompt_lens) if start_pos is None else [int(x) for x in start_pos]
        lengths = [int(x) for x in prompt_lens]
        slots = list(range(len(lengths))) if empty_slots is None else list(empty_slots)
        tables = self._tables(kv_cache, page_tables_per_layer, slots=slots)
        full_layer = next(i for i, layer in enumerate(self.generator.model.layers) if not layer.sliding)
        keys = [int(tables[full_layer][slot, 0]) for slot in slots]
        self.prefill_slot_keys = {s: k for s, k in self.prefill_slot_keys.items() if s not in slots and k not in keys}
        self.prefill_slot_keys.update(zip(slots, keys))
        # The generator's own new-shape check, ahead of its persistent allocations.
        novel = (kv_cache.batch, kv_cache.capacity, tuple(lengths), tuple(starts), False) not in (
            self.generator.prefill_shapes
        )
        if novel:
            self._release_sampling_trace()
        outputs = self.generator.prefill_forward(
            tokens,
            page_table=tables,
            kv_cache=kv_cache,
            prompt_lens=lengths,
            slots=slots,
            return_device=sampling_params is not None,
            start_positions=starts,
            history_keys=keys,
        )
        if novel:
            self._l1_guard(f"prefill {tuple(lengths)} from {tuple(starts)}")
        if sampling_params is None:
            return outputs
        return self._sample_prefill(outputs, sampling_params, lengths)

    def _sample_prefill(self, outputs, sampling_params, lengths):
        """Sample the prefill rows eagerly into private tokens: no trace, no write to the engine feed."""
        batch = len(outputs)
        params = self._sampling(sampling_params, batch, [n - 1 for n in lengths])
        self.generator.configure_sampling(batch=batch, **params)
        merged = outputs[0] if batch == 1 else ttnn.concat(outputs, dim=2)
        if 1 < batch < 32:
            merged = ttnn.pad(merged, [(0, 0), (0, 0), (0, 32 - batch), (0, 0)], value=0.0)
        shape = ttnn.Shape([1, 1, 32, self.generator.model.config.vocab_size // 4])
        logits = ttnn.reshape(merged, shape, shape)
        outputs.clear()
        self.generator._sample_step({"logits": logits, "tokens": self.prefill_tokens})
        return self.generator._read_tokens(self.prefill_tokens, batch).reshape(-1, 1)

    # ------------------------------------------------------------------ decode

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        *,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        page_tables_per_layer=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        **kwargs,
    ):
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1 requires explicit reload flags")
        if not enable_trace:
            raise ValueError("Resident decode is always a replay")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Resetting sampling state requires current tokens and positions")
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        if self.feed_stale and not reload_inputs:
            raise ValueError("The resident feed was reset (engine rebuild); the step needs current inputs")
        batch = self.max_batch_size
        if reload_inputs or reload_page_table:
            tables = self._tables(kv_cache, page_tables_per_layer)
            self.engine.write_tables(self._resident_tables(tables))
        if reload_inputs:
            positions = [int(p) for p in start_pos.reshape(-1)[:batch]]
            full_layer = next(i for i, layer in enumerate(self.generator.model.layers) if not layer.sliding)
            done = {int(tables[full_layer][row, 0]) for row, position in enumerate(positions) if position >= 0}
            for key in done:
                kv_cache.prefill_history.pop(key, None)
            self.prefill_slot_keys = {slot: key for slot, key in self.prefill_slot_keys.items() if key not in done}
            self.engine.write_rows(range(batch), positions, [int(t) for t in tokens.reshape(-1)[:batch]])
            self.positions = positions
            self.feed_stale = False
        if any(p >= self.max_seq_len for p in self.positions):
            raise ValueError("No decode steps remain within the cache capacity")
        host = sampling_params is None
        greedy = not host and all(t == 0 for t in self._rows(sampling_params.temperature, batch))
        if not host and (reload_sampling_params or reset_sampling_state):
            params = self._sampling(sampling_params, batch, start_pos, reset_seed=reset_sampling_state)
            self.generator.configure_sampling(batch=batch, **params)
        mode = ("none", True) if host else ("argmax", False) if greedy else ("external", True)
        if not host and not greedy and self.sampling_trace is None:
            # Before the launch, with nothing in flight. After it, the capture waited for the step (11.2 ms
            # median in M1), so the step returned late and vLLM emitted the previous step's token (a novel
            # prefill's first token) together with this one.
            self._capture_sampling()
            self._guard_after_capture()
        if mode != self.mode:
            self.engine.set_mode(feed=mode[0], write_logits=mode[1])
            self.mode = mode
        self.engine.launch(self.ROWS[0])
        if not host:
            self.positions = [p + 1 if p >= 0 else p for p in self.positions]
        if host:
            logits = self._host_logits(batch).unsqueeze(1)
            self._check_status(ttnn.get_device_tensors(self.engine.status)[0].cpu())
            return logits
        if not greedy:
            self._replay_sampling()
        output = self.engine.tokens
        return self.process_decode_output_host(output, is_tokens=True) if read_from_device else output

    def _host_logits(self, rows):
        """The first `rows` logits rows as float [rows, vocab] for the plugin's host sampler: the engine's raw logits
        (`logits_raw`, the values its argmax scans) soft-capped here in fp32 (cap_logits). A host-sampled greedy
        request (logprobs, penalties) then picks the device argmax's token, and its logprobs are those of the fp32-capped
        logits. The engine's own capped `logits` are bf16: the cap compresses the top logits onto a 0.125 grid, where
        values that differ raw tie, and the plugin's greedy took the first of a tie the device argmax did not have
        (M2 review, 2026-11-11: logprobs-requested greedy left device greedy at prose 108, math 90). An engine without
        raw logits (the stub) keeps `logits`.
        """
        raw = getattr(self.engine, "logits_raw", None)
        out = self._read_logit_rows(self.engine.logits if raw is None else raw, rows)
        return out if raw is None else cap_logits(out)

    def _read_logit_rows(self, logits, rows):
        """Rows [0, rows) of a device logits tensor as float [rows, vocab].

        The rows are untilized and cut on the device, so only rows x vocab bf16 crosses PCIe. Reading the
        whole 32-row tile and untilizing it on the host cost 8.4 ms per step against 0.16 ms here
        (bench/resident/m1_serving/logits_readback_probe.json). The row-major copy is a private
        temporary, freed when its last reference drops before this returns (N11: no deallocate in
        this file), so a later sampling-trace replay that reuses its addresses overwrites nothing live.
        """
        width = ttnn.get_device_tensors(logits)[0].shape[-1]
        part = ttnn.untilize_with_unpadding(logits, [0, 0, rows - 1, width - 1])
        out = ttnn.to_torch(part, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3)).float()[0, 0]
        del part
        return out

    @staticmethod
    def _rows(value, batch):
        value = value.tolist() if isinstance(value, torch.Tensor) else value
        return list(value)[:batch] if isinstance(value, (list, tuple)) else [value] * batch

    def read_decode_output(self, output, async_read=False, **kwargs):
        if isinstance(output, torch.Tensor):
            return (output, []) if async_read else output
        # Tokens and status are replicated: one rank each, in one ordered readback.
        tokens = ttnn.get_device_tensors(output)[0].cpu(blocking=not async_read)
        status = ttnn.get_device_tensors(self.engine.status)[0].cpu(blocking=not async_read)
        self.generator.counters["token_readbacks"] += 1
        host = _Readback(tokens, status)
        return (host, [ttnn.record_event(self.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, output, is_tokens=True, **kwargs):
        if not is_tokens:
            return output.reshape(self.max_batch_size, 1, -1)
        if not isinstance(output, _Readback):
            output = self.read_decode_output(output)
        self._check_status(output.status)
        return ttnn.to_torch(output.tokens).reshape(-1)[: self.max_batch_size].long().reshape(-1, 1)

    def _check_status(self, status):
        """Contract §5: word 1 holds sticky error flags. Any flag fails the step loudly."""
        words = ttnn.to_torch(status).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        if int(words[1]) == 0:
            return
        signed = [w - (1 << 32) if w >= 1 << 31 else w for w in words.tolist()]
        row = signed[2]
        slot = row if 0 <= row < self.max_batch_size else None  # plain decode: row r is slot r
        message = (
            f"Resident engine flagged a decode input: flags 0x{words[1]:x} (launches {signed[0]}), row {row}, "
            f"slot {slot}, position {signed[3]}, page {signed[4]}, entry {signed[5]}; "
            f"host positions {self.positions}; status words {signed}"
        )
        logger.error(message)
        self.engine.clear_status()
        # The device fed nothing valid for the flagged row: rebuild inputs from the host.
        self.mode = None
        raise RuntimeError(message)

    # ------------------------------------------------------------------ sampling trace

    def _replay_sampling(self):
        if self.sampling_trace is None:
            raise RuntimeError("The sampling trace is captured before the launch of a sampled step")
        ttnn.execute_trace(self.mesh, self.sampling_trace, cq_id=0, blocking=False)
        self.generator.counters["sampling_replays"] += 1

    def _guard_after_capture(self):
        """The lazy capture can allocate like a novel prefill shape, so the L1 guard runs after it too.

        It runs before this step's launch. A rebuild's warmup zeroes the engine's tokens, positions and
        status, so the last step's status is checked first and this step's feed (its input tokens at the
        current positions) is written back: the step launches as if nothing happened.
        """
        for _ in range(2):
            kept = {}

            def keep_feed():
                self._check_status(ttnn.get_device_tensors(self.engine.status)[0].cpu())
                kept["tokens"] = self.engine.read_tokens(self.max_batch_size).tolist()

            self._l1_guard("sampling trace capture", before_rebuild=keep_feed)
            if kept:
                positions = [p if p < self.max_seq_len else -1 for p in self.positions]
                tokens = [t if p >= 0 else 0 for t, p in zip(kept["tokens"], positions)]
                self.engine.write_rows(range(self.max_batch_size), positions, tokens)
                self.feed_stale = False
            if self.sampling_trace is not None:
                return
            # The rebuild released the trace just captured: capture it again, after the rebuild's allocations.
            self._capture_sampling()
        raise RuntimeError("The sampling trace was released by an engine rebuild twice in a row")

    def _capture_sampling(self):
        """sampler(engine.logits) -> engine.tokens, then seeds + 1, as one trace.

        The eager warm-up compiles and allocates first. It runs before this step's launch, so it must not
        consume the caller's seeds or overwrite the feed in engine.tokens: both are restored after it.
        """
        import time

        start = time.perf_counter()
        entry = {"logits": self.engine.logits, "tokens": self.engine.tokens}
        seeds = self.generator.sampler.seeds_tt_tensor
        saved = (ttnn.clone(seeds), ttnn.clone(self.engine.tokens))
        self.generator._sample_step(entry)
        ttnn.copy(saved[0], seeds)
        ttnn.copy(saved[1], self.engine.tokens)
        del saved
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.generator._sample_step(entry)
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self.sampling_trace = trace
        self.sampling_captures.append(time.perf_counter() - start)
        logger.info("Sampling trace captured in {:.1f} ms (host)", self.sampling_captures[-1] * 1000)

    def _release_sampling_trace(self):
        if self.sampling_trace is not None:
            ttnn.synchronize_device(self.mesh)
            ttnn.release_trace(self.mesh, self.sampling_trace)
            self.sampling_trace = None

    # ------------------------------------------------------------------ L1 guard

    def _l1_floor(self):
        """Lowest address of any allocated L1 buffer, or None. Block addresses are allocator-relative."""
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1)
        allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
        if not allocated:
            return None
        return ttnn.get_allocator_base_address(self.mesh, ttnn.BufferType.L1) + min(allocated)

    def _l1_guard(self, reason, before_rebuild=None):
        """No persistent ttnn L1 buffer may sit inside the engine's CB range: replay skips that check."""
        floor, top = self._l1_floor(), self.engine.cb_top()
        self.l1_checks.append((reason, floor, top))
        logger.info("L1 guard after {}: lowest ttnn L1 buffer {} (absolute), engine CB top {}", reason, floor, top)
        if floor is None or floor >= top:
            return
        logger.warning("L1 buffer at {} below the resident CB top {} after {}: rebuilding", floor, top, reason)
        if before_rebuild is not None:
            before_rebuild()
        self.engine.rebuild(l1_top=floor)
        # rebuild re-warms: rows inert, mode "none", tables kept. The next decode must reload.
        self.mode = None
        self.feed_stale = True
        # The rebuild allocates after the sampling trace was captured, and a replay can overwrite what was allocated
        # after its capture: release it; it is captured again before the next sampled launch.
        self._release_sampling_trace()
        if self.engine.cb_top() > floor:
            raise RuntimeError(f"Resident engine cannot fit below the ttnn L1 floor {floor}")

    # ------------------------------------------------------------------ lifecycle

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        """Build the prefill programs a first request would otherwise compile on its TTFT, before any trace.

        - ttnn.embedding: one kernel per 32-row prompt bucket up to the chunk limit.
        - One prefill per 128-row bucket (L = 128k - 1, so the tile-pad select runs too): every layer's
          programs for that chunk size. GEMMA4_RESIDENT_WARM_BUCKETS=0 skips it.
        A kernel compiles once per machine (disk cache); program objects once per process. The writes land in
        free pool blocks before vLLM has allocated any. The runner warms prefill before decode.
        """
        if self.embedding_warm_s is not None:
            return
        import time

        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        limit = min(self.MAX_PREFILL_TOKENS, self.max_seq_len)
        start = time.perf_counter()
        for rows in range(32, limit + 1, 32):
            ids = self.generator._upload(torch.zeros(1, rows, dtype=torch.int32), ttnn.uint32)
            self.generator.model.embed(ids, sequence_length=rows)
        ttnn.synchronize_device(self.mesh)
        self.embedding_warm_s = time.perf_counter() - start
        logger.info("Embedding programs for {} prompt buckets built in {:.1f} s", limit // 32, self.embedding_warm_s)
        if os.environ.get("GEMMA4_RESIDENT_WARM_BUCKETS", "1") != "1":
            return
        start = time.perf_counter()
        columns, groups = self.max_seq_len // 128, max(self.groups) + 1
        if groups * (limit // 128) >= self.blocks:
            raise ValueError("The pool is too small for the prefill bucket warmup")
        for pages in range(1, limit // 128 + 1):
            length = pages * 128 - 1
            # Group g writes blocks 1 + g * (limit / 128) + c: distinct, nonzero and below N.
            source = []
            for g in range(groups):
                table = torch.zeros(1, columns, dtype=torch.int32)
                table[0, :pages] = 1 + g * (limit // 128) + torch.arange(pages, dtype=torch.int32)
                source.append(table)
            self.prefill_forward(
                torch.zeros(1, length, dtype=torch.int32), None, kv_cache, [length],
                page_tables_per_layer=[source[g] for g in self.groups], start_pos=[0], empty_slots=[0],
            )
            kv_cache.prefill_history.clear()
            self.prefill_slot_keys.clear()
        ttnn.synchronize_device(self.mesh)
        self.bucket_warm_s = time.perf_counter() - start
        logger.info("Prefill programs for {} chunk buckets built in {:.1f} s", limit // 128, self.bucket_warm_s)

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        if self.prefill_tokens is not None:
            return
        self.prefill_tokens = self.generator._upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        batch = self.max_batch_size
        self.engine.write_tables(self._resident_tables(self._tables(kv_cache, list(kv_cache.page_tables.values()))))
        self.engine.write_rows(range(batch), [-1] * batch, [0] * batch)
        self.generator.configure_sampling(batch=self.max_batch_size, temperature=0)
        self.engine.warmup()
        self._capture_sampling()
        self._l1_guard("warmup")

    def note_state_slots_moved(self, moves):
        super().note_state_slots_moved(moves)
        if moves and self.engine is not None:
            self.engine.permute_slots(dict(moves))

    def release_request(self, slot):
        super().release_request(slot)
        if self.engine is not None:
            self.engine.release_slot(slot)
            self.positions[slot] = -1

    def release_persistent_capture(self):
        self._release_sampling_trace()
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        super().release_persistent_capture()

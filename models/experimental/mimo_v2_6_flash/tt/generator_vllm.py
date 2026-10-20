# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM serving adapter for MiMo-V2.6-Flash on a Blackhole Galaxy (vllm-tt-plugin, tenstorrent/vllm-tt-plugin#216).

Data parallelism: ``--data-parallel-size 4`` runs as four in-process TT lanes
(``model_capabilities["folds_dp_into_lanes"]``): one engine drives MiMoGalaxyGenerator's four TP=8 replicas, one
per row of the 4x8 mesh. Lane ``l`` owns the plugin's slots [l * per_lane, (l + 1) * per_lane), which are decode
slots 0 .. per_lane - 1 of replica ``l``. Each lane has its own vLLM scheduler and block pool, so block ids are
lane-local and each replica has its own KV storage.

KV cache: vLLM owns the blocks. ``get_kv_cache_spec`` reports a FullAttentionSpec for the global layers and a
SlidingWindowSpec (window 128) for the SWA layers, both K 192 / V 128 at block size 64. vLLM's hybrid KV manager
splits them into KV cache groups of equal size (for 9 global and 39 SWA layers: one global group and five SWA
groups of up to 9 layers), every group with its own block table and all of them drawing block ids from one pool.
The model gets one page table per group (TtMiMoModel.setup_vllm_kv_cache). A SWA group's table is indexed by
absolute position with the blocks no window can read set to vLLM's null block 0, the layout the model's own
kv.SwaBlockTable keeps.

Storage follows vLLM's sharing rule: the i-th layer of every group shares one buffer (``tensor_idx``), which is
safe because a block id belongs to one group at a time. On a chip every layer's K cache is
[num_blocks, 1, 64, 192] and every layer's V cache [num_blocks, 1, 64, 128] (attention.py keeps V's real dims on
the SWA layers too), each shared that way: 9 K and 9 V buffers for 9 global and 39 SWA layers. The global layers
keep both copies of their KV head on a chip pair (setup_vllm_kv_cache): a block can be shared by requests in slots
of either parity.

Sampling: prefill returns last-token logits [B, 1, vocab] for the host to sample. Decode replays one traced step
per replica (captured in ``warmup_model_decode``) that also computes each slot's greedy token and writes it, with
the slot's position plus one, into the trace's own inputs (TtMiMoModel._feed_back). With ``sample_on_device_mode``
the plugin samples greedy rows on device (``max_device_top_k`` 1: a row with temperature > 0 and any other top_k,
penalties, logprobs or structured output send the whole batch to host sampling, which reads the logit rows) and
overlaps the steps (``decode_input_update_contract`` 1, ``supports_async_decode``): a steady step copies nothing
(the trace reads its RoPE cos / sin from device tables at its own positions; past 128K context it copies those of
the adapter's own position counter), and its tokens are read back without blocking. Prefill runs
eagerly in chunks of up to 4096 tokens, between the replays. A program first compiled after the capture would keep buffers the replays overwrite, so
``warmup_model_prefill`` compiles every prefill program a served chunk can run first: they key only on the chunk's
padded length and the alignment of its start (attention.py), a bounded set.
"""

import math
import os
import time

import numpy as np
import torch
from loguru import logger

import ttnn

from ..reference.checkpoint import MiMoCheckpoint, default_model_path
from ..reference.config import MiMoConfig
from .generator import MiMoGalaxyGenerator
from .model import prefill_padded_len, prefill_warmup_batches, prefill_warmup_chunks

TP = 8
BLOCK_SIZE = 64
# Per-chip bytes per token of the vLLM KV layout above for MiMo-V2.6-Flash (9 global, 39 SWA layers, so 9 shared K
# buffers and 9 shared V buffers), BFP8: (9 * 192 + 9 * 128) values * 1088 / 1024 bytes = 3060.
KV_BYTES_PER_TOKEN = (9 * 192 + 9 * 128) * 1088 / 1024
# Per-chip DRAM given to the KV pool when MIMO_KV_GB_PER_CHIP is unset (weights take 23.7 of 34 GB per chip).
DEFAULT_KV_GB_PER_CHIP = 6.0
# Blocks vLLM's sliding-window groups can hold per request while a prefill chunk of up to 4096 tokens is in flight
# (SlidingWindowSpec.max_admission_blocks_per_request), in tokens: 5 SWA groups x (127 + 4096 tokens + 1 block).
SWA_TOKENS_PER_USER = 5 * (math.ceil((127 + 4096) / BLOCK_SIZE) + 1) * BLOCK_SIZE
# MIMO_VLLM_TIMING=1 logs each prefill's host enqueue and total time, and decode's host time every 128 steps.
TIMING = os.environ.get("MIMO_VLLM_TIMING") == "1"
# MIMO_PREFILL_BATCH=0: prefill every row on its own (TtMiMoModel.prefill_enqueue_batch is not used).
PREFILL_BATCH = os.environ.get("MIMO_PREFILL_BATCH", "1") == "1"
# MIMO_PREFILL_TRACE=0: run prefill batches eagerly instead of replaying their traces.
PREFILL_TRACE = os.environ.get("MIMO_PREFILL_TRACE", "1") == "1"
# Decode reads its RoPE cos / sin from device tables (TtMiMoModel.setup_decode_rope_tables) up to this context
# (1M would cost 1.6 GB per chip); MIMO_DEVICE_ROPE=0 uploads them every step instead.
DEVICE_ROPE = os.environ.get("MIMO_DEVICE_ROPE", "1") == "1"
DEVICE_ROPE_MAX_LEN = 131072


def _model_path(hf_config):
    path = getattr(hf_config, "_name_or_path", None)
    if path and os.path.exists(os.path.join(path, "config.json")):
        return path
    return default_model_path()


def _cache_dir(model_path):
    """The converted TTNN weights, as the tests keep them (tests/common.py cache_dir("model_bfp4experts"))."""
    root = os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path)), "model_bfp4experts")


class MiMoV2ForCausalLM:
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_prefix_caching": True,
        "folds_dp_into_lanes": True,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D_RING},
        # Device sampling is the decode trace's greedy token (module docstring).
        "supports_sample_on_device": True,
        "supports_async_decode": True,
        "max_device_top_k": 1,
        "supports_device_penalties": False,
        "supports_device_logprobs": False,
    }
    # The four reload commands of the plugin's decode reload contract (docs/DECODE_RELOAD_CONTRACT.md).
    decode_input_update_contract = 1
    # Emits SlidingWindowSpec groups: the plugin's KV budget adds its sliding-window headroom.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True
    # (vllm_config, specs) recorded by get_kv_cache_spec, which the plugin calls in the worker process that later
    # allocates the KV cache. allocate_kv_cache_per_layer groups the specs with vLLM's own function, as the engine
    # did: vLLM resolves the KV cache layout the grouping reads only after collecting the specs.
    _kv_cache_spec_source = None
    # vLLM's KV cache group of each layer, when set directly (tests driving the adapter without vLLM).
    _layer_groups = None
    # Tokens vLLM schedules per step (--max-num-batched-tokens), which bounds a prefill chunk: None reads it from the
    # vLLM config get_kv_cache_spec recorded (set directly when driving the adapter without vLLM).
    prefill_token_budget = None

    def __init__(self, generator, max_seq_len):
        self.gen = generator
        self.models = generator.models
        self.dp = generator.dp
        self.per_lane = generator.users_per_replica
        self.max_seq_len = max_seq_len
        self.already_warmed_up_prefill = False
        self.prefill_programs_compiled = False
        self.layer_groups = None
        self.kv_caches = None
        self.decode_recaptures = 0  # decode traces captured again after a prefill compiled new programs
        # Per lane: int32 [max_batch] positions the next decode step reads (-1: idle slot), set by a reloading step
        # and advanced with every step, as the trace advances its own (a steady step reads no host positions).
        self.decode_positions = [None] * self.dp

    # ------------------------------------------------------------------ construction and sizing
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        if optimizations not in (None, "performance"):
            raise ValueError(f"MiMo-V2.6-Flash serves one datatype policy; optimizations={optimizations!r}")
        dp = tt_data_parallel
        num_devices = mesh_device.get_num_devices()
        if num_devices != TP * dp:
            raise ValueError(
                f"MiMo runs one TP={TP} replica per lane: {dp} lanes need {TP * dp} chips, got {num_devices}"
            )
        if max_batch_size % dp:
            raise ValueError(f"max_batch_size {max_batch_size} does not split over {dp} lanes")
        if max_seq_len % BLOCK_SIZE:
            raise ValueError(f"max_model_len {max_seq_len} must be a multiple of the block size {BLOCK_SIZE}")
        path = _model_path(hf_config)
        cfg = MiMoConfig.from_model_path(path)
        num_layers = hf_config.num_hidden_layers
        if num_layers > cfg.num_hidden_layers:
            raise ValueError(f"num_hidden_layers {num_layers} > the checkpoint's {cfg.num_hidden_layers}")
        logger.info(f"MiMo vLLM: {dp} replicas x {max_batch_size // dp} users, {num_layers} layers, {path}")
        generator = MiMoGalaxyGenerator(
            mesh_device,
            cfg,
            MiMoCheckpoint(path, cfg),
            _cache_dir(path),
            dp=dp,
            users_per_replica=max_batch_size // dp,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            block_size=BLOCK_SIZE,
            setup_kv=False,
        )
        return cls(generator, max_seq_len)

    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=TP, tt_data_parallel=1, **kwargs):
        """KV tokens of one lane's pool: enough for ``max_num_seqs`` requests of ``max_model_len`` (plus what vLLM's
        SWA groups hold per request), capped by MIMO_KV_GB_PER_CHIP of DRAM per chip."""
        budget = float(os.environ.get("MIMO_KV_GB_PER_CHIP", DEFAULT_KV_GB_PER_CHIP)) * 1e9
        capacity = int(budget // KV_BYTES_PER_TOKEN)
        max_model_len, max_num_seqs = kwargs.get("max_model_len"), kwargs.get("max_num_seqs")
        if not max_model_len or not max_num_seqs:
            return capacity
        return min(capacity, max_num_seqs * (max_model_len + SWA_TOKENS_PER_USER))

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        hf = vllm_config.model_config.hf_config
        block_size = vllm_config.cache_config.block_size
        if block_size != BLOCK_SIZE:
            raise ValueError(
                f"MiMo's paged caches use {BLOCK_SIZE}-token blocks; launch with --block-size {BLOCK_SIZE}"
            )
        full = dict(num_kv_heads=hf.num_key_value_heads, head_size=hf.head_dim, head_size_v=hf.v_head_dim)
        swa = dict(num_kv_heads=hf.swa_num_key_value_heads, head_size=hf.swa_head_dim, head_size_v=hf.swa_v_head_dim)
        dtype = torch.bfloat16
        # vLLM's hybrid manager needs one page size for every group; pad the global layers' page to the SWA page.
        page = max(block_size * d["num_kv_heads"] * (d["head_size"] + d["head_size_v"]) * 2 for d in (full, swa))
        window = getattr(hf, "sliding_window_size", None) or hf.sliding_window
        specs = {}
        for i in range(hf.num_hidden_layers):
            name = f"model.layers.{i}.self_attn"
            if hf.hybrid_layer_pattern[i] == 1:
                specs[name] = SlidingWindowSpec(
                    block_size=block_size, dtype=dtype, sliding_window=window, page_size_padded=page, **swa
                )
            else:
                specs[name] = FullAttentionSpec(block_size=block_size, dtype=dtype, page_size_padded=page, **full)
        cls._kv_cache_spec_source = (vllm_config, specs)
        return specs

    @classmethod
    def _vllm_layer_groups(cls):
        """vLLM's KV cache group of each layer, from the specs get_kv_cache_spec returned."""
        if cls._layer_groups is not None:
            return cls._layer_groups
        if cls._kv_cache_spec_source is None:
            raise RuntimeError("allocate_kv_cache_per_layer before get_kv_cache_spec: no KV cache groups")
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        vllm_config, specs = cls._kv_cache_spec_source
        groups = [None] * len(specs)
        for g, group in enumerate(get_kv_cache_groups(vllm_config, dict(specs))):
            for name in group.layer_names:
                groups[int(name.split(".")[2])] = g
        if None in groups:
            raise RuntimeError(f"layers without a KV cache group: {groups}")
        logger.info(f"MiMo vLLM KV cache groups per layer: {groups}")
        return groups

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """per_layer_specs: (shape, dtype, tensor_idx) per layer, shape[0] the lane's block count. Returns, per lane,
        [K, V] per layer, with buffers shared as in the module docstring."""
        groups = self._vllm_layer_groups()
        if len(groups) != len(per_layer_specs):
            raise RuntimeError(f"{len(per_layer_specs)} layer KV specs but {len(groups)} layer groups")
        num_blocks, block_size = per_layer_specs[0][0][0], per_layer_specs[0][0][2]
        assert block_size == BLOCK_SIZE and all(s[0][0] == num_blocks for s in per_layer_specs)
        # Layers sharing a buffer must come from different groups: their block ids never overlap.
        seen = set()
        for i, (_, _, t) in enumerate(per_layer_specs):
            assert (groups[i], t) not in seen, f"layer {i}: two layers of group {groups[i]} share buffer {t}"
            seen.add((groups[i], t))
        table_width = min(math.ceil(self.max_seq_len / block_size), num_blocks)
        self.layer_groups = groups
        self.kv_caches = []
        for model in self.models:
            buffers, caches = {}, []
            for layer, (_, _, t) in zip(model.layers, per_layer_specs):
                attn = layer.attn
                caches.append(
                    [
                        self._buffer(model, buffers, ("k", t, attn.head_dim), num_blocks, attn.head_dim),
                        self._buffer(model, buffers, ("v", t, attn.v_head_dim), num_blocks, attn.v_head_dim),
                    ]
                )
            model.setup_vllm_kv_cache(self.per_lane, caches, groups, table_width)
            if DEVICE_ROPE and self.max_seq_len <= DEVICE_ROPE_MAX_LEN:
                model.setup_decode_rope_tables()
            self.kv_caches.append(caches)
        logger.info(
            f"MiMo vLLM KV: {num_blocks} blocks of {block_size} per lane, {len(buffers)} buffers per replica, "
            f"page tables {table_width} wide for groups {sorted(set(groups))}"
        )
        return self.kv_caches

    @staticmethod
    def _buffer(model, buffers, key, num_blocks, dim):
        if key not in buffers:
            cache = ttnn.allocate_tensor_on_device(
                ttnn.Shape((num_blocks, 1, BLOCK_SIZE, dim)),
                model.kv_dtype,
                ttnn.TILE_LAYOUT,
                model.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill(cache, 0.0, output_tensor=cache)
            buffers[key] = cache
        return buffers[key]

    # ------------------------------------------------------------------ page tables
    def _group_rows(self, tables, row):
        """{group: vLLM block table row ``row``}, one per KV cache group (the layers of a group share a table)."""
        out = {}
        for i, g in enumerate(self.layer_groups):
            if g not in out:
                out[g] = torch.as_tensor(tables[i][row])
        return out

    def _layer_tables(self, page_table, page_tables_per_layer):
        """One block table per layer. The plugin sends ``page_tables_per_layer`` when vLLM made several KV cache
        groups; with one group, ``page_table`` serves every layer."""
        if page_tables_per_layer is None and len(set(self.layer_groups)) == 1:
            return [page_table] * len(self.layer_groups)
        if page_tables_per_layer is None or len(page_tables_per_layer) != len(self.layer_groups):
            raise ValueError(
                f"MiMo needs a page table per attention layer ({len(self.layer_groups)}); got "
                f"{None if page_tables_per_layer is None else len(page_tables_per_layer)}"
            )
        return page_tables_per_layer

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        tokens,
        start_pos,
        prompt_lens,
        page_table,
        kv_cache,
        enable_trace=False,
        empty_slots=None,
        page_tables_per_layer=None,
        sampling_params=None,
        **kwargs,
    ):
        """Row ``i`` computes tokens[i, start_pos[i]:prompt_lens[i]] for the request in slot ``empty_slots[i]``.
        Returns its last-token logits, float32 [B, 1, vocab] (host sampling)."""
        if sampling_params is not None:
            raise ValueError("MiMo samples on the host; got device sampling_params")
        tables = self._layer_tables(page_table, page_tables_per_layer)
        rows = tokens.shape[0]
        starts = np.asarray(start_pos).reshape(-1).astype(np.int64)
        ends = np.asarray(prompt_lens).reshape(-1).astype(np.int64)
        slots = list(range(rows)) if empty_slots is None else [int(s) for s in empty_slots]
        placed = []
        table_rows = [{} for _ in range(self.dp)]
        for i in range(rows):
            lane, slot = divmod(slots[i], self.per_lane)
            # vLLM starts chunks and prefix hits on block boundaries; a chunk that does not is moved back to one and
            # recomputes the positions in front of it (their K / V are rewritten with the same values).
            start = int(starts[i]) // BLOCK_SIZE * BLOCK_SIZE
            cached = 0
            if (
                start > 0
                and PREFILL_BATCH
                and prefill_padded_len(int(ends[i])) <= self.models[0].PREFILL_BATCH_MAX_PADDED
            ):
                # A short prompt whose first blocks are cached (a prefix hit) is recomputed from position 0 in a batch
                # pass (with the lane's other first chunks, traced) instead of an eager later chunk. Its cached blocks
                # are not written: another request may read them, and a pass of another length could compute their
                # K / V slightly differently. Their fill entries go to the null block.
                cached, start = start // BLOCK_SIZE, 0
            placed.append((i, lane, slot, start, int(ends[i]), cached))
            table_rows[lane][slot] = self._group_rows(tables, i)
        for lane, rows_of_lane in enumerate(table_rows):
            if rows_of_lane:
                self.models[lane].set_vllm_page_tables(rows_of_lane)
        # A lane's first chunks of one padded length (at most PREFILL_BATCH_MAX_PADDED) go in one pass
        # (TtMiMoModel.prefill_enqueue_batch, which pads every chunk to the longest); the other rows one by one. Every
        # pass is enqueued before any is read, the lanes interleaved so all replicas have work.
        # A lane's batch passes go before its other rows: a batch reads no cached KV (it starts at 0), and a traced
        # batch's replay must not run while an eager pass's output of that lane is alive.
        by_lane = []
        for lane in range(self.dp):
            groups = {}
            for p in placed:
                if p[1] == lane:
                    key = prefill_padded_len(p[4]) if self._batchable(p) else ("row", p[0])
                    groups.setdefault(key, []).append(p)
            batches = [g for key, g in groups.items() if not isinstance(key, tuple)]
            by_lane.append(batches + [g for key, g in groups.items() if isinstance(key, tuple)])
        order = [passes[k] for k in range(max(map(len, by_lane))) for passes in by_lane if k < len(passes)]
        programs = [model.mesh_device.num_program_cache_entries() for model in self.models]
        t0 = time.perf_counter()
        pending = []
        for rows_of_pass in order:
            lane = rows_of_pass[0][1]
            chunks = [
                (torch.as_tensor(tokens[i, start:end]).to(torch.int32), slot, cached)
                for i, _, slot, start, end, cached in rows_of_pass
            ]
            if self._batchable(rows_of_pass[0]):
                handle = self.models[lane].prefill_enqueue_batch(chunks)
            else:
                (chunk, slot, _), start = chunks[0], rows_of_pass[0][3]
                handle = self.models[lane].prefill_enqueue(chunk, slot, start_pos=start)
            pending.append((rows_of_pass, lane, handle))
        t1 = time.perf_counter()
        logits = [None] * rows
        for rows_of_pass, lane, handle in pending:
            out = self.models[lane].prefill_read(handle)[0].reshape(len(rows_of_pass), -1)
            for k, p in enumerate(rows_of_pass):
                logits[p[0]] = out[k]
        if TIMING:
            logger.info(
                f"MiMo vLLM timing: prefill {rows} rows in {len(pending)} passes "
                f"{[[p[4] - p[3] for p in rows_of] for rows_of, _, _ in pending]}: enqueue {1e3 * (t1 - t0):.0f} ms, "
                f"total {1e3 * (time.perf_counter() - t0):.0f} ms"
            )
        self._recapture_after_new_programs(programs)
        return torch.stack(logits).unsqueeze(1)

    def _batchable(self, placed_row):
        """A first chunk (start 0) whose padded length a prefill batch takes."""
        start, end = placed_row[3], placed_row[4]
        return PREFILL_BATCH and start == 0 and prefill_padded_len(end) <= self.models[0].PREFILL_BATCH_MAX_PADDED

    def _recapture_after_new_programs(self, programs):
        """``programs``: each replica's program-cache size before the prefill. A prefill that compiled new programs
        (a chunk length or offset first seen while serving) put their program-cache buffers in memory the replica's
        decode trace may write on replay (tech_reports/.../TraceCorrectness.md, unsafe allocations), so that replica's
        decode trace is captured again, around them. Every slot is idle during the capture, so no KV is written."""
        for lane, model in enumerate(self.models):
            if model.decode_trace is None or model.mesh_device.num_program_cache_entries() == programs[lane]:
                continue
            new = model.mesh_device.num_program_cache_entries() - programs[lane]
            model.release_decode_trace()
            model.capture_decode_trace()
            self.decode_recaptures += 1
            logger.info(
                f"MiMo vLLM: replica {lane} decode trace recaptured after {new} new prefill programs "
                f"({self.decode_recaptures} recaptures so far)"
            )

    # ------------------------------------------------------------------ decode
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        page_tables_per_layer=None,
        sampling_params=None,
        slot_remap=None,
        prompt_tokens=None,
        output_tokens=None,
        *,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        **kwargs,
    ):
        """One decode step for every lane's slots: tokens [B, 1], start_pos [B] (-1: idle slot), B = lanes x per_lane.

        ``sampling_params`` None: host sampling, every step reloads the inputs and the step returns logits
        [B, 1, vocab]. Otherwise device sampling (greedy only, ``max_device_top_k`` 1): the step returns the greedy
        tokens [B]. ``reload_inputs`` False (steady device decode) leaves ``tokens`` and ``start_pos`` unread: the
        trace consumes the tokens and positions its last replay wrote, and the adapter copies the RoPE inputs of its
        own position counter (``decode_positions``); ``reload_page_table`` then copies the page tables.

        The model keeps no other slot state (no penalty or RNG state), so a reloading step needs no ``slot_remap``:
        it rewrites every slot's tokens and positions. A steady step with a non-identity remap raises."""
        if "reset_batch" in kwargs:
            raise ValueError("MiMo implements decode_input_update_contract 1; got the legacy reset_batch keyword")
        if kwargs.get("defer_device_sampling"):
            raise ValueError("MiMo has no deferred device sampling (no device grammar)")
        if reload_inputs and reload_page_table:
            raise ValueError("reload_inputs already copies the page tables; got reload_page_table too")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("reset_sampling_state without reload_inputs")
        on_device = sampling_params is not None
        if not reload_inputs:
            if not on_device:
                raise ValueError("host sampling decodes from host inputs: reload_inputs must be True")
            if not enable_trace:
                raise ValueError("a steady decode (reload_inputs False) replays the decode trace: enable_trace False")
            if slot_remap is not None and any(int(j) != i for i, j in enumerate(slot_remap)):
                raise ValueError(f"a steady decode cannot move resident slot state: slot_remap {list(slot_remap)}")
        t0 = time.perf_counter()
        tables = self._layer_tables(page_table, page_tables_per_layer)
        rows = self.dp * self.per_lane
        if reload_inputs:
            tokens = torch.as_tensor(tokens).reshape(-1)
            positions = torch.as_tensor(start_pos).reshape(-1)
            rows = positions.numel()
            if rows > self.dp * self.per_lane:
                raise ValueError(f"{rows} decode rows > {self.dp} lanes x {self.per_lane} slots")
        handles = []
        for lane, model in enumerate(self.models):
            first = lane * self.per_lane
            last = min(first + self.per_lane, rows)
            if first >= rows:
                break
            if reload_inputs:
                tok = torch.zeros(model.max_batch, dtype=torch.int32)
                pos = torch.full((model.max_batch,), -1, dtype=torch.int32)
                tok[: last - first] = tokens[first:last].to(torch.int32)
                pos[: last - first] = positions[first:last].to(torch.int32)
            else:
                pos = self.decode_positions[lane]
                if pos is None:
                    raise RuntimeError(f"lane {lane}: a steady decode before any reloading decode")
            active = pos >= 0
            if on_device:
                self._check_greedy(sampling_params, first, [s for s in range(last - first) if active[s]])
            if not bool(active.any()):
                self.decode_positions[lane] = pos
                handles.append(None)
                continue
            if reload_inputs or reload_page_table:
                model.set_vllm_page_tables(
                    {s: self._group_rows(tables, first + s) for s in range(last - first) if active[s]}
                )
            if enable_trace and model.decode_trace is None:
                raise RuntimeError("decode trace requested before warmup_model_decode captured it")
            handle = model.decode_enqueue(tok, pos) if reload_inputs else model.decode_enqueue_resident(pos)
            self.decode_positions[lane] = torch.where(active, pos + 1, pos)
            handles.append((lane, last - first, handle))
        out = {"rows": rows, "handles": handles, "tokens": on_device}
        if TIMING:
            self._decode_timing(t0, reload_inputs, on_device)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(out), is_tokens=on_device)
        return out

    def _decode_timing(self, t0, reload_inputs, on_device):
        """Every 128 decode steps: the mean host time of decode_forward's submission, and the share of steps that
        reloaded their inputs or sampled on device."""
        now = time.perf_counter()
        stats = getattr(self, "_timing", None) or {"steps": 0, "host": 0.0, "reload": 0, "device": 0, "since": now}
        stats["steps"] += 1
        stats["host"] += now - t0
        stats["reload"] += int(reload_inputs)
        stats["device"] += int(on_device)
        if stats["steps"] == 128:
            logger.info(
                f"MiMo vLLM timing: decode {stats['steps']} steps in {1e3 * (now - stats['since']):.0f} ms, submit "
                f"{1e3 * stats['host'] / stats['steps']:.2f} ms/step, {stats['reload']} reloads, "
                f"{stats['device']} device-sampled"
            )
            stats = None
        self._timing = stats

    @staticmethod
    def _check_greedy(sampling_params, first, slots):
        """Device sampling is greedy: every active row has temperature 0 or top_k 1 (the plugin routes any other
        row to host sampling through ``max_device_top_k``)."""
        for s in slots:
            temperature = float(sampling_params.temperature[first + s])
            top_k = int(sampling_params.top_k[first + s])
            if temperature != 0.0 and top_k != 1:
                raise ValueError(
                    f"row {first + s}: device sampling is greedy only; got temperature {temperature}, top_k {top_k}"
                )

    def read_decode_output(self, out, async_read=False, **kwargs):
        """Starts reading a decode step's output: its greedy tokens (device sampling), without blocking when
        ``async_read`` (returns the per-lane host tensors and their read events), or its logit rows (host sampling;
        read now, returned with no events)."""
        if kwargs.get("sample_rows") is not None:
            raise ValueError("MiMo declares no selective host readback; got sample_rows")
        if not out["tokens"]:
            logits = self._read_logits(out)
            return (logits, []) if async_read else logits
        reads, events = [], []
        for entry in out["handles"]:
            if entry is None:
                continue
            lane, n, handle = entry
            reads.append((lane, n, handle["next_tokens"].cpu(blocking=not async_read)))
            if async_read:
                events.append(ttnn.record_event(self.models[lane].mesh_device, 0))
        host = {"rows": out["rows"], "reads": reads}
        return (host, events) if async_read else host

    def _read_logits(self, out):
        vocab = self.gen.cfg.vocab_size
        logits = torch.zeros(out["rows"], 1, vocab, dtype=torch.float32)
        for entry in out["handles"]:
            if entry is None:
                continue
            lane, n, handle = entry
            logits[lane * self.per_lane : lane * self.per_lane + n, 0] = self.models[lane].decode_read_rows(handle)[:n]
        return logits

    def process_decode_output_host(self, out, is_tokens=False, **kwargs):
        """Host output of ``read_decode_output``: greedy tokens int32 [B] in slot order (``is_tokens``; idle slots
        0), else the logits it read."""
        if kwargs.get("sample_rows") is not None:
            raise ValueError("MiMo declares no compact host logits; got sample_rows")
        if not is_tokens:
            if not isinstance(out, torch.Tensor):
                raise ValueError("host sampling: read_decode_output returns the logits")
            return out
        if isinstance(out, torch.Tensor) or "reads" not in out:
            raise ValueError("device sampling: expected the token reads of read_decode_output")
        tokens = torch.zeros(out["rows"], dtype=torch.int32)
        for lane, n, host in out["reads"]:
            row = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[:n]
            tokens[lane * self.per_lane : lane * self.per_lane + n] = row.to(torch.int32)
        return tokens

    # ------------------------------------------------------------------ warmup and lifetime
    def _max_prefill_tokens(self):
        """The most tokens one prefill chunk of the model can hold: vLLM's per-step token budget plus the
        BLOCK_SIZE - 1 positions prefill_forward recomputes in front of an unaligned start, at most the model's
        max_prefill_chunk (prefill_enqueue splits longer chunks) and the context."""
        budget = self.prefill_token_budget
        if budget is None and self._kv_cache_spec_source is not None:
            budget = self._kv_cache_spec_source[0].scheduler_config.max_num_batched_tokens
        model = self.models[0]
        if budget is None:
            budget = model.max_prefill_chunk
        return min(budget + BLOCK_SIZE - 1, model.max_prefill_chunk, self.max_seq_len)

    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, can_sample_on_device=False, **kwargs):
        """Compile every prefill program a served chunk can run, on every replica, before any decode trace is
        captured (prefill runs eagerly, so ``enable_trace`` changes nothing). One chunk per padded length at start 0
        and per later start that changes what the programs key on (model.prefill_warmup_chunks), each through one
        layer of each kind (TtMiMoModel.prefill_warmup_layers). Writes only KV blocks no request owns yet."""
        if self.already_warmed_up_prefill or self.prefill_programs_compiled:
            self.already_warmed_up_prefill = True
            return
        t0 = time.time()
        # Decode makes persistent L1 buffers on its first step (the MoE fabric row gather's among them). Layernorm and
        # some reductions size their circular buffers from the free L1 and key the program on that plan
        # (ttnn/cpp/ttnn/operations/core/program_cache_l1.hpp), so prefill is compiled with those buffers in place, as
        # served prefills run. The plugin warms prefill before decode and, with trace_mode decode_only, not again after.
        idle = torch.full((self.models[0].max_batch,), -1, dtype=torch.int32)  # every slot idle: no KV is written
        handles = [(m, m.decode_enqueue(torch.zeros_like(idle), idle)) for m in self.models if m.decode_trace is None]
        for m, handle in handles:
            m.decode_read(handle)
        model0 = self.models[0]
        groups = sorted(set(self.layer_groups))
        width = model0.vllm_tables[groups[0]].shape[1]
        num_blocks = self.kv_caches[0][0][0].shape[0]
        window = next((l.attn.sliding_window for l in model0.layers if l.is_swa), BLOCK_SIZE)
        max_tokens = self._max_prefill_tokens()
        chunks = prefill_warmup_chunks(max_tokens, self.max_seq_len, BLOCK_SIZE, window)
        layers = model0.prefill_warmup_layers()
        # Slot 0 of every group maps positions from 0 to blocks 1, 2, ..: free while vLLM has not started.
        blocks = torch.arange(1, min(width, num_blocks - 1) + 1, dtype=torch.int32)
        if blocks.numel() * BLOCK_SIZE < max(start for _, start in chunks):  # the later chunks' window blocks
            raise RuntimeError(f"{num_blocks} KV blocks cannot back the prefill warmup's chunks {chunks}")
        for model in self.models:
            model.set_vllm_page_tables({0: {g: blocks for g in groups}})
        programs = [m.mesh_device.num_program_cache_entries() for m in self.models]
        tokens = torch.arange(max_tokens, dtype=torch.int32) % 1000 + 1000
        for n, start in chunks:
            handles = [(m, m.prefill_enqueue(tokens[:n], 0, start_pos=start, layers=layers)) for m in self.models]
            for m, h in handles:
                m.prefill_read(h)
        # Batches of first chunks (prefill_forward): every user count and padded length a lane's step can hold. Slots
        # 1 .. users - 1 write the blocks after slot 0's.
        batches = prefill_warmup_batches(max_tokens, self.per_lane, model0.PREFILL_BATCH_MAX_PADDED, self.max_seq_len)
        batches = batches if PREFILL_BATCH else []  # prefill_forward runs every batchable first chunk as a batch
        span = -(-model0.PREFILL_BATCH_MAX_PADDED // BLOCK_SIZE)
        if self.per_lane * span > blocks.numel():
            raise RuntimeError(f"{num_blocks} KV blocks cannot back the prefill warmup's batches {batches}")
        for model in self.models:
            model.set_vllm_page_tables({s: {g: blocks[s * span :] for g in groups} for s in range(self.per_lane)})
        for users, n in batches:
            chunks_of = [(tokens[:n], s) for s in range(users)]
            handles = [(m, m.prefill_enqueue_batch(chunks_of, layers=layers)) for m in self.models]
            for m, h in handles:
                m.prefill_read(h)
        for model in self.models:
            empty = torch.zeros(width, dtype=torch.int32)
            model.set_vllm_page_tables({s: {g: empty for g in groups} for s in range(self.per_lane)})
        # Traced batches: their persistent buffers now, before any capture, and one eager pass each through every
        # layer (compiling whatever the capture will record); warmup_model_decode captures them after the decode trace.
        if PREFILL_TRACE and PREFILL_BATCH:
            for model in self.models:
                for users, n in batches:
                    padded = prefill_padded_len(n)
                    if (users, padded) not in model.prefill_traces:
                        model.setup_prefill_batch_trace(users, padded)
                        model.warm_prefill_batch_trace(users, padded)
            for model in self.models:
                ttnn.synchronize_device(model.mesh_device)
        self.already_warmed_up_prefill = self.prefill_programs_compiled = True
        new = [m.mesh_device.num_program_cache_entries() - p for m, p in zip(self.models, programs)]
        logger.info(
            f"MiMo vLLM: prefill warmed up in {time.time() - t0:.1f}s: {len(chunks)} chunks of up to {max_tokens} "
            f"tokens and {len(batches)} batches through layers {layers}, {new} new programs per replica"
        )

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, max_batch_size=None, **kwargs):
        """``enable_trace``: capture each replica's decode trace (every slot idle, so no KV is written); otherwise
        compile the eager step."""
        for lane, model in enumerate(self.models):
            if enable_trace:
                if model.decode_trace is None:
                    model.capture_decode_trace()
                    logger.info(f"MiMo vLLM: replica {lane} decode trace captured")
                pending = [key for key, t in model.prefill_traces.items() if t["id"] is None]
                for k, (users, padded) in enumerate(pending):
                    logger.debug(f"MiMo vLLM: replica {lane} capturing prefill trace {k + 1}/{len(pending)}")
                    model.capture_prefill_batch_trace(users, padded)
                if pending:
                    logger.info(f"MiMo vLLM: replica {lane} prefill traces captured for (users, length) {pending}")
            else:
                idle = torch.full((model.max_batch,), -1, dtype=torch.int32)
                model.decode_read(model.decode_enqueue(torch.zeros(model.max_batch, dtype=torch.int32), idle))

    def release_persistent_capture(self):
        for model in self.models:
            model.release_decode_trace()
            model.release_prefill_traces()

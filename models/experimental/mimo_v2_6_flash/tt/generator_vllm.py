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
[num_blocks, 1, 64, 192] and is shared that way; V is 128 wide on the global layers and 192 wide (zero-padded,
attention.py) on the SWA layers, so V is shared among layers of the same kind. The global layers keep both copies
of their KV head on a chip pair (setup_vllm_kv_cache): a block can be shared by requests in slots of either parity.

Sampling runs on the host: prefill and decode return last-token logits [B, 1, vocab]. Decode replays one traced
step per replica (captured in ``warmup_model_decode``); prefill runs eagerly in chunks of up to 4096 tokens, between
the replays. A program first compiled after the capture would keep buffers the replays overwrite, so
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
from .model import prefill_warmup_chunks

TP = 8
BLOCK_SIZE = 64
# Per-chip bytes per token of the vLLM KV layout above for MiMo-V2.6-Flash (9 global, 39 SWA layers, so 9 shared K
# buffers, 9 global V buffers and 8 SWA V buffers), BFP8: (9 * 192 + 9 * 128 + 8 * 192) values * 1088 / 1024 bytes.
KV_BYTES_PER_TOKEN = (9 * 192 + 9 * 128 + 8 * 192) * 1088 / 1024
# Per-chip DRAM given to the KV pool when MIMO_KV_GB_PER_CHIP is unset (weights take 23.7 of 34 GB per chip).
DEFAULT_KV_GB_PER_CHIP = 6.0
# Blocks vLLM's sliding-window groups can hold per request while a prefill chunk of up to 4096 tokens is in flight
# (SlidingWindowSpec.max_admission_blocks_per_request), in tokens: 5 SWA groups x (127 + 4096 tokens + 1 block).
SWA_TOKENS_PER_USER = 5 * (math.ceil((127 + 4096) / BLOCK_SIZE) + 1) * BLOCK_SIZE


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
    }
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
                v_dim = attn.v_head_dim if attn.split_users else attn.head_dim
                caches.append(
                    [
                        self._buffer(model, buffers, ("k", t, attn.head_dim), num_blocks, attn.head_dim),
                        self._buffer(model, buffers, ("v", t, v_dim), num_blocks, v_dim),
                    ]
                )
            model.setup_vllm_kv_cache(self.per_lane, caches, groups, table_width)
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
            placed.append((i, lane, slot, start, int(ends[i])))
            table_rows[lane][slot] = self._group_rows(tables, i)
        for lane, rows_of_lane in enumerate(table_rows):
            if rows_of_lane:
                self.models[lane].set_vllm_page_tables(rows_of_lane)
        # Enqueue every row before reading any, the rows of different lanes interleaved so all replicas have work.
        by_lane = [[p for p in placed if p[1] == lane] for lane in range(self.dp)]
        order = [rows_of[k] for k in range(max(map(len, by_lane))) for rows_of in by_lane if k < len(rows_of)]
        programs = [model.mesh_device.num_program_cache_entries() for model in self.models]
        pending = {}
        for i, lane, slot, start, end in order:
            chunk = torch.as_tensor(tokens[i, start:end]).to(torch.int32)
            pending[i] = (lane, self.models[lane].prefill_enqueue(chunk, slot, start_pos=start))
        logits = [None] * rows
        for i, (lane, handle) in pending.items():
            logits[i] = self.models[lane].prefill_read(handle)[0]
        self._recapture_after_new_programs(programs)
        return torch.stack(logits).unsqueeze(1)

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
        **kwargs,
    ):
        """One decode step for every lane's slots: tokens [B, 1], start_pos [B] (-1: idle slot), B = lanes x per_lane.

        ``slot_remap`` is ignored: the model keeps no slot-owned state between steps (tokens, positions and page
        tables come from the host every step; the KV is addressed through the page tables)."""
        if sampling_params is not None:
            raise ValueError("MiMo samples on the host; got device sampling_params")
        tables = self._layer_tables(page_table, page_tables_per_layer)
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
            tok = torch.zeros(model.max_batch, dtype=torch.int32)
            pos = torch.full((model.max_batch,), -1, dtype=torch.int32)
            tok[: last - first] = tokens[first:last].to(torch.int32)
            pos[: last - first] = positions[first:last].to(torch.int32)
            if int(pos.max()) < 0:
                handles.append(None)
                continue
            model.set_vllm_page_tables(
                {s: self._group_rows(tables, first + s) for s in range(last - first) if pos[s] >= 0}
            )
            if enable_trace and model.decode_trace is None:
                raise RuntimeError("decode trace requested before warmup_model_decode captured it")
            handles.append((lane, last - first, model.decode_enqueue(tok, pos)))
        out = {"rows": rows, "handles": handles}
        if read_from_device:
            return self._read_logits(out)
        return out

    def _read_logits(self, out):
        vocab = self.gen.cfg.vocab_size
        logits = torch.zeros(out["rows"], 1, vocab, dtype=torch.float32)
        for entry in out["handles"]:
            if entry is None:
                continue
            lane, n, handle = entry
            logits[lane * self.per_lane : lane * self.per_lane + n, 0] = self.models[lane].decode_read_rows(handle)[:n]
        return logits

    def read_decode_output(self, out, async_read=False, **kwargs):
        """Reads the logits now (the read is not split from the forward yet); returns them with no pending events."""
        logits = self._read_logits(out)
        return (logits, []) if async_read else logits

    def process_decode_output_host(self, out, is_tokens=False, **kwargs):
        if is_tokens:
            raise ValueError("MiMo samples on the host; decode returns logits")
        return out if isinstance(out, torch.Tensor) else self._read_logits(out)

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
        for model in self.models:
            model.set_vllm_page_tables({0: {g: torch.zeros(width, dtype=torch.int32) for g in groups}})
        self.already_warmed_up_prefill = self.prefill_programs_compiled = True
        new = [m.mesh_device.num_program_cache_entries() - p for m, p in zip(self.models, programs)]
        logger.info(
            f"MiMo vLLM: prefill warmed up in {time.time() - t0:.1f}s: {len(chunks)} chunks of up to {max_tokens} "
            f"tokens through layers {layers}, {new} new programs per replica"
        )

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, max_batch_size=None, **kwargs):
        """``enable_trace``: capture each replica's decode trace (every slot idle, so no KV is written); otherwise
        compile the eager step."""
        for lane, model in enumerate(self.models):
            if enable_trace:
                if model.decode_trace is None:
                    model.capture_decode_trace()
                    logger.info(f"MiMo vLLM: replica {lane} decode trace captured")
            else:
                idle = torch.full((model.max_batch,), -1, dtype=torch.int32)
                model.decode_read(model.decode_enqueue(torch.zeros(model.max_batch, dtype=torch.int32), idle))

    def release_persistent_capture(self):
        for model in self.models:
            model.release_decode_trace()

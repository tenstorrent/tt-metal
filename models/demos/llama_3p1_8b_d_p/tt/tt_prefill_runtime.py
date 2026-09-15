# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B chunked-prefill runtime (tt-blaze#4148).

The engine-facing handle for the prefill model: it owns everything that is built once per rank —
the model, the whole-cache indexed RoPE tables, the RoPE transformation matrix and the CCL manager
— and drives one chunk per ``prefill_chunk`` call. It does **not** own the KV cache; the engine
allocates that (via the adapter's ``allocate_kv_cache``) and passes it into every call that touches
it, so a runtime can be rebuilt without disturbing a live cache.

Structural contract is ``models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`` §2, and the
implementation follows ``minimax_m3/tt/tt_prefill_runtime.py``. Llama's version is much smaller:
there is no MoE gate, no MSA/sparse cache, no LM head (prefill is headless — the populated cache is
the product), and no per-layer type schedule, so the runtime is the chunk loop and nothing else.

**What "chunked" costs and why the geometry rules exist.** The KV cache is block-cyclic across the
SP rows, and the indexed RoPE tables are reordered to match, so a chunk's tokens must land on the
same chip as the RoPE rows that rotate them and the cache rows that store them. Three rules keep
that true, all checked up front rather than surfacing as a KV PCC failure at some interior
position:

  * ``chunk_size % (sp * 32) == 0`` — each chip's per-chunk slab is a whole number of 32-token DRAM
    shards (``kv_cache.validate_chunk_layout``).
  * ``max_seq_len % chunk_size == 0`` — the cache is an exact number of chunk slabs, so no chunk
    straddles its end.
  * ``actual_start % chunk_size == 0`` — a continuation resumes on a chunk boundary. The cache read
    addresses the prefix in whole chunks, so resuming mid-chunk scrambles it.

Kept free of reference-model and safetensors imports; the import-light contract is asserted by
``tests/unit/test_scaffold.py``.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from loguru import logger

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.ccl import CCLManager
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    Llama31KVCache,
    validate_chunk_layout,
)
from models.demos.llama_3p1_8b_d_p.tt.model import TtLlamaPrefillModel
from models.demos.llama_3p1_8b_d_p.tt.rope import build_indexed_rope, build_transformation_mat


@dataclass
class TtPrefillRuntimeConfig:
    """Per-rank knobs. Mirrors the fields the engine reads off ``runtime.config``."""

    max_seq_len: int
    chunk_size: int
    mesh_shape: tuple = (4, 8)
    num_layers: int = Llama31_8BConfig.NUM_LAYERS
    num_users: int = 1
    tp_axis: int = 1
    num_links: int = 1
    topology: ttnn.Topology = ttnn.Topology.Linear
    weight_cache_path: Optional[Path] = None
    # The engine allocates the cache and hands it in; a runtime that owned one would keep a second
    # copy alive across a rebuild.
    owns_kv_cache: bool = False
    # Pipeline role. Single-rank prefill is both first and last: it embeds tokens and keeps the
    # headless cache-fill contract.
    is_first_rank: bool = True
    is_last_rank: bool = True
    first_layer_idx: int = 0
    vocab_size: int = Llama31_8BConfig.VOCAB_SIZE
    activations_dtype: ttnn.DataType = ttnn.bfloat16
    weights_dtype: ttnn.DataType = ttnn.bfloat16
    _derived: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.tp_axis not in (0, 1):
            raise ValueError(f"tp_axis must be 0 or 1, got {self.tp_axis}")
        validate_chunk_layout(self.max_seq_len, self.chunk_size, self.sp_factor)

    @property
    def sp_axis(self) -> int:
        return 1 - self.tp_axis

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]

    @property
    def chunk_size_local(self) -> int:
        """Rows of one chunk that land on each chip."""
        return self.chunk_size // self.sp_factor


class TtPrefillRuntime:
    """Chunked-prefill runtime for one rank."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: TtPrefillRuntimeConfig,
        state_dict: Optional[dict] = None,
    ):
        """
        Args:
            state_dict: a HuggingFace Llama-3.1-8B state dict. Empty/None builds random weights,
                which is shape and memory bring-up only.
        """
        if tuple(mesh_device.shape) != tuple(config.mesh_shape):
            raise ValueError(f"mesh_device shape {tuple(mesh_device.shape)} != config.mesh_shape {config.mesh_shape}")

        self.mesh_device = mesh_device
        self.config = config
        self.mesh_config = MeshConfig(config.mesh_shape, tp=config.tp_factor, tp_axis=config.tp_axis)
        self._layer_completion_sink: Optional[Callable[[int, int], None]] = None

        self.ccl_manager = CCLManager(mesh_device, num_links=config.num_links, topology=config.topology)

        # Built ONCE and persistent across every chunk: the tables cover every cache position, and
        # rotary_embedding_indexed derives this chunk's start row on-device from cached_len plus the
        # device's SP coordinate. Rebuilding them per chunk would be a host reshard per layer-group.
        self.rope_mats = build_indexed_rope(
            mesh_device,
            head_dim=Llama31_8BConfig.HEAD_DIM,
            max_seq_len=config.max_seq_len,
            chunk_size=config.chunk_size,
            sp_axis=config.sp_axis,
        )
        self.transformation_mat = build_transformation_mat(mesh_device)

        self.model = TtLlamaPrefillModel(
            mesh_device=mesh_device,
            mesh_config=self.mesh_config,
            state_dict=state_dict,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            first_layer_idx=config.first_layer_idx,
            num_links=config.num_links,
            topology=config.topology,
            activations_dtype=config.activations_dtype,
            weights_dtype=config.weights_dtype,
            weight_cache_path=config.weight_cache_path,
        )
        logger.info(
            f"TtPrefillRuntime: mesh={config.mesh_shape} tp={config.tp_factor} sp={config.sp_factor}, "
            f"chunk={config.chunk_size} ({config.chunk_size_local}/chip), max_seq_len={config.max_seq_len}, "
            f"layers {config.first_layer_idx}..{config.first_layer_idx + config.num_layers - 1}"
        )

    # =====================================================================
    # Engine-facing interface
    # =====================================================================
    def set_layer_completion_sink(self, sink: Callable[[int, int], None]) -> None:
        """Register ``sink(global_layer_idx, request_id)``, called once per layer mid-forward.

        The index is the layer's GLOBAL index — the model gives its layers global indices at build
        time (``first_layer_idx + offset``), so nothing needs adding here. A rank-local index would
        make every rank's local layer k collide on one ``seq = request_id * num_layers + layer_idx``
        and all but one completion would be dropped, silently, as a duplicate.
        """
        self._layer_completion_sink = sink

    def make_chunk_input(self, token_ids) -> ttnn.Tensor:
        """One chunk's token IDs as an SP-sharded uint32 ROW_MAJOR tensor, per-chip ``(1, 1, s_local)``.

        Row ``r`` holds the contiguous slice ``[r * s_local : (r+1) * s_local]``, replicated across
        the TP columns — the same per-chip layout the request-mode H2D socket delivers, so the
        socket path and this one feed one code path.

        On a non-first pipeline rank the real input is a hidden-state activation arriving over the
        D2D socket; a placeholder of the right spec is returned so warm-up can still run.
        """
        if not self.config.is_first_rank:
            return self.make_placeholder_activation()

        if len(token_ids) != self.config.chunk_size:
            raise ValueError(
                f"chunk input must be exactly chunk_size={self.config.chunk_size} tokens (pad the "
                f"tail), got {len(token_ids)}"
            )
        sp = self.config.sp_factor
        tokens = torch.as_tensor(token_ids, dtype=torch.int32).reshape(sp, 1, self.config.chunk_size_local)
        dims = [None, None]
        dims[self.config.sp_axis] = 0  # the sp-major leading dim; replicated across TP
        return ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=self.config.mesh_shape, dims=tuple(dims)),
        )

    def make_placeholder_activation(self) -> ttnn.Tensor:
        """A TP-sharded residual-shaped activation, for warming up a non-first pipeline rank."""
        return ttnn.from_torch(
            torch.zeros(1, 1, self.config.chunk_size_local, self.mesh_config.shard_size(Llama31_8BConfig.EMB_SIZE)),
            device=self.mesh_device,
            dtype=self.config.activations_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def compile(self, kv_cache: Llama31KVCache) -> None:
        """Warm up every KV length the served loop can reach, so no served chunk pays a first-run JIT.

        Sweeps ``actual_start`` over the whole per-user cache in chunk steps rather than warming
        only chunk 0. Each distinct ``cached_len`` is a **different program** — the cache-read half
        of attention reads a prefix whose length grows with the chunk index, so its shapes differ
        per bucket. Warming only chunk 0 would move the JIT cost off the first chunk and onto every
        subsequent one, which is worse than not warming at all: it puts a multi-second stall in the
        middle of a served request instead of before it.

        Every warm-up writes zeros into slot 0, and the first real request's chunks overwrite
        exactly the same ranges in the same order, so nothing stale survives.
        """
        config = self.config
        starts = list(range(0, config.max_seq_len - config.chunk_size + 1, config.chunk_size))
        if config.sp_factor == 1:
            # At SP=1 there is no cache-read path at all (plain causal SDPA cannot express a Q
            # offset), so a bucket past the first would raise rather than compile. One bucket is
            # also all there is to warm: without a cache read, every chunk runs the same program.
            starts = starts[:1]

        logger.info(f"TtPrefillRuntime.compile: warming {len(starts)} KV-length bucket(s)")
        t0 = time.perf_counter()
        for start in starts:
            out = self.prefill_chunk(
                self.make_chunk_input([0] * config.chunk_size),
                kv_cache,
                slot_id=0,
                actual_start=start,
                actual_end=start + config.chunk_size,
            )
            if out is not None:
                ttnn.deallocate(out)
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"TtPrefillRuntime.compile: {len(starts)} bucket(s) in {(time.perf_counter() - t0) * 1e3:.0f} ms")

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_cache: Llama31KVCache,
        *,
        slot_id: int = 0,
        actual_start: int = 0,
        actual_end: Optional[int] = None,
        request_id: int = 0,
        d2h_service=None,
        return_hidden_states: bool = False,
    ):
        """Prefill ONE chunk into user ``slot_id``'s slice of the engine-owned ``kv_cache``.

        ``[actual_start, actual_end)`` is the absolute KV-position range of this chunk's *real*
        tokens. ``actual_start`` is the cache write offset (the valid prefix already cached); the
        final chunk's tail may be pad, so ``actual_end`` can be less than
        ``actual_start + chunk_size``. The chunk still occupies physical positions
        ``[actual_start, actual_start + chunk_size)`` — causality makes the pad tail inert, because
        pad sits at the *end* and no real token attends forward to it.

        Must be called once per chunk, in order: a chunk's KV has to be written before the next
        chunk reads it.

        Returns ``None`` on the last/single rank (headless — the populated cache is the output), or
        this rank's output hidden-state activation on a non-last pipeline rank. Pass
        ``return_hidden_states`` to get the final-normed hidden states anyway, which is what the PCC
        tests grade; serving never sets it.
        """
        if d2h_service is not None:
            raise NotImplementedError(
                "Llama-3.1-8B prefill emits layer acks through the host callback "
                "(set_layer_completion_sink), not the D2H path; run with PREFILL_LAYER_ACK_D2H=0."
            )
        config = self.config
        if actual_end is None:
            actual_end = actual_start + config.chunk_size
        if not 0 <= slot_id < config.num_users:
            raise ValueError(f"slot_id {slot_id} out of range [0, {config.num_users})")
        if actual_start + config.chunk_size > config.max_seq_len:
            raise ValueError(
                f"chunk at actual_start={actual_start} would run past the per-user cache "
                f"({config.max_seq_len}); the request is longer than the cache allows"
            )
        if not actual_start < actual_end <= actual_start + config.chunk_size:
            raise ValueError(
                f"[actual_start={actual_start}, actual_end={actual_end}) is not within one chunk of "
                f"{config.chunk_size}"
            )
        # The block-cyclic cache read addresses the prefix in whole chunks. Resuming from a prefix
        # that is not chunk-aligned does not fail in the kernel, it reads the wrong rows — so refuse
        # here, where the caller's offset is named, rather than deep in attention.
        if actual_start % config.chunk_size:
            raise ValueError(
                f"actual_start={actual_start} must be a multiple of chunk_size={config.chunk_size}; "
                f"resuming from a non-chunk-aligned prefix is not supported"
            )
        if actual_start % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK:
            raise ValueError(f"actual_start={actual_start} must be a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}")
        # Rank-local, not global: the cache holds this rank's layers only, indexed from 0 (see
        # tt/decoder.py's cache_layer_idx).
        if kv_cache.num_layers < config.num_layers:
            raise ValueError(
                f"kv_cache has {kv_cache.num_layers} layer slots per user but this rank runs "
                f"{config.num_layers} layers (global {config.first_layer_idx}.."
                f"{config.first_layer_idx + config.num_layers - 1})"
            )

        # The first rank embeds the SP-sharded tokens; a later rank is handed the upstream hidden
        # state and feeds it straight in.
        if config.is_first_rank:
            x = self.model.embed(input_tensor)
            ttnn.deallocate(input_tensor)
        else:
            x = input_tensor

        sink = self._layer_completion_sink
        on_layer_complete = (lambda layer_idx: sink(layer_idx, request_id)) if sink is not None else None

        out = self.model(
            x,
            self.rope_mats,
            self.transformation_mat,
            kv_cache=kv_cache,
            ccl_manager=self.ccl_manager,
            user_id=slot_id,
            cached_len=actual_start,
            indexed_rope=True,
            # The final norm belongs to the last rank only — a middle rank forwards the raw
            # residual, and normalising there would apply the final norm once per rank. On the last
            # rank it is still skipped unless someone wants the hidden states, because headless
            # prefill discards them and the norm is pure cost.
            apply_final_norm=config.is_last_rank and return_hidden_states,
            on_layer_complete=on_layer_complete,
        )

        if not config.is_last_rank:
            return out  # hidden-state activation for the next pipeline rank
        if return_hidden_states:
            return out
        ttnn.deallocate(out)
        return None

    def prefill_prompt(
        self,
        token_ids,
        kv_cache: Llama31KVCache,
        *,
        slot_id: int = 0,
        start_pos: int = 0,
        request_id: int = 0,
        return_hidden_states: bool = False,
    ):
        """Drive the whole chunk loop for one prompt. Convenience for tests and single-process runs.

        The serving engine owns this loop itself (it interleaves sockets, migration and several
        users), so this is deliberately the simple version: pad the prompt up to a chunk boundary,
        walk it in order, and hand each chunk the prefix length before it.

        ``request_id`` is passed to every chunk unchanged, because a multi-chunk prompt is ONE
        request. It is not the chunk index: the migration layer keys on
        ``request_id * num_layers + layer_idx``, so varying it per chunk would announce each chunk
        as a different request and scatter one request's KV across several keys.

        Returns the last chunk's final-normed hidden states when ``return_hidden_states``, else None.
        """
        chunk_size = self.config.chunk_size
        tokens = list(token_ids)
        total = len(tokens)
        if start_pos % chunk_size:
            raise ValueError(f"start_pos={start_pos} must be a multiple of chunk_size={chunk_size}")
        if start_pos + total > self.config.max_seq_len:
            raise ValueError(
                f"prompt of {total} tokens at start_pos={start_pos} exceeds the per-user cache "
                f"({self.config.max_seq_len})"
            )

        last = None
        for offset in range(0, total, chunk_size):
            chunk = tokens[offset : offset + chunk_size]
            real = len(chunk)
            if real < chunk_size:
                # Pad the tail to a full chunk. Token 0 is as good as any: causality means no real
                # token attends forward to it, and its own KV rows are never read because
                # actual_end stops short of them.
                chunk = chunk + [0] * (chunk_size - real)
            if last is not None:
                ttnn.deallocate(last)
            actual_start = start_pos + offset
            last = self.prefill_chunk(
                self.make_chunk_input(chunk),
                kv_cache,
                slot_id=slot_id,
                actual_start=actual_start,
                actual_end=actual_start + real,
                request_id=request_id,
                return_hidden_states=return_hidden_states,
            )
        ttnn.synchronize_device(self.mesh_device)
        return last

    def kv_migration_base_address(self, kv_cache: Llama31KVCache) -> int:
        """This rank's KV base DRAM address — the anchor the engine all-gathers to merge stages.

        The K cache, not V: the two are separate allocations, and the address table the migration
        layer builds is anchored on K with V at a fixed stride from it.
        """
        return int(kv_cache.k.buffer_address())

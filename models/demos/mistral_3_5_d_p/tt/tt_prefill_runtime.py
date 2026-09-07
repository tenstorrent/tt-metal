# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 single-rank chunked-prefill runtime.

Donor: ``gpt_oss_d_p/tt/tt_prefill_runtime.py`` (``serving.runtime``), which pairs with the KV-cache
cluster as the map requires and is already the GQA shape: ``compile`` / ``make_chunk_input`` /
``_resolve_kv`` / ``prefill_chunk`` / ``read_slot_kv`` / ``gather_layer``, SP-sharded uint32 H2D
token delivery, and per-layer acks through ``set_layer_ack_channel`` rather than the D2H path.
Satisfies the contract in ``models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`` §2.

Adapted from the donor:
  * **88 layers, head_dim 128**, and no sinks / sliding-window plumbing to thread;
  * **no MoE gate mode** — ``default_gate_mode`` and the expert dtype are MoE-only knobs;
  * **capacity vs context.** ``max_seq_len`` here is the ALLOCATED per-user capacity and must be a
    whole number of chunks (the block-cyclic rope and the address table both tile by chunk). The
    spec's servable context, 262144, is 51.2 chunks of 5120, so the adapter passes
    ``spec.cache_capacity`` (266240) and ``prefill_chunk`` asserts requests stay inside the SERVABLE
    context, not merely inside the capacity — otherwise a request could quietly address the padding
    tail that no golden covers;
  * **``kv_migration_stages`` is written fresh** (see the method): the donor migrates through
    ``kv_migration_base_address`` + a multi-config table and does not implement stages at all.

Cache ownership works as in the donor: the runtime can OWN its KV cache (``owns_kv_cache=True``, the
standalone galaxy harness) or run engine-owned (``owns_kv_cache=False``, the adapter path, where the
engine allocates via the adapter and passes the ``KvCaches`` handle into every call).

CHUNKED prefill is supported: the SP cache-backed RingJointSDPA path reads the block-cyclic packed
cache from chunk 0 onward. A one-shot request whose capacity equals its single chunk has equal-sized
Q and K/V slabs, which the ring reader rejects, so it keeps the all-gather bootstrap. The single-chip
(sp == 1) cache-read remains ``NotImplementedError`` and is never taken on the galaxy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.mistral_3_5_d_p.spec import SPEC

from .attention import MistralKVCache, allocate_kv_cache
from .rope import build_indexed_rope, hf_to_meta_head_permutation, yarn_params_from_config


def resolve_chunk_sizes(default_chunk_size: int, additional_chunk_sizes: tuple, capacity: int) -> tuple:
    """Supported chunk sizes, deduped, largest first; each must divide the cache capacity.

    The constraint is the whole-cache indexed rope: its block-cyclic period IS the chunk size, so a
    size that does not tile the cache would put a chunk's rope rows at the wrong cache positions.
    """
    sizes = tuple(sorted({default_chunk_size, *additional_chunk_sizes}, reverse=True))
    for size in sizes:
        if capacity % size != 0:
            raise ValueError(
                f"cache capacity ({capacity}) must be a multiple of every supported chunk size; "
                f"{size} does not divide it (supported: {sizes})"
            )
    return sizes


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int  # layers built/cached by this runtime (== the model total for single-rank)
    # ALLOCATED per-user KV capacity in tokens. Must be a whole number of chunk_size; pass
    # spec.cache_capacity, not spec.max_seq_len (262144 is 51.2 chunks of 5120).
    max_seq_len: int
    # The SERVABLE context in tokens: requests may not address beyond this. Defaults to the spec's
    # max_seq_len, capped at the capacity for a reduced-capacity run.
    servable_seq_len: Optional[int] = None
    mesh_shape: tuple = field(default_factory=lambda: SPEC.mesh_shape)  # (SP rows, TP cols)
    default_chunk_size: int = SPEC.chunk_size  # tokens per prefill_chunk() call
    # Other sizes this instance can serve per call (each gets its own indexed rope).
    additional_chunk_sizes: tuple = ()
    num_users: int = 1  # independent cache slots (user-major batch)
    sp_axis: int = SPEC.sp_axis
    tp_axis: int = SPEC.tp_axis
    topology: ttnn.Topology = ttnn.Topology.Linear
    cache_dtype: ttnn.DataType = SPEC.kv_cache_dtype
    # Override the spec's weight dataformats. None => the spec. Bring-up A/B only; see
    # DecoderLayer's weight_dtype and README.md "Accuracy".
    weight_dtype: Optional[ttnn.DataType] = None
    weight_cache_path: Optional[Path] = None
    # True -> the runtime allocates and owns its KV cache (self.kv_cache), the standalone harness
    # path. The adapter/engine path sets this False and passes the engine-owned KvCaches in.
    owns_kv_cache: bool = True
    # Pipeline-rank flags the common prefill runner reads off runtime.config (single-rank => both
    # True). first_layer_idx is the GLOBAL index of this rank's first layer.
    is_first_rank: bool = True
    is_last_rank: bool = True
    first_layer_idx: int = 0
    # Read off runtime.config by the common runner (``prefill_runner``) alongside is_last_rank.
    # Capturing the per-chunk forward as a ttnn trace and replaying it is a PERF feature and out of
    # scope for bring-up, so it stays False — but the field must exist, because the engine reads it
    # unconditionally and an AttributeError there kills the runner during startup. The gpt-oss donor
    # predates that read and omits the field; M3 carries it.
    use_trace: bool = False

    def __post_init__(self):
        if self.servable_seq_len is None:
            self.servable_seq_len = min(SPEC.max_seq_len, self.max_seq_len)
        assert (
            self.servable_seq_len <= self.max_seq_len
        ), f"servable_seq_len ({self.servable_seq_len}) exceeds the allocated capacity ({self.max_seq_len})"

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]

    # The engine's runtime contract reads `chunk_size` off this object.
    @property
    def chunk_size(self) -> int:
        return self.default_chunk_size


class TtPrefillRuntime:
    """Single-rank prefill lifecycle: build model -> (optionally allocate KV) -> build the indexed
    rope -> compile -> prefill_chunk."""

    def __init__(self, mesh_device, hf_config, state_dict: dict, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config

        self.chunk_sizes = resolve_chunk_sizes(
            config.default_chunk_size, config.additional_chunk_sizes, config.max_seq_len
        )
        self.max_chunk_size = self.chunk_sizes[0]
        assert config.topology in (
            ttnn.Topology.Ring,
            ttnn.Topology.Linear,
        ), f"sequence-parallel prefill supports Ring or Linear topology, got {config.topology}"

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False
        self.kv_cache = None
        self._on_layer_complete = None  # set by set_layer_ack_channel

        self._build_model(state_dict)
        if config.owns_kv_cache:
            self._allocate_kv_cache()
        self._build_indexed_rope()

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def _build_model(self, state_dict: dict) -> None:
        from models.demos.mistral_3_5_d_p.utils.general_utils import get_default_num_links

        from .ccl import CCLManager
        from .config import MeshConfig
        from .model import Model

        rows, cols = self.config.mesh_shape
        logger.info(
            f"Building Mistral-Medium-3.5 TtPrefillRuntime: num_layers={self.config.num_layers} "
            f"capacity={self.config.max_seq_len} servable={self.config.servable_seq_len} "
            f"chunk_sizes={self.chunk_sizes} num_users={self.config.num_users} mesh={self.config.mesh_shape}"
        )
        mesh_config = MeshConfig((rows, cols), tp=cols, tp_axis=self.config.tp_axis)
        ccl = CCLManager(
            self.mesh_device, num_links=get_default_num_links(self.mesh_device), topology=self.config.topology
        )
        self.ccl_manager = ccl
        self.model = Model(
            mesh_device=self.mesh_device,
            hf_config=self.hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=self.config.weight_cache_path,
            max_local_batch_size=1,
            max_seq_len=self.config.max_seq_len,
            sequence_parallel=True,
            num_layers=self.config.num_layers,
            first_layer_idx=self.config.first_layer_idx,
            weight_dtype=self.config.weight_dtype,
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        # ONE cache holding num_users * num_layers slots (user-major); each (user, layer) slot is
        # filled per chunk. KV heads shard on the TP cols; the sequence is SP-sharded block-cyclic.
        self.kv_cache = allocate_kv_cache(
            self.mesh_device,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            head_dim=self.hf_config.head_dim,
            cache_dtype=self.config.cache_dtype,
        )
        self.kv_cache_allocated = True

    def _build_indexed_rope(self) -> None:
        """Whole-cache indexed rope, one per supported chunk size (the block-cyclic period is
        size-specific). ``self.rope_indexed`` maps chunk_size -> [cos, sin]."""
        yarn = yarn_params_from_config(self.hf_config)
        self.rope_indexed = {
            size: build_indexed_rope(
                self.mesh_device,
                head_dim=self.hf_config.head_dim,
                max_seq_len=self.config.max_seq_len,
                chunk_size=size,
                sp_axis=self.config.sp_axis,
                **yarn,
            )
            for size in self.chunk_sizes
        }

    def _resolve_kv(self, kv_caches) -> MistralKVCache:
        """Resolve the cache from the (optional) caller arg: None (self-owned), a MistralKVCache, or
        the engine's ``KvCaches`` handle (index 0)."""
        if kv_caches is None:
            assert self.kv_cache is not None, "runtime has no KV cache (owns_kv_cache=False): pass kv_caches"
            return self.kv_cache
        if isinstance(kv_caches, MistralKVCache):
            return kv_caches
        return kv_caches[0]

    # ------------------------------------------------------------------
    # per-chunk
    # ------------------------------------------------------------------
    def make_chunk_input(self, token_ids: list, chunk_size: Optional[int] = None) -> ttnn.Tensor:
        """Build one chunk's device input for ``prefill_chunk``.

        On the first rank: SP-sharded uint32 ROW_MAJOR DRAM tokens of per-chip shape
        ``(1, 1, chunk_size // sp)`` — the SAME layout request-mode H2D delivery produces, so both
        paths feed one code path and ``prefill_chunk`` embeds on device. On a non-first pipeline rank
        the input is already a hidden-state activation (D2D), so a correctly-shaped placeholder is
        returned for compile warm-up.
        """
        chunk_size = self.config.default_chunk_size if chunk_size is None else chunk_size
        assert chunk_size in self.rope_indexed, f"chunk_size={chunk_size} not in supported {tuple(self.rope_indexed)}"
        sp = self.config.sp_factor
        s_local = chunk_size // sp
        if not self.config.is_first_rank:
            return ttnn.from_torch(
                torch.zeros(1, 1, s_local, self.hf_config.hidden_size),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        assert (
            len(token_ids) == chunk_size
        ), f"chunk input must be exactly chunk_size={chunk_size} tokens (pad the tail), got {len(token_ids)}"
        tokens = torch.tensor(token_ids, dtype=torch.int32).reshape(sp, 1, s_local)
        return ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=self.config.mesh_shape, dims=(self.config.sp_axis, None)
            ),
        )

    def compile(self, kv_caches=None) -> None:
        """Warm up the kernels by running zero-token chunks through ``prefill_chunk``.

        The first chunk exercises the cache-backed ring when the cache is larger than the chunk;
        an equal-sized one-shot request instead warms the all-gather fallback. When the config is
        multi-chunk, a second chunk is warmed too so its cache-growth runtime arguments are covered
        before the first served request. (Separate from the one-time empty-disk kernel-cache compile
        that only the very first run on a machine pays.)
        """
        assert self.model_built
        for chunk in self.chunk_sizes:
            ring = self.config.max_seq_len > chunk
            logger.info(
                f"TtPrefillRuntime.compile() — warming "
                f"{'2 cache-backed ring chunks' if ring else 'one all-gather fallback chunk'} of {chunk} tokens"
            )
            # prefill_chunk consumes (deallocates) its input, so build a fresh one per call.
            self.prefill_chunk(
                self.make_chunk_input([0] * chunk, chunk),
                kv_caches,
                slot_id=0,
                actual_start=0,
                actual_end=chunk,
                chunk_size=chunk,
            )
            if ring and 2 * chunk <= self.config.servable_seq_len:
                # actual_start > 0 drives the ring cache-read over the prefix just written.
                self.prefill_chunk(
                    self.make_chunk_input([0] * chunk, chunk),
                    kv_caches,
                    slot_id=0,
                    actual_start=chunk,
                    actual_end=2 * chunk,
                    chunk_size=chunk,
                )
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_caches=None,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        skip_lm_head: bool = True,
        get_last_token: int = -1,
        chunk_size: Optional[int] = None,
        request_id: int = -1,  # accepted for the common-runner contract; single-request prefill ignores it
        d2h_service=None,  # the device-side per-layer ack transport (see _layer_ack_callback)
        record_dev=None,  # accepted for the contract; the D1H record path is unused here
        metadata_msg=None,  # the ack record the D2H transport sends; required when d2h_service is set
    ) -> Optional[ttnn.Tensor]:
        """Prefill ONE chunk into user ``slot_id``'s slice of the KV cache.

        ``[actual_start, actual_end)`` is the absolute KV-position range of this chunk's REAL tokens.
        ``actual_start`` is the cache write offset; the last chunk's tail may be pad, so
        ``actual_end`` can be less than ``actual_start + chunk_size``. Call once per chunk, in order.

        Returns None under ``skip_lm_head`` — the populated cache is the output.
        """
        del record_dev, request_id
        assert self.model_built, "build the model before prefill_chunk()"
        chunk_size = self.config.default_chunk_size if chunk_size is None else chunk_size
        assert chunk_size in self.rope_indexed, f"chunk_size={chunk_size} not in supported {tuple(self.rope_indexed)}"
        kv = self._resolve_kv(kv_caches)
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        # Bound against the SERVABLE context, not the allocated capacity: the capacity is rounded up
        # to a whole number of chunks and its tail is not part of the model's context.
        assert actual_start + chunk_size <= self.config.servable_seq_len, (
            f"chunk at actual_start={actual_start} (+{chunk_size}) exceeds the servable context "
            f"{self.config.servable_seq_len}"
        )
        assert (
            actual_start < actual_end <= actual_start + chunk_size
        ), f"[actual_start={actual_start}, actual_end={actual_end}) not within one chunk of {chunk_size}"
        assert actual_start % ttnn.TILE_SIZE == 0, (
            f"actual_start ({actual_start}) must be tile-aligned; the block-cyclic cache write "
            f"assumes a tile-aligned chunk boundary"
        )

        if self.config.is_first_rank:
            x = self.model.embed(ttnn.reshape(input_tensor, [1, 1, input_tensor.shape[-1]]))
            ttnn.deallocate(input_tensor)
        else:
            x = input_tensor

        out = self.model.prefill_forward(
            x,
            rot_mats_global=self.rope_indexed[chunk_size],  # persistent; not deallocated per chunk
            kv_cache=kv,
            cached_len=actual_start,
            user_id=slot_id,
            get_last_token=get_last_token,
            skip_lm_head=skip_lm_head,
            indexed_rope=True,
            on_layer_complete=self._layer_ack_callback(d2h_service, metadata_msg),
        )
        if not self.config.is_last_rank:
            return out
        if skip_lm_head:
            if out is not None:
                out.deallocate(True)
            return None
        return out  # logits [1,1,chunk_local,vocab_shard], SP-sharded on seq / TP-sharded on vocab

    def _layer_ack_callback(self, d2h_service, metadata_msg):
        """The per-layer ack callback for this chunk, or None.

        TWO transports, and the engine picks between them — this runtime supports both, unlike the
        gpt-oss and M3 donors, which reject the D2H one:

          * **D2H (device-side)**, when the engine passes a ``d2h_service``: one
            ``outbound_socket_service_sync`` device op per layer, enqueued on the same command queue
            as the compute, so the ack costs no host sync. This is the path the SHARED
            ``test_producer_runner_e2e`` takes — it sets ``PREFILL_ENABLE_LAYER_ACK=1`` and
            ``PREFILL_LAYER_ACK_D2H=1`` on the runner unconditionally, so a runtime that rejects it
            cannot pass that test at all (the donors raise ``NotImplementedError`` here).
          * **host callback**, when ``set_layer_ack_channel`` has been registered and no
            ``d2h_service`` is passed: an ``inject(1)`` on the inter-process counter channel.

        Mutually exclusive: the engine's single-rank and D2H branches are disjoint, and wiring both
        would deliver acks twice, so this asserts rather than picking one.

        NOT implemented, and not needed by the ack itself: DeepSeek's block zeroes the KV pad window
        past ``actual_end`` before acking, so a migration of the final chunk does not move stale
        bytes. That matters for a REAL migration of a partially-filled chunk (Gate 2), not for the
        ack or for a PCC over ``[0, real_len)``. See README "Known gaps".
        """
        if d2h_service is None:
            return self._on_layer_complete
        assert self._on_layer_complete is None, (
            "both ack transports are wired (a D2H service AND a registered layer-ack channel); they "
            "are mutually exclusive and wiring both would emit every ack twice"
        )
        assert metadata_msg is not None, (
            "the D2H layer ack sends metadata_msg as its record, but the engine passed None; a chunk "
            "cannot be acked without one"
        )

        def on_layer_complete(layer_idx: int) -> None:
            del layer_idx  # the record identifies the chunk; the engine counts acks
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=metadata_msg)

        return on_layer_complete

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the per-layer LayerAck channel (engine-created and owned). ``prefill_chunk``
        bumps it once per layer; the scheduler/driver drains the delta."""
        assert self.compiled, "call compile() before set_layer_ack_channel()"

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete

    # ------------------------------------------------------------------
    # migration hooks (optional; the serving loop never calls these)
    # ------------------------------------------------------------------
    def kv_migration_base_address(self, kv_caches) -> int:
        """This rank's KV base DRAM address — the anchor the engine all-gathers to merge stages.
        Returns K's base; the multi-config table builder uses each tensor's own address."""
        return int(self._resolve_kv(kv_caches).k.buffer_address())

    def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
        """One ``KvCacheStage`` per migratable cache, in the order ``build_kv_chunk_table`` consumes
        their gathered layouts: k, then v.

        Written fresh — the donor (gpt_oss_d_p) does NOT implement this: it is single-rank and
        migrates through ``kv_migration_base_address`` plus the multi-config table, and rejects
        ``PREFILL_ENABLE_MIGRATION=1`` for more than one rank. The pattern is MiniMax-M3's
        ``tt_prefill_runtime.kv_migration_stages``, which emits THREE stages because of its MSA
        ``index_k``; dense GQA has two. The addresses come from this package's own allocator, which
        is what keeps them consistent with the table that describes them.

        Both caches share one layer-index space (every layer allocates a K and a V slot), so both
        stages carry the same range.
        """
        from models.demos.common.prefill.runners.migration import KvCacheStage

        kv = self._resolve_kv(kv_caches)
        first_layer_idx = self.config.first_layer_idx if first_layer_idx is None else int(first_layer_idx)
        num_my_layers = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        return [KvCacheStage(int(tensor.buffer_address()), first_layer_idx, num_my_layers) for tensor in (kv.k, kv.v)]

    def build_kv_chunk_table(
        self,
        kv_caches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers: Optional[int] = None,
        stage_layout=None,
        stage_layouts=None,
    ) -> str:
        """Build + serialize the multi-config KV chunk address table (k_h0..N, v_h0..N) to ``path``.

        Issues no comms — the engine publishes it. Single-rank: the table spans the whole model. The
        extra kwargs match the DeepSeek/PP runner call site and are ignored here.
        """
        del first_layer_idx, num_my_layers, stage_layout, stage_layouts  # single-rank: whole-model table
        from models.demos.mistral_3_5_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        config = self.config
        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kv_cache=self._resolve_kv(kv_caches),
            seq_len=config.max_seq_len,
            num_layers=config.num_layers,
            mesh_shape=config.mesh_shape,
            sp_axis=config.sp_axis,
            num_users=config.num_users,
            chunk_size=config.default_chunk_size,
            num_kv_heads=self.hf_config.num_key_value_heads,
            head_dim=self.hf_config.head_dim,
            path=path,
        )

    # ------------------------------------------------------------------
    # read-back / validation
    # ------------------------------------------------------------------
    def read_slot_kv(self, kv_caches, slot: int):
        """Read one slot's KV from device to host: ``[k, v]``, each
        ``[num_layers, num_kv_heads, seq_cache, head_dim]`` in the RAW on-device (block-cyclic)
        layout. Used by pairwise migration validation. ``DRAM_MEMORY_CONFIG`` on the slice is
        required — the cache is ND-sharded ROUND_ROBIN_1D."""
        kv = self._resolve_kv(kv_caches)
        num_layers = self.config.num_layers

        def block(tensor):
            shape = list(tensor.shape)
            sliced = ttnn.slice(
                tensor,
                [slot * num_layers, 0, 0, 0],
                [(slot + 1) * num_layers, shape[1], shape[2], shape[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.to_torch(
                sliced,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(2, 1), mesh_shape=tuple(self.mesh_device.shape)
                ),
            ).float()  # [num_layers, nkv (= TP cols), seq_cache, head_dim]
            ttnn.deallocate(sliced)
            return out

        return [block(kv.k), block(kv.v)]

    def gather_layer(self, slot_id: int, layer_idx: int, n_tokens: int, kv_caches=None, chunk_size=None):
        """Read one layer's device K/V back in NATURAL token order (un-rotating the block-cyclic SP
        layout).

        Returns ``(k, v)`` in DEVICE convention: K is Meta-RoPE swizzled over the full head_dim (the
        caller reconciles against the HF golden with ``hf_to_meta_head_permutation``), V is raw.
        Shapes ``[1, num_kv_heads, n_tokens, head_dim]``. No index_k — this is dense GQA.
        """
        kv = self._resolve_kv(kv_caches)
        sp = self.config.sp_factor
        cols = self.config.tp_factor  # KV head c lives on TP column c
        n_kv = self.hf_config.num_key_value_heads
        slot = slot_id * self.config.num_layers + layer_idx
        chunk_size = self.config.default_chunk_size if chunk_size is None else chunk_size
        # shard row -> natural global position (the inverse of the update_padded_kv_cache writer).
        positions = blockcyclic_positions(sp, chunk_size, self.config.max_seq_len)

        def gather(cache_tensor, col):
            shards = ttnn.get_device_tensors(cache_tensor)
            device_order = torch.cat([ttnn.to_torch(shards[r * cols + col])[slot, 0].float() for r in range(sp)], dim=0)
            natural = torch.empty_like(device_order)
            natural[positions] = device_order
            return natural[:n_tokens]

        k = torch.stack([gather(kv.k, c) for c in range(n_kv)], dim=0).unsqueeze(0)
        v = torch.stack([gather(kv.v, c) for c in range(n_kv)], dim=0).unsqueeze(0)
        return k, v

    def kv_cache_pcc_check(
        self,
        kv_caches=None,
        *,
        slot_id: int,
        n_chunks: int,
        trace_dir=None,
        first_layer_idx: int = 0,
        chunk_size=None,
        real_len=None,
        pt_path_override=None,
    ) -> float:
        """PCC the populated KV cache for ``slot_id`` against the golden trace; return the minimum
        per-layer PCC over K and V. A bring-up hook — never called in production serving.

        Golden layout: ``{trace_dir}/kv_cache/layer_N.safetensors`` with ``key_cache_layer_N``
        (post-RoPE K, HF concat-halves convention) and ``value_cache_layer_N`` (raw V), each
        ``[1, num_kv_heads, seq_len, head_dim]``. Dense GQA, so there is no index_k. The device K is
        Meta-RoPE swizzled over the full head_dim, so the golden K is permuted HF -> Meta first.

        ``n_chunks`` caps the comparison to what this run actually wrote; ``real_len`` caps it
        further to non-pad tokens.
        """
        from safetensors import safe_open

        from models.common.utility_functions import comp_pcc

        if pt_path_override is not None:
            raise NotImplementedError("no per-slot .pt golden path; use PREFILL_TRACE_DIR")
        from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir

        raw_trace = trace_dir or os.environ.get("PREFILL_TRACE_DIR")
        assert raw_trace, "kv_cache_pcc_check needs PREFILL_TRACE_DIR or trace_dir="
        trace_dir = resolve_trace_dir(raw_trace)
        token_ids = list(json.load(open(Path(trace_dir) / "metadata.json"))["token_ids"])

        chunk_size = self.config.default_chunk_size if chunk_size is None else chunk_size
        n_tokens = min(len(token_ids), n_chunks * chunk_size)
        if real_len is not None:
            n_tokens = min(n_tokens, int(real_len))
        assert n_tokens > 0, f"n_tokens=0 (n_chunks={n_chunks}, chunk_size={chunk_size})"

        permutation = hf_to_meta_head_permutation(
            self.hf_config.head_dim, getattr(self.hf_config, "rotary_dim", self.hf_config.head_dim)
        )
        kv_dir = Path(trace_dir) / "kv_cache"
        logger.info(
            f"[kv-pcc] per-layer K / V vs golden ({trace_dir}) over [0,{n_tokens}) "
            f"({self.config.num_layers} layers):"
        )
        min_k, min_v = 1.0, 1.0
        for local_layer in range(self.config.num_layers):
            global_layer = first_layer_idx + local_layer
            dev_k, dev_v = self.gather_layer(
                slot_id=slot_id,
                layer_idx=local_layer,
                n_tokens=n_tokens,
                kv_caches=kv_caches,
                chunk_size=chunk_size,
            )
            with safe_open(str(kv_dir / f"layer_{global_layer}.safetensors"), framework="pt") as handle:
                golden_k = handle.get_tensor(f"key_cache_layer_{global_layer}").float()[:, :, :n_tokens, :]
                golden_k = golden_k[..., permutation]  # HF -> Meta
                golden_v = handle.get_tensor(f"value_cache_layer_{global_layer}").float()[:, :, :n_tokens, :]
            pcc_k = float(comp_pcc(golden_k, dev_k, 0.0)[1])
            pcc_v = float(comp_pcc(golden_v, dev_v, 0.0)[1])
            min_k, min_v = min(min_k, pcc_k), min(min_v, pcc_v)
            logger.info(f"  layer {global_layer:>2}: K={pcc_k:.5f} V={pcc_v:.5f}")
        logger.info(f"[kv-pcc] min PCC across {self.config.num_layers} layers: K={min_k:.5f} V={min_v:.5f}")
        return min(min_k, min_v)

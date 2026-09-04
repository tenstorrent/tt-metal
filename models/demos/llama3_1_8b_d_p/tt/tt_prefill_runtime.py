# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B single-rank prefill runtime.

Copied from ``gpt_oss_d_p/tt/tt_prefill_runtime.py``. The shape that transfers is the one the
serving contract needs (``ADDING_A_PREFILL_MODEL.md`` §2): ``compile`` / ``make_chunk_input`` /
``prefill_chunk``, the ``_resolve_kv`` indirection so the engine can own the cache, and the chunk-range
assertions that make an out-of-contract chunk fail loudly instead of writing the wrong cache rows.

Lifecycle: build model -> (optionally allocate KV cache) -> build indexed rope -> compile ->
prefill_chunk per chunk, in order.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.llama3_1_8b_d_p.tt.attention import LlamaKVCache, allocate_kv_cache
from models.demos.llama3_1_8b_d_p.tt.ccl import CCLManager
from models.demos.llama3_1_8b_d_p.tt.config import MeshConfig
from models.demos.llama3_1_8b_d_p.tt.model import Model
from models.demos.llama3_1_8b_d_p.tt.rope import build_indexed_rope
from models.demos.llama3_1_8b_d_p.utils.general_utils import get_default_num_links


def resolve_chunk_sizes(default_chunk_size: int, additional_chunk_sizes: tuple, max_seq_len: int) -> tuple:
    """Supported chunk sizes, deduped, largest first. Each must divide ``max_seq_len``.

    The indexed rope tiles the whole cache in units of ``chunk_size // sp``; a chunk size that does not
    divide the cache leaves a partial final block whose rope rows describe the wrong positions.
    """
    sizes = tuple(sorted({default_chunk_size, *additional_chunk_sizes}, reverse=True))
    for cs in sizes:
        if max_seq_len % cs != 0:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be a multiple of every supported chunk size; "
                f"{cs} does not divide it (supported: {sizes})"
            )
    return sizes


@dataclass
class TtPrefillRuntimeConfig:
    """Resolved knobs for one runtime instance. Defaults are the prefill spec's values."""

    num_layers: int  # layers built by this runtime (== 32 for single-rank full model)
    max_seq_len: int  # per-user KV cache length in tokens; a multiple of every chunk size
    mesh_shape: tuple = (8, 4)  # (SP rows, TP cols) — spec topology.mesh_shape_per_stage
    default_chunk_size: int = 4096  # spec shapes.chunk_size (131072 / 4096 = 32 chunks, no tail)
    additional_chunk_sizes: tuple = ()
    num_users: int = 1
    sp_axis: int = 0
    tp_axis: int = 1
    # A plain-MESH Galaxy has no wrap links, so Linear is the default here (gpt-oss defaults to Ring
    # for torus pods). Mismatching this with the fabric config hangs rather than fails.
    topology: ttnn.Topology = ttnn.Topology.Linear
    attn_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None
    # True: the runtime allocates and owns self.kv_cache (the standalone harness path).
    # False: the engine owns the cache and passes it into every call (the adapter path).
    owns_kv_cache: bool = True
    # Pipeline-rank flags the common runner reads off runtime.config. Llama 3.1 8B is a single stage
    # (spec topology.pipeline_stages == 1), so both are True and first_layer_idx is 0.
    is_first_rank: bool = True
    is_last_rank: bool = True
    first_layer_idx: int = 0

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Single-rank Llama 3.1 8B prefill lifecycle."""

    def __init__(self, mesh_device, hf_config, state_dict: dict, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config

        self.chunk_sizes = resolve_chunk_sizes(
            config.default_chunk_size, config.additional_chunk_sizes, config.max_seq_len
        )
        self.max_chunk_size = self.chunk_sizes[0]
        assert config.topology in (ttnn.Topology.Ring, ttnn.Topology.Linear), f"unsupported topology {config.topology}"

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False
        self.kv_cache = None
        self._on_layer_complete = None  # set by set_layer_ack_channel

        self._build_model(state_dict)
        if config.owns_kv_cache:
            self._allocate_kv_cache()
        self._build_indexed_rope()

    # ---------------------------------------------------------------- build

    def _build_model(self, state_dict: dict) -> None:
        cfg = self.config
        self.mesh_config = MeshConfig(cfg.mesh_shape, tp=cfg.tp_factor, tp_axis=cfg.tp_axis)
        self.ccl_manager = CCLManager(
            self.mesh_device, num_links=get_default_num_links(self.mesh_device), topology=cfg.topology
        )
        self.num_kv_heads_local = self.hf_config.num_key_value_heads // self.mesh_config.tp
        assert self.num_kv_heads_local >= 1, (
            f"TP={self.mesh_config.tp} exceeds num_key_value_heads={self.hf_config.num_key_value_heads}; "
            f"KV-head replication is not implemented (spec known_risks)"
        )
        self.model = Model(
            self.mesh_device,
            self.hf_config,
            state_dict,
            ccl_manager=self.ccl_manager,
            tensor_cache_path=cfg.weight_cache_path,
            mesh_config=self.mesh_config,
            max_seq_len=cfg.max_seq_len,
            attn_weight_dtype=cfg.attn_weight_dtype,
            mlp_weight_dtype=cfg.mlp_weight_dtype,
            sequence_parallel=True,
            num_layers=cfg.num_layers,
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        cfg = self.config
        self.kv_cache = allocate_kv_cache(
            self.mesh_device,
            num_layers=cfg.num_layers,
            max_seq_len=cfg.max_seq_len,
            sp_axis=cfg.sp_axis,
            num_users=cfg.num_users,
            head_dim=self.hf_config.head_dim,
            num_kv_heads_local=self.num_kv_heads_local,
            cache_dtype=cfg.cache_dtype,
        )
        self.kv_cache_allocated = True

    def _build_indexed_rope(self) -> None:
        """One whole-cache block-cyclic SP cos/sin PER supported chunk size, built once and reused.

        The block-cyclic reorder is keyed on ``chunk_size // sp``, so a runtime serving several chunk
        sizes needs one rope per size — sharing them would give the smaller size the larger size's
        block boundaries.
        """
        cfg = self.config
        self.rope_indexed = {
            chunk: build_indexed_rope(
                self.mesh_device,
                head_dim=self.hf_config.head_dim,
                max_seq_len=cfg.max_seq_len,
                chunk_size=chunk,
                sp_axis=cfg.sp_axis,
            )
            for chunk in self.chunk_sizes
        }

    # ------------------------------------------------------------ chunk I/O

    def _resolve_kv(self, kv_caches) -> LlamaKVCache:
        """Resolve the cache from the caller arg: None (self-owned), a LlamaKVCache, or a sequence."""
        if kv_caches is None:
            assert self.kv_cache is not None, "runtime has no KV cache (owns_kv_cache=False): pass kv_caches"
            return self.kv_cache
        if isinstance(kv_caches, LlamaKVCache):
            return kv_caches
        return kv_caches[0]

    def make_chunk_input(self, token_ids: list, chunk_size: Optional[int] = None) -> ttnn.Tensor:
        """Build one chunk's device input for ``prefill_chunk``.

        First rank: SP-sharded uint32 ROW_MAJOR DRAM tokens, per-chip shape ``(1, 1, chunk_size // sp)``
        — the same layout request-mode H2D delivers, so the harness and the served path share one code
        path. ``prefill_chunk`` embeds on device. Non-first ranks get a placeholder activation of the
        right spec for warm-up (Llama is single-stage, so that branch is only reachable if a caller
        constructs a multi-rank config).
        """
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
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
        tok = torch.tensor(token_ids, dtype=torch.int32).reshape(sp, 1, s_local)
        return ttnn.from_torch(
            tok,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=self.config.mesh_shape, dims=(self.config.sp_axis, None)
            ),
        )

    def compile(self, kv_caches=None) -> None:
        """JIT-warm the kernels by running zero-token chunks through ``prefill_chunk``.

        Warms EVERY supported chunk size, and — when the cache is bigger than one chunk — a second
        chunk at ``actual_start > 0`` so the cache-backed ring path's growth arguments are compiled
        before the first served request rather than inside it.
        """
        assert self.model_built
        for chunk in self.chunk_sizes:
            ring = self.config.max_seq_len > chunk
            logger.info(
                f"Llama-3.1-8B TtPrefillRuntime.compile() — warming "
                f"{'2 cache-backed ring chunks' if ring else 'one all-gather fallback chunk'} of {chunk} tokens"
            )
            # prefill_chunk consumes its input tensor, so build a fresh one per call.
            self.prefill_chunk(
                self.make_chunk_input([0] * chunk, chunk),
                kv_caches,
                slot_id=0,
                actual_start=0,
                actual_end=chunk,
                chunk_size=chunk,
            )
            if ring:
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
        chunk_size: Optional[int] = None,
        # The engine passes these on every call. A single-stage, single-rank runtime uses none of
        # them, but the contract is to ACCEPT them — the runner does not inspect the signature, so a
        # missing keyword is a TypeError in the serving loop rather than at build time.
        request_id: int = -1,  # engine chunk counter; only the pipelined layer-completion sink reads it
        d2h_service=None,  # this runtime emits LayerAcks through set_layer_ack_channel, not D2H
        record_dev=None,  # the D1H record path is unused here
        metadata_msg=None,  # raw socket metadata, forwarded verbatim to the next PIPELINE rank; there is none
    ) -> Optional[ttnn.Tensor]:
        """Prefill ONE chunk into user ``slot_id``'s slice of the KV cache. Call in order.

        ``[actual_start, actual_end)`` is the absolute KV-position range of this chunk's REAL tokens:
        ``actual_start`` is the cache write offset (the valid prefix already cached), and the final
        chunk's tail may be pad, so ``actual_end < actual_start + chunk_size``. The asserts below are
        the contract — an out-of-range chunk would otherwise write another user's slot, or write past
        the cache, with no error.
        """
        if d2h_service is not None:
            raise NotImplementedError(
                "this runtime emits layer acks via set_layer_ack_channel, not the D2H path; "
                "run with PREFILL_ENABLE_LAYER_ACK=0 or wire the D2H ack in"
            )
        assert self.model_built, "build the model before prefill_chunk()"
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        assert chunk_size in self.rope_indexed, f"chunk_size={chunk_size} not in supported {tuple(self.rope_indexed)}"
        kv = self._resolve_kv(kv_caches)
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        assert (
            actual_start + chunk_size <= self.config.max_seq_len
        ), f"chunk at actual_start={actual_start} (+{chunk_size}) exceeds per-user cache {self.config.max_seq_len}"
        assert (
            actual_start < actual_end <= actual_start + chunk_size
        ), f"[actual_start={actual_start}, actual_end={actual_end}) is not within one chunk of {chunk_size}"
        assert (
            actual_start % ttnn.TILE_SIZE == 0
        ), f"actual_start ({actual_start}) must be tile-aligned; the block-cyclic write assumes it"

        if self.config.is_first_rank:
            x = self.model.embed(input_tensor)
            ttnn.deallocate(input_tensor)
        else:
            x = input_tensor

        out = self.model.forward_layers(
            x,
            self.rope_indexed[chunk_size],  # persistent; never deallocated per chunk
            kv_cache=kv,
            user_id=slot_id,
            cached_len=actual_start,
            indexed_rope=True,
            on_layer_complete=self._on_layer_complete,
        )
        if not self.config.is_last_rank:
            return out
        if skip_lm_head:
            out.deallocate(True)
            return None
        return self.model.lm_head(out)

    # --------------------------------------------------------- engine hooks

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the per-layer LayerAck channel (engine-created and owned).

        ``forward_layers`` bumps it once per layer through ``on_layer_complete``, which is how the
        scheduler learns a layer's KV is final and may be migrated. The channel is a COUNTER, not a
        per-layer message: the contract is ``inject(1)`` per completed layer and the reader drains the
        delta, so the layer index is implied by the count, not passed.

        Without this, the shared producer's ``PREFILL_PRODUCER_CHECK_PCC=1`` read would race the
        runner's prefill — the H2D push returning does not mean the layers are done.
        """
        assert self.compiled, "call compile() before set_layer_ack_channel()"
        self._layer_ack_channel = layer_ack_channel

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete if layer_ack_channel is not None else None

    def kv_migration_base_address(self, kv_caches=None) -> int:
        """This rank's KV base DRAM address — the anchor the engine merges pipeline stages around.

        K is the base; V is a separate tensor and is described as its own config in the chunk table
        (see ``runners/kv_chunk_table.py``), which is why a single base address is not enough on its
        own and ``kv_migration_stages`` is the method the engine actually uses for this model.
        """
        kv = self._resolve_kv(kv_caches)
        return int(kv.k.buffer_address())

    def kv_migration_stages(self, kv_caches=None, first_layer_idx=None, num_my_layers=None):
        """One ``KvCacheStage`` per migratable cache (K, then V) — see ``runners/kv_chunk_table.py``.

        Two configs, and ``"k"`` sorts before ``"v"``, so config_id 0 is K and 1 is V. That ordering is
        the src<->dst contract (spec ``interfaces.decode.config_names_and_order``); protobuf rebuilds
        configs through a ``std::map``, so the names are what fixes the order, not insertion.
        """
        from models.demos.llama3_1_8b_d_p.tt.runners.kv_chunk_table import build_migration_stages

        kv = self._resolve_kv(kv_caches)
        return build_migration_stages(
            kv,
            mesh_device=self.mesh_device,
            first_layer_idx=self.config.first_layer_idx if first_layer_idx is None else first_layer_idx,
            num_layers=self.config.num_layers if num_my_layers is None else num_my_layers,
        )

    def build_kv_chunk_table(self, kv_caches, path: str) -> str:
        """Build + serialize the KV chunk address table to ``path``. Issues no comms."""
        from models.demos.llama3_1_8b_d_p.tt.runners.kv_chunk_table import serialize_table

        return serialize_table(self, self._resolve_kv(kv_caches), path)

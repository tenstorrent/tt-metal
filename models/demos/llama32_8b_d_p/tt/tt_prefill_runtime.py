# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B single-rank chunked-prefill runtime — the engine-facing lifecycle.

Satisfies the runtime half of the common prefill contract
(``models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`` §2): ``mesh_device``, a ``config``
exposing ``chunk_size`` / ``max_seq_len`` / ``first_layer_idx`` / ``is_first_rank`` /
``is_last_rank``, then ``compile(kv_cache)`` -> ``make_chunk_input(token_ids)`` ->
``prefill_chunk(input, kv_cache, *, slot_id, actual_start, actual_end, request_id=0)``.

Template: ``models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:46`` (``resolve_chunk_sizes``),
``:59`` (``TtPrefillRuntimeConfig``), ``:88`` / ``:92`` (``sp_factor`` / ``tp_factor``), ``:96``
(the class), ``:204`` (``make_chunk_input``), ``:250`` (``compile``), ``:288``
(``prefill_chunk``), ``:359`` (``set_layer_ack_channel``), ``:370``
(``kv_migration_base_address``), ``:432`` (``gather_layer``), ``:505``
(``kv_cache_pcc_check``).

**Cache ownership: the engine's, always.** ``owns_kv_cache`` defaults to ``False`` here where the
template defaults to ``True`` (``DEC-055``) — the contract says the engine allocates via the
adapter's ``allocate_kv_cache`` and passes the cache into every call, and a runtime whose default
is to allocate its own is one forgotten keyword away from filling a cache nobody reads.

**Deletions vs the template** (``03_OUTLINE.md`` §3.18): every MoE field
(``use_ep_moe``, ``expert_weight_dtype``, ``ep_seq_len_per_chip``), and
``_build_indexed_rope``'s YaRN specifics — replaced by ``tt/rope.build_indexed_rope``, which reads
theta and the llama3 scaling off the ``LlamaHFConfig`` instead of ``getattr(cfg, "rope_theta",
150000.0)`` (``tt_prefill_runtime.py:185``). That ``getattr`` default is the highest-severity
silent-wrongness trap in this bring-up: on transformers 5.12.1 ``rope_theta`` is **not** an
attribute, so the default is what gets used, and a wrong theta is a RoPE that is wrong at every
position with no exception anywhere (Appendix F.2, ``R-014``).

**Two loud single-device blockers, both by design, both P8's to lift** (``DEC-056``):

1. ``actual_start > 0`` needs the chunked cache-read attention, which
   ``tt/attention/prefill.py:218`` refuses on a single device: plain ``is_causal`` SDPA assumes Q
   row 0 aligns with K row 0, so with a non-empty cache it is off by ``cached_len`` and silently
   wrong. :meth:`prefill_chunk` checks this *before* touching the device so the message names the
   cause instead of arriving from three frames down.
2. The KV cache holds exactly **one** KV head per chip
   (``tt/attention/kv_cache.py:130`` allocates ``[num_users*num_layers, 1, seq_local, head_dim]``),
   so ``TP`` must equal ``num_key_value_heads`` (8). At TP<8 the model's attention produces
   ``8/TP`` local KV heads and ``update_padded_kv_cache`` dies with
   ``TT_FATAL ... cache and input num-heads dim must match`` — measured, ``R-027``. Exact assert:
   ``ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230``.
   :meth:`_resolve_kv` turns that into a message that says which knob is wrong.

Both are re-stated as asserts rather than trusted to the callee because the failure they replace is
either a wrong number (1) or a C++ ``TT_FATAL`` (2).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions

from .attention.kv_cache import LlamaKVCache, allocate_kv_cache
from .rope import build_indexed_rope


def resolve_chunk_sizes(default_chunk_size: int, additional_chunk_sizes: tuple, max_seq_len: int) -> tuple:
    """Supported chunk sizes, deduped, largest first; each must divide ``max_seq_len``.

    Copied from ``models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:46``. The divisibility is not
    cosmetic: ``build_indexed_rope`` block-cyclic-reorders the whole-cache table with a period of
    ``chunk_size // sp``, so a chunk size that does not tile the cache produces a table whose rows
    stop lining up with the cache rows partway through (``tt/rope.py:151``).
    """
    sizes = tuple(sorted({default_chunk_size, *additional_chunk_sizes}, reverse=True))
    for size in sizes:
        if max_seq_len % size != 0:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be a multiple of every supported chunk size; "
                f"{size} does not divide it (supported: {sizes})"
            )
    return sizes


def _dense_sp_is_implemented() -> bool:
    """Is ``tt/attention/dense_sp.dense_sp_attention`` a real port yet, or still the P5 stub?

    Probed, not hard-coded, so this runtime needs no edit when P8 lands the port. The probe calls
    it with **no arguments**: the stub's signature is ``(*args, **kwargs)`` and its very first
    statement raises ``NotImplementedError``, while any real implementation takes named parameters
    and therefore raises ``TypeError`` before touching a device. Nothing is allocated either way.
    """
    from .attention.dense_sp import dense_sp_attention

    try:
        dense_sp_attention()
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 - a TypeError here means a real signature, i.e. implemented
        return True
    return True


@dataclass
class TtPrefillRuntimeConfig:
    """Runtime configuration. ``chunk_size`` is the engine-contract name; see the property below."""

    num_layers: int  # layers this rank builds (== 32 for single-rank Llama-3.1-8B)
    max_seq_len: int  # per-user KV-cache capacity in tokens; a multiple of every chunk size
    mesh_shape: tuple = (4, 8)  # (SP rows, TP cols) on the Blackhole Galaxy — DEC-002
    default_chunk_size: int = 8192  # tokens per prefill_chunk() call; 8k divides the 128k context
    additional_chunk_sizes: tuple = ()  # other sizes this instance can serve (own rope table each)
    num_users: int = 1  # independent cache slots (user-major batch)
    sp_axis: int = 0
    tp_axis: int = 1
    topology: ttnn.Topology = ttnn.Topology.Ring
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None
    # DEC-055: the engine owns the cache. True is the standalone-harness escape hatch only.
    owns_kv_cache: bool = False
    # Pipeline-parallel rank flags the common runner reads off `runtime.config`. Single-rank =>
    # both True, first_layer_idx 0. `first_layer_idx` is the GLOBAL index of this rank's first
    # layer and is what offsets the golden-trace layer numbering.
    is_first_rank: bool = True
    is_last_rank: bool = True
    first_layer_idx: int = 0
    # P8 flips this to True once tt/attention/dense_sp.py is a real port; it is the single switch
    # that turns on both SP prefill and the chunked cache-read path (see
    # TtPrefillRuntime._chunked_read_supported). Left False so the runtime cannot half-enable SP.
    sequence_parallel: bool = False

    @property
    def chunk_size(self) -> int:
        """The engine contract's name for ``default_chunk_size`` (``DEC-054``).

        ``ADDING_A_PREFILL_MODEL.md`` §2 requires ``config.chunk_size``; the template's dataclass
        only has ``default_chunk_size`` and its adapter bridges the two at the call site
        (``models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:132``). Exposing both means the
        engine reads the name it documents and the multi-size ergonomics survive, with no
        possibility of the two drifting.
        """
        return self.default_chunk_size

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Build model -> build the indexed rope -> ``compile`` -> ``prefill_chunk`` per chunk."""

    def __init__(self, mesh_device, hf_config, state_dict: dict, config: TtPrefillRuntimeConfig):
        """
        Args:
            mesh_device: the open ttnn mesh device. The engine synchronizes and closes it.
            hf_config: a ``LlamaHFConfig`` from ``tt/model_config.llama_hf_config`` (``DEC-009``).
                A raw ``dict`` or a ``transformers`` object is refused: theta and the llama3
                scaling must come from the one normalising constructor, or the
                ``getattr(cfg, "rope_theta", DEFAULT)`` trap reappears (``R-014``).
            state_dict: the full HF-layout checkpoint, or ``{}`` for cache-only mode (which needs
                ``config.weight_cache_path``).
            config: :class:`TtPrefillRuntimeConfig`.
        """
        for attr in ("head_dim", "num_key_value_heads", "rope_theta", "rope_type"):
            assert hasattr(hf_config, attr), (
                f"hf_config is missing {attr!r} — pass the LlamaHFConfig from "
                f"tt/model_config.llama_hf_config(), not a raw dict or a transformers LlamaConfig "
                f"(on transformers 5.12.1 the latter has no .rope_theta at all, and a getattr "
                f"default silently substitutes the wrong theta — R-014)"
            )
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config

        # Config-internal consistency first, then config-vs-device: a config that is wrong on its
        # own terms should say so regardless of which mesh happens to be open.
        assert not (config.sequence_parallel and not _dense_sp_is_implemented()), (
            "sequence_parallel=True but tt/attention/dense_sp.dense_sp_attention is still the P5 "
            "stub, so every layer would raise NotImplementedError mid-forward after the KV cache "
            "had already been written. SP prefill is P8's deliverable (DEC-056)."
        )
        assert not (config.sequence_parallel and config.sp_factor == 1), (
            f"sequence_parallel=True with sp={config.sp_factor}: there is no sequence axis to "
            f"shard on mesh_shape={tuple(config.mesh_shape)}"
        )
        assert tuple(mesh_device.shape) == tuple(config.mesh_shape), (
            f"config.mesh_shape={tuple(config.mesh_shape)} but the open device is "
            f"{tuple(mesh_device.shape)}; the weight cache and the rope tables are both mesh-shape "
            f"specific (DEC-048), so this mismatch is silently wrong, not merely inconsistent"
        )
        assert config.topology in (
            ttnn.Topology.Ring,
            ttnn.Topology.Linear,
        ), f"topology must be Ring or Linear, got {config.topology}"
        self.chunk_sizes = resolve_chunk_sizes(
            config.default_chunk_size, config.additional_chunk_sizes, config.max_seq_len
        )
        self.max_chunk_size = self.chunk_sizes[0]
        for size in self.chunk_sizes:
            # Mirrors build_indexed_rope's own assert (tt/rope.py:148) and the engine doc's
            # CHUNK_SIZE % (SP*32) == 0. Checked here too so a bad config fails before the model
            # spends a minute loading 8 B parameters.
            assert size % (ttnn.TILE_SIZE * config.sp_factor) == 0, (
                f"chunk_size {size} must be a multiple of TILE_SIZE*sp " f"({ttnn.TILE_SIZE * config.sp_factor})"
            )

        self.model_built = False
        self.compiled = False
        self.kv_cache = None
        self._on_layer_complete = None

        self._build_model(state_dict)
        if config.owns_kv_cache:
            self._allocate_kv_cache()
        self._build_indexed_rope()

    # -------------------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------------------
    def _build_model(self, state_dict: dict) -> None:
        from models.demos.llama32_8b_d_p.utils.general_utils import get_default_num_links

        from .ccl import CCLManager
        from .config import MeshConfig
        from .model import Model

        cfg = self.config
        rows, cols = cfg.mesh_shape
        logger.info(
            f"[llama32_8b_d_p] TtPrefillRuntime: num_layers={cfg.num_layers} "
            f"max_seq_len={cfg.max_seq_len} chunk_sizes={self.chunk_sizes} "
            f"num_users={cfg.num_users} mesh_shape={cfg.mesh_shape} sp={cfg.sp_factor} tp={cfg.tp_factor}"
        )
        mesh_config = MeshConfig((rows, cols), tp=cfg.tp_factor, tp_axis=cfg.tp_axis)
        ccl = CCLManager(self.mesh_device, num_links=get_default_num_links(self.mesh_device), topology=cfg.topology)
        self.mesh_config = mesh_config
        self.ccl_manager = ccl
        self.model = Model(
            self.mesh_device,
            self.hf_config,
            state_dict,
            mesh_config=mesh_config,
            ccl_manager=ccl,
            max_seq_len=cfg.max_seq_len,
            num_layers=cfg.num_layers,
            weight_dtype=cfg.weight_dtype,
            tensor_cache_path=cfg.weight_cache_path,
            sequence_parallel=cfg.sequence_parallel,
            # Scheme A (DEC-018): the TP tail is an all-reduce and the residual stays full-width.
            # DecoderLayer refuses scatter_output=True outright (DEC-049), so this is the only
            # value that is not an error.
            scatter_output=False,
            # Prefill's product is the KV cache; the ~1 GiB LM head is only needed for a top-1
            # check, and `skip_lm_head=False` is refused rather than silently returning hidden
            # states (tt/model.py:374).
            with_lm_head=False,
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        """Standalone-harness escape hatch (``owns_kv_cache=True``). The engine path never runs it."""
        self.kv_cache = allocate_kv_cache(
            self.mesh_device,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            head_dim=self.hf_config.head_dim,
            cache_dtype=self.config.cache_dtype,
        )

    def _build_indexed_rope(self) -> None:
        """One whole-cache indexed rope table per supported chunk size.

        The block-cyclic period is ``chunk_size // sp``, so the table is chunk-size specific and
        cannot be shared between sizes. Persistent for the runtime's lifetime — ``prefill_chunk``
        must not deallocate it (``tt/rope.py:141``).
        """
        self.rope_indexed = {
            size: build_indexed_rope(
                self.mesh_device,
                self.hf_config,
                max_seq_len=self.config.max_seq_len,
                chunk_size=size,
                sp_axis=self.config.sp_axis,
            )
            for size in self.chunk_sizes
        }

    # -------------------------------------------------------------------------------------
    # Cache plumbing
    # -------------------------------------------------------------------------------------
    def _resolve_kv(self, kv_cache) -> LlamaKVCache:
        """Resolve a ``LlamaKVCache`` from the caller's argument, and check the TP invariant.

        Accepts ``None`` (use the self-owned cache), a ``LlamaKVCache``, or the engine's
        ``KvCaches`` sequence (index 0 — Llama is dense GQA and has exactly one cache).
        """
        if kv_cache is None:
            assert self.kv_cache is not None, (
                "this runtime does not own a KV cache (owns_kv_cache=False, the engine contract): "
                "pass the cache the engine allocated into compile()/prefill_chunk()"
            )
            resolved = self.kv_cache
        elif isinstance(kv_cache, LlamaKVCache):
            resolved = kv_cache
        else:
            resolved = kv_cache[0]
            assert isinstance(
                resolved, LlamaKVCache
            ), f"kv_cache[0] is {type(resolved).__name__}, expected LlamaKVCache"

        n_kv = self.hf_config.num_key_value_heads
        assert self.config.tp_factor == n_kv, (
            f"the packed KV cache holds exactly ONE KV head per chip "
            f"(tt/attention/kv_cache.py:130 allocates [.., 1, seq_local, head_dim]), so TP must "
            f"equal num_key_value_heads. Got TP={self.config.tp_factor} at mesh_shape="
            f"{tuple(self.config.mesh_shape)} with num_key_value_heads={n_kv}: attention would "
            f"produce {n_kv // max(self.config.tp_factor, 1)} local KV heads and "
            f"update_padded_kv_cache would die with 'cache and input num-heads dim must match' "
            f"(ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230). Use a mesh whose TP axis is "
            f"{n_kv} wide (R-027)."
        )
        assert (
            resolved.num_layers >= self.config.num_layers
        ), f"cache has {resolved.num_layers} layer slots, runtime builds {self.config.num_layers}"
        assert (
            resolved.max_seq_len >= self.config.max_seq_len
        ), f"cache capacity {resolved.max_seq_len} < runtime max_seq_len {self.config.max_seq_len}"
        assert (
            resolved.head_dim == self.hf_config.head_dim
        ), f"cache head_dim {resolved.head_dim} != model head_dim {self.hf_config.head_dim}"
        return resolved

    # -------------------------------------------------------------------------------------
    # The engine contract
    # -------------------------------------------------------------------------------------
    def make_chunk_input(self, token_ids, chunk_size: Optional[int] = None) -> ttnn.Tensor:
        """One chunk's device input for :meth:`prefill_chunk`.

        On the first rank: SP-sharded ``uint32`` ROW_MAJOR DRAM token ids of per-chip shape
        ``[1, 1, 1, chunk_size // sp]`` — the same layout request-mode H2D delivers, so the two
        paths share one code path; :meth:`prefill_chunk` embeds on device. On a non-first pipeline
        rank the real input arrives as an activation over D2D, so this returns a correctly-specced
        zero placeholder for compile warm-up.

        The leading ``[1, 1, 1, ...]`` (rather than the template's ``[sp, 1, s_local]``) is what
        ``tt/embedding.py:87`` documents as its input: ``ttnn.embedding`` gathers rows and tilizes
        the output, and the 4-D form is what keeps the result ``[1, 1, s_local, hidden]`` with no
        reshape.
        """
        chunk_size = self.config.default_chunk_size if chunk_size is None else chunk_size
        assert chunk_size in self.rope_indexed, (
            f"chunk_size={chunk_size} is not one of this runtime's supported sizes "
            f"{tuple(self.rope_indexed)}; add it to config.additional_chunk_sizes"
        )
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

        token_ids = list(token_ids)
        assert len(token_ids) == chunk_size, (
            f"a chunk input must be exactly chunk_size={chunk_size} token ids (pad the tail); " f"got {len(token_ids)}"
        )
        tokens = torch.tensor(token_ids, dtype=torch.int32).reshape(1, 1, 1, chunk_size)
        if sp == 1:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        else:
            dims = [None, None]
            dims[self.config.sp_axis] = 3  # sequence across the SP rows, replicated across TP
            mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=tuple(dims)
            )
        return ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def compile(self, kv_cache=None) -> None:
        """Warm up / JIT-compile the per-chunk program so the served loop pays no first-run cost.

        One zero-token chunk per supported size. A *second* chunk (at ``actual_start = chunk``) is
        warmed only when the chunked cache-read path is actually available — see
        :meth:`prefill_chunk`'s ``sp == 1`` refusal — because warming a program that the next call
        will refuse to run buys nothing and turns ``compile()`` into the place the blocker
        surfaces.
        """
        assert self.model_built, "build the model before compile()"
        kv = self._resolve_kv(kv_cache)
        for size in self.chunk_sizes:
            multi_chunk = self.config.max_seq_len > size and self._chunked_read_supported()
            logger.info(
                f"[llama32_8b_d_p] compile(): warming {'2 chunks' if multi_chunk else '1 chunk'} " f"of {size} tokens"
            )
            self.prefill_chunk(
                self.make_chunk_input([0] * size, size),
                kv,
                slot_id=0,
                actual_start=0,
                actual_end=size,
                chunk_size=size,
            )
            if multi_chunk:
                self.prefill_chunk(
                    self.make_chunk_input([0] * size, size),
                    kv,
                    slot_id=0,
                    actual_start=size,
                    actual_end=2 * size,
                    chunk_size=size,
                )
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def _chunked_read_supported(self) -> bool:
        """Is the cache-read attention path (``cached_len > 0``) available on this configuration?

        It needs **all three**: ``sp > 1`` (a single SP row has no ring to read the prefix over),
        ``sequence_parallel=True`` so ``tt/attention/prefill.py:195`` selects the SP branch instead
        of the ``cached_len > 0`` refusal at ``:218``, and a real
        ``tt/attention/dense_sp.dense_sp_attention`` — which is still a raising stub
        (``tt/attention/dense_sp.py:43``, P8). ``_dense_sp_is_implemented`` probes the stub rather
        than hard-coding ``False``, so this predicate becomes ``True`` the moment P8 lands the port
        and nothing here needs editing (``DEC-056``).
        """
        return self.config.sp_factor > 1 and self.config.sequence_parallel and _dense_sp_is_implemented()

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_cache=None,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        chunk_size: Optional[int] = None,
        skip_lm_head: bool = True,
        get_last_token: int = -1,
        d2h_service=None,
        record_dev=None,
    ) -> Optional[ttnn.Tensor]:
        """Prefill ONE chunk into user ``slot_id``'s slice of ``kv_cache``. Call in order.

        ``[actual_start, actual_end)`` is the absolute KV-position range of this chunk's **real**
        tokens: ``actual_start`` is the cache write offset (the valid prefix already cached) and the
        last chunk's tail may be pad, so ``actual_end`` can be below ``actual_start + chunk_size``.

        Returns this rank's output hidden state on a non-last pipeline rank, else ``None`` — the
        populated cache is the output (``skip_lm_head=True``, and ``tt/model.py`` returns the
        **post-final-norm** hidden state on that path, ``DEC-043``).

        ``request_id`` is accepted and ignored: the runner always passes it, and only the pipelined
        layer-completion sink consumes it (to build a globally-dense
        ``seq = request_id * num_layers + layer_idx``); a single-rank LayerAck channel carries no
        payload.
        """
        if d2h_service is not None:
            raise NotImplementedError(
                "llama32_8b_d_p emits layer acks through set_layer_ack_channel, not the D2H path; "
                "run with PREFILL_ENABLE_LAYER_ACK=0 or wire a D2H ack into this runtime"
            )
        del record_dev  # accepted for the common-runner contract; the D1H record path is unused
        assert self.model_built, "build the model before prefill_chunk()"
        chunk_size = self.config.default_chunk_size if chunk_size is None else chunk_size
        assert chunk_size in self.rope_indexed, f"chunk_size={chunk_size} not in supported {tuple(self.rope_indexed)}"
        # The pure-contract checks run BEFORE `_resolve_kv` on purpose: they are cheaper, they are
        # about the caller's arguments rather than the cache, and the most specific complaint
        # should win. `_resolve_kv` (which also enforces the TP == num_kv_heads invariant) runs
        # last, immediately before the cache is used.
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        assert actual_start + chunk_size <= self.config.max_seq_len, (
            f"chunk at actual_start={actual_start} (+{chunk_size}) exceeds the per-user cache "
            f"capacity {self.config.max_seq_len}"
        )
        assert actual_start < actual_end <= actual_start + chunk_size, (
            f"[actual_start={actual_start}, actual_end={actual_end}) is not within one chunk of " f"{chunk_size}"
        )
        # write_kv_chunk's own precondition, hoisted: the block-cyclic per-device write offset
        # assumes a tile-aligned boundary (tt/attention/kv_cache.py:192).
        assert actual_start % ttnn.TILE_SIZE == 0, (
            f"actual_start={actual_start} must be tile-aligned (a multiple of {ttnn.TILE_SIZE}); "
            f"update_padded_kv_cache asserts kv_actual_global % 32 == 0"
        )
        if actual_start > 0 and not self._chunked_read_supported():
            raise NotImplementedError(
                f"llama32_8b_d_p: chunked prefill past the first chunk (actual_start="
                f"{actual_start} > 0) needs the cache-read attention, which is not available at "
                f"sp={self.config.sp_factor}. Q is this chunk at global offset {actual_start} "
                f"while K/V span [0, {actual_end}), so the single-device is_causal SDPA (which "
                f"assumes Q row 0 aligns with K row 0) is off by {actual_start} and SILENTLY "
                f"WRONG — tt/attention/prefill.py:218 refuses it for that reason. The available "
                f"paths are the SP ring-joint SDPA over the block-cyclic cache "
                f"(tt/attention/dense_sp.py, P8) or a paged chunked_scaled_dot_product_attention. "
                f"Chunk 0 and one-shot prefill both work; see DEC-056 / R-028."
            )
        kv = self._resolve_kv(kv_cache)

        if self.config.is_first_rank:
            x = self.model.embedding(input_tensor)
            input_tensor.deallocate(True)
        else:
            x = input_tensor

        out = self.model.prefill_forward(
            x,
            rot_mats_global=self.rope_indexed[chunk_size],  # persistent; never deallocated here
            kv_cache=kv,
            cached_len=actual_start,
            user_id=slot_id,
            get_last_token=get_last_token,
            skip_lm_head=skip_lm_head,
            indexed_rope=True,
            on_layer_complete=self._on_layer_complete,
        )
        if not self.config.is_last_rank:
            return out
        if skip_lm_head:
            if out is not None:
                out.deallocate(True)
            return None
        return out

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the engine-owned per-layer LayerAck channel; each layer bumps it by 1.

        The callback takes **two** arguments here — ``(layer_idx, hidden_states)`` — because
        ``tt/model.py``'s seam does (``DEC-045``), unlike the template's one-argument version
        (``models/demos/gpt_oss_d_p/tt/model.py:211``). The hidden-state tensor is **live**: this
        callback must not deallocate it.
        """
        assert self.compiled, "call compile() before set_layer_ack_channel()"

        def on_layer_complete(layer_idx: int, hidden_states) -> None:  # noqa: ARG001 - live tensor
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete

    # -------------------------------------------------------------------------------------
    # Optional migration hooks (the serving loop never calls these)
    # -------------------------------------------------------------------------------------
    def kv_migration_base_address(self, kv_cache) -> int:
        """K's base DRAM address — the anchor the engine all-gathers to merge pipeline stages."""
        return int(self._resolve_kv(kv_cache).k.buffer_address())

    def build_kv_chunk_table(self, kv_cache, path: str, **kwargs) -> str:
        """Not implemented here — P10 owns it (``tt/runners/kv_chunk_table.py``).

        Raising rather than returning an empty table is deliberate: the engine publishes whatever
        this returns to the migration worker, and a table that is structurally valid but wrong
        migrates the wrong DRAM ranges, which presents as a corrupted decode long after prefill.
        """
        del kv_cache, path, kwargs
        raise NotImplementedError(
            "llama32_8b_d_p has no KV chunk-address table yet: it belongs in "
            "tt/runners/kv_chunk_table.py, which is P10's deliverable (03_OUTLINE.md §3.21). Run "
            "with PREFILL_ENABLE_MIGRATION=0 until then. The geometry it must encode is fixed and "
            "gated: NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK=32, shard row [1, 1, 32, 128], "
            "ROUND_ROBIN_1D over mesh_device.dram_grid_size().x banks (G-KV)."
        )

    # -------------------------------------------------------------------------------------
    # Bring-up read-back (G-CHUNK / G-GOLDEN). Never called in production serving.
    # -------------------------------------------------------------------------------------
    def gather_layer(self, *, slot_id, layer_idx, n_tokens, kv_cache=None, chunk_size=None):
        """One layer's device K/V read back in **natural token order**.

        Inverts the block-cyclic SP layout the writer applied, and stacks the per-TP-col KV heads
        back into one tensor. Returns ``(k, v)`` torch fp32, each
        ``[1, num_kv_heads, n_tokens, head_dim]``, in **device convention**: K is Meta-RoPE
        swizzled over the full ``head_dim``; V is raw. A caller comparing against the golden must
        permute the golden's K lanes — ``scripts/verify_golden_kv.hf_to_meta_lane_permutation``.
        """
        kv = self._resolve_kv(kv_cache)
        cfg = self.config
        sp, cols = cfg.sp_factor, cfg.tp_factor
        n_kv = self.hf_config.num_key_value_heads
        slot = slot_id * cfg.num_layers + layer_idx
        assert 0 <= layer_idx < cfg.num_layers, f"layer_idx {layer_idx} out of range"
        chunk_size = cfg.default_chunk_size if chunk_size is None else chunk_size
        # shard row -> natural global position: the exact inverse of update_padded_kv_cache's
        # block-cyclic write (gated host-only by G-KV's test_blockcyclic_positions_are_an_exact_inverse).
        positions = blockcyclic_positions(sp, chunk_size, cfg.max_seq_len)

        def _one(cache_tensor, col):
            device_tensors = ttnn.get_device_tensors(cache_tensor)
            device_rows = torch.cat(
                [ttnn.to_torch(device_tensors[r * cols + col])[slot, 0].float() for r in range(sp)], dim=0
            )
            natural = torch.empty_like(device_rows)
            natural[positions] = device_rows
            return natural[:n_tokens]

        k = torch.stack([_one(kv.k, c) for c in range(n_kv)], dim=0).unsqueeze(0)
        v = torch.stack([_one(kv.v, c) for c in range(n_kv)], dim=0).unsqueeze(0)
        return k, v

    def dump_slot_kv(self, out_dir, *, slot_id, n_tokens, kv_cache=None, chunk_size=None) -> Path:
        """Write one slot's whole KV read-back in the layout ``scripts/verify_golden_kv.py`` reads.

        ``{out_dir}/layer_<i>.safetensors`` (``key_cache_layer_<i>`` / ``value_cache_layer_<i>``,
        ``[1, num_kv_heads, n_tokens, head_dim]``) plus a ``metadata.json`` declaring
        ``convention: "meta"`` — so the reader applies the HF->Meta lane permutation to the golden
        rather than guessing. Layer indices are **global** (``first_layer_idx + local``), matching
        the golden trace's numbering on a pipelined rank.
        """
        from safetensors.torch import save_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        layers = []
        for local in range(self.config.num_layers):
            global_idx = self.config.first_layer_idx + local
            k, v = self.gather_layer(
                slot_id=slot_id, layer_idx=local, n_tokens=n_tokens, kv_cache=kv_cache, chunk_size=chunk_size
            )
            save_file(
                {
                    f"key_cache_layer_{global_idx}": k.contiguous(),
                    f"value_cache_layer_{global_idx}": v.contiguous(),
                },
                str(out_dir / f"layer_{global_idx}.safetensors"),
            )
            layers.append(global_idx)
        with open(out_dir / "metadata.json", "w") as fh:
            json.dump(
                {
                    "convention": "meta",
                    "n_tokens": int(n_tokens),
                    "layers": layers,
                    "head_dim": int(self.hf_config.head_dim),
                    "rotary_dim": int(self.hf_config.head_dim),  # Llama is full rotary
                    "num_kv_heads": int(self.hf_config.num_key_value_heads),
                    "mesh_shape": list(self.config.mesh_shape),
                    "chunk_size": int(self.config.default_chunk_size if chunk_size is None else chunk_size),
                    "cache_dtype": str(self.config.cache_dtype),
                },
                fh,
                indent=2,
            )
        logger.info(f"[llama32_8b_d_p] dumped {len(layers)} layers of slot {slot_id} KV -> {out_dir}")
        return out_dir

    def kv_cache_pcc_check(self, kv_cache=None, *, slot_id, n_chunks, trace_dir=None, chunk_size=None, real_len=None):
        """Per-layer K/V PCC of the populated cache against the golden trace; returns the min.

        Golden layout is ``scripts/generate_golden_kv_cache.py``'s
        (``{trace_dir}/kv_cache/layer_<i>.safetensors``). ``n_chunks`` caps the comparison to what
        this run actually wrote; ``real_len`` caps it further to the non-pad tokens. Optional
        bring-up hook — never called in production serving.
        """
        from safetensors import safe_open

        from models.demos.llama32_8b_d_p.scripts.verify_golden_kv import hf_to_meta_lane_permutation, pcc

        raw_trace = trace_dir or os.environ.get("PREFILL_TRACE_DIR")
        assert raw_trace, "kv_cache_pcc_check needs trace_dir= or $PREFILL_TRACE_DIR"
        trace = Path(raw_trace)
        with open(trace / "metadata.json") as fh:
            token_ids = list(json.load(fh)["token_ids"])
        chunk_size = self.config.default_chunk_size if chunk_size is None else chunk_size
        n_tokens = min(len(token_ids), n_chunks * chunk_size)
        if real_len is not None:
            n_tokens = min(n_tokens, int(real_len))
        # The read-back gathers whole cache rows, so a partial tile at the tail would compare
        # device rows that were never written against golden rows that exist.
        n_tokens -= n_tokens % ttnn.TILE_SIZE
        assert n_tokens > 0, f"kv_cache_pcc_check: n_tokens=0 (n_chunks={n_chunks}, chunk_size={chunk_size})"

        head_dim = self.hf_config.head_dim
        perm = hf_to_meta_lane_permutation(head_dim, head_dim)
        kv_dir = trace / "kv_cache"
        logger.info(
            f"[kv-pcc] per-layer K / V vs golden ({trace}) over [0, {n_tokens}) " f"({self.config.num_layers} layers):"
        )
        min_k, min_v = 1.0, 1.0
        rows = []
        for local in range(self.config.num_layers):
            global_idx = self.config.first_layer_idx + local
            dev_k, dev_v = self.gather_layer(
                slot_id=slot_id, layer_idx=local, n_tokens=n_tokens, kv_cache=kv_cache, chunk_size=chunk_size
            )
            with safe_open(str(kv_dir / f"layer_{global_idx}.safetensors"), framework="pt") as handle:
                golden_k = handle.get_tensor(f"key_cache_layer_{global_idx}").float()[:, :, :n_tokens, :][..., perm]
                golden_v = handle.get_tensor(f"value_cache_layer_{global_idx}").float()[:, :, :n_tokens, :]
            score_k, score_v = pcc(golden_k, dev_k), pcc(golden_v, dev_v)
            min_k, min_v = min(min_k, score_k), min(min_v, score_v)
            rows.append((global_idx, score_k, score_v))
            logger.info(f"  layer {global_idx:>2}: K={score_k:.5f} V={score_v:.5f}")
        logger.info(
            f"[kv-pcc] over {len(rows)} layers: min K={min_k:.5f} mean K="
            f"{sum(r[1] for r in rows) / len(rows):.5f} | min V={min_v:.5f} mean V="
            f"{sum(r[2] for r in rows) / len(rows):.5f}"
        )
        return min(min_k, min_v)

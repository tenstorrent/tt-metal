# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS single-rank chunked-prefill runtime.

Mirrors ``minimax_m3/tt/tt_prefill_runtime.py`` and satisfies the common/prefill runtime contract
(``models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`` §2): build model -> allocate KV
cache -> compile -> ``prefill_chunk`` (per chunk). The runtime is mode-agnostic; the caller (the
common engine, or ``tests/galaxy_prefill_kv_pcc.py``) drives one-shot vs chunked and does the golden
KV-cache PCC.

Cache ownership: the runtime can OWN its KV cache (``owns_kv_cache=True``, used by the standalone
galaxy harness — ``self.kv_cache``) OR run engine-owned (``owns_kv_cache=False``, the adapter path:
the engine allocates via ``GptOssPrefillAdapter.allocate_kv_cache`` and passes the ``KvCaches`` tuple
into every call). ``prefill_chunk`` / ``compile`` / ``gather_layer`` / ``kv_cache_pcc_check`` accept
an optional cache arg that defaults to ``self.kv_cache``.

Migration hooks (Gate 1–2 in ``PREFILL_MIGRATION_TESTING.md``): ``build_kv_chunk_table`` (via
``tt/runners/kv_chunk_table.py``), ``kv_migration_base_address``, ``read_slot_kv``, and
``set_layer_ack_channel``. Request-mode H2D delivers SP-sharded uint32 tokens; ``prefill_chunk``
embeds them on the first rank (same path as ``make_chunk_input``).

CHUNKED prefill is supported: the SP cache-backed RingJointSDPA path uses the block-cyclic packed KV
cache (``attention/dense_sp.py``) from chunk 0 onward. One-shot SP prefill retains its exact
all-gather fallback because sliding RingJointSDPA requires short-Q/long-K. The single-chip (sp==1)
cache-read is still ``NotImplementedError`` (not used on the galaxy). The galaxy KV-PCC harness runs
both one-shot and multi-chunk.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions

from .attention import GptOssKVCache, allocate_kv_cache
from .rope import build_indexed_rope


def resolve_chunk_sizes(default_chunk_size: int, additional_chunk_sizes: tuple, max_seq_len: int) -> tuple:
    """Supported chunk sizes, deduped, largest first; each must divide max_seq_len (rope must tile the cache)."""
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
    num_layers: int  # layers built/cached by this runtime (== model total for single-rank)
    max_seq_len: int  # per-user KV-cache length in tokens; must be a multiple of chunk_size
    mesh_shape: tuple = (4, 8)  # (SP rows, TP cols) on the Blackhole galaxy
    default_chunk_size: int = (
        8192  # per prefill_chunk() call unless overridden (8k divides 128k context); one-shot sets == max_seq_len
    )
    # Other sizes this instance can serve per call (own rope each; MoE buffers sized at the largest).
    additional_chunk_sizes: tuple = ()  # semantically a set; tuple for dataclass-default ergonomics
    num_users: int = 1  # independent cache slots (user-major batch)
    sp_axis: int = 0
    tp_axis: int = 1
    # Chunked SP requests use RingJointSDPA from chunk 0; one-shot uses the all-gather fallback.
    topology: ttnn.Topology = ttnn.Topology.Ring
    use_ep_moe: bool = True
    expert_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None
    # When True the runtime allocates + owns its KV cache (self.kv_cache) — the standalone galaxy
    # harness path. The adapter/engine path sets this False and passes the engine-owned KvCaches in.
    owns_kv_cache: bool = True
    # Pipeline-parallel rank flags the common prefill runner reads off runtime.config
    # (single-rank standalone/harness => both True). first_layer_idx is the GLOBAL index of this
    # rank's first layer (0 on single-rank); used by PREFILL_STANDALONE_PCC golden offset.
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
    """Single-rank GPT-OSS prefill lifecycle: build model -> (optionally allocate KV cache) ->
    build indexed rope -> compile -> prefill_chunk."""

    def __init__(self, mesh_device, hf_config, state_dict: dict, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config

        self.chunk_sizes = resolve_chunk_sizes(
            config.default_chunk_size, config.additional_chunk_sizes, config.max_seq_len
        )
        self.max_chunk_size = self.chunk_sizes[0]
        # Ring by default (faster CCLs on torus pods); Linear is supported for pods without wraparound.
        assert config.topology in (
            ttnn.Topology.Ring,
            ttnn.Topology.Linear,
        ), f"GPT-OSS sequence-parallel prefill supports Ring or Linear topology, got {config.topology}"

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False
        self.kv_cache = None
        self._on_layer_complete = None  # set by set_layer_ack_channel (LayerAck inject)

        self._build_model(state_dict)
        if config.owns_kv_cache:
            self._allocate_kv_cache()
        self._build_indexed_rope()

    def _build_model(self, state_dict: dict) -> None:
        from models.demos.gpt_oss_d_p.tt.config import MeshConfig
        from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links

        from .ccl import CCLManager
        from .model import Model

        rows, cols = self.config.mesh_shape
        logger.info(
            f"Building GPT-OSS TtPrefillRuntime model: num_layers={self.config.num_layers} "
            f"max_seq_len={self.config.max_seq_len} chunk_sizes={self.chunk_sizes} "
            f"num_users={self.config.num_users} mesh_shape={self.config.mesh_shape}"
        )
        mesh_config = MeshConfig((rows, cols), tp=cols)
        ccl = CCLManager(
            self.mesh_device, num_links=get_default_num_links(self.mesh_device), topology=self.config.topology
        )
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
            use_ep_moe=self.config.use_ep_moe,
            # MoE buffers are max capacities: size at the largest supported chunk.
            ep_seq_len_per_chip=self.max_chunk_size // self.config.sp_factor,
            expert_weight_dtype=self.config.expert_weight_dtype,
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        # ONE cache holding num_users * num_layers slots (user-major); each (user, layer) slot is
        # filled per chunk. K/V heads shard on the TP cols; the sequence is SP-sharded block-cyclic.
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
        size-specific). ``self.rope_indexed`` maps chunk_size -> rope."""
        rs = getattr(self.hf_config, "rope_scaling", None) or {}
        self.rope_indexed = {
            cs: build_indexed_rope(
                self.mesh_device,
                head_dim=self.hf_config.head_dim,
                max_seq_len=self.config.max_seq_len,
                chunk_size=cs,
                sp_axis=self.config.sp_axis,
                rope_theta=getattr(self.hf_config, "rope_theta", 150000.0),
                yarn_factor=rs.get("factor", 32.0),
                yarn_orig_max_pos=rs.get("original_max_position_embeddings", 4096),
                yarn_beta_fast=rs.get("beta_fast", 32.0),
                yarn_beta_slow=rs.get("beta_slow", 1.0),
            )
            for cs in self.chunk_sizes
        }

    def _resolve_kv(self, kv_caches) -> GptOssKVCache:
        """Resolve the GptOssKVCache from the (optional) caller arg. Accepts None (use self-owned),
        a GptOssKVCache, or the engine's KvCaches tuple (index 0)."""
        if kv_caches is None:
            assert self.kv_cache is not None, "runtime has no KV cache (owns_kv_cache=False): pass kv_caches"
            return self.kv_cache
        if isinstance(kv_caches, GptOssKVCache):
            return kv_caches
        return kv_caches[0]

    def make_chunk_input(self, token_ids: list, chunk_size: Optional[int] = None) -> ttnn.Tensor:
        """Build one chunk's device input for ``prefill_chunk``.

        ``chunk_size`` (default: config default) selects the chunk width and must be a supported size.
        On the first rank: SP-sharded uint32 ROW_MAJOR DRAM tokens of per-chip shape
        ``(1, 1, chunk_size // sp)`` — the SAME layout request-mode H2D delivers, so both paths feed
        one code path; ``prefill_chunk`` embeds on device. On a non-first pipeline rank the input is
        already a hidden-state activation (D2D) — return a placeholder of the right spec for warm-up.
        """
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        assert chunk_size in self.rope_indexed, f"chunk_size={chunk_size} not in supported {tuple(self.rope_indexed)}"
        sp = self.config.sp_factor
        s_local = chunk_size // sp
        if not self.config.is_first_rank:
            # Placeholder activation for compile warm-up on non-first ranks (unused in single-rank).
            emb = self.hf_config.hidden_size
            return ttnn.from_torch(
                torch.zeros(1, 1, s_local, emb),
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

    def _embed_tokens(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed SP-sharded uint32 tokens into the bf16 residual stream the layers consume."""
        x = ttnn.embedding(tokens, self.model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        return x

    def compile(self, kv_caches=None) -> None:
        """Warm up the kernels by running zero-token chunks through prefill_chunk (JIT-compiles).

        The first chunk exercises cache-backed RingJointSDPA when the cache is larger than the chunk;
        equal-sized one-shot requests instead warm the all-gather fallback. When the config is multi-chunk
        (max_seq_len > chunk_size), warm a second chunk too so its cache-growth runtime arguments are
        covered before the first served/timed request. (This is separate from the one-time empty-disk
        kernel-cache compile that only the very first run ever pays.)"""
        assert self.model_built
        # Warm each supported size (else its kernels JIT inside the first served chunk).
        for chunk in self.chunk_sizes:
            ring = self.config.max_seq_len > chunk
            logger.info(
                f"GPT-OSS TtPrefillRuntime.compile() — warming up "
                f"{'2 cache-backed ring chunks' if ring else 'one all-gather fallback chunk'} of {chunk} tokens"
            )
            # prefill_chunk consumes (deallocates) its input tensor, so build a fresh input per call.
            self.prefill_chunk(
                self.make_chunk_input([0] * chunk, chunk),
                kv_caches,
                slot_id=0,
                actual_start=0,
                actual_end=chunk,
                chunk_size=chunk,
            )
            if ring:
                # actual_start>0 drives the ring cache-read; it reads the prefix we just wrote at [0, chunk).
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
        chunk_size: Optional[int] = None,  # variable chunk length: which supported size this chunk is
        request_id: int = -1,  # accepted for the common-runner contract; single-request prefill ignores it
        d2h_service=None,  # accepted for the common-runner contract; this runtime uses host-callback LayerAcks
        record_dev=None,  # accepted for the common-runner contract; the D1H record path is unused here
        input_owned_by_caller: bool = False,  # caller reuses input_tensor for later chunks: do not free it
    ) -> Optional[ttnn.Tensor]:
        """Prefill ONE chunk into user ``slot_id``'s slice of the KV cache (self-owned or the engine's
        ``kv_caches``). Returns None (skip_lm_head) — the populated cache is the output.

        [actual_start, actual_end): absolute KV-position range of this chunk's real tokens. actual_start
        is the cache write offset (valid prefix already cached); the last chunk's tail may be pad, so
        actual_end < actual_start + chunk_size. Call once per chunk, in order.

        On the first rank ``input_tensor`` is SP-sharded uint32 tokens (``make_chunk_input`` / H2D);
        they are embedded here. Non-first ranks receive activations over D2D already embedded.
        If a LayerAck channel is registered, the model bumps it once per layer via ``on_layer_complete``.

        Every SP chunk writes K/V. A chunked request, including actual_start == 0, then uses the
        cache-backed RingJointSDPA path; an equal-sized one-shot request uses the all-gather fallback.
        """
        if d2h_service is not None:
            raise NotImplementedError(
                "GPT-OSS prefill emits layer acks via set_layer_ack_channel, not the D2H path; "
                "run with PREFILL_ENABLE_LAYER_ACK=0 or wire the D2H ack into this runtime."
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
        ), f"[actual_start={actual_start}, actual_end={actual_end}) not within one chunk of {chunk_size}"

        # input_owned_by_caller (the engine's persistent socket-input buffer, reused for every chunk)
        # means this call must not free the input. On the first rank that is just declining the free, as
        # the embedding only reads it. A non-first rank has to copy instead — see below.
        if self.config.is_first_rank:
            x_embd = self._embed_tokens(input_tensor)
            if not input_owned_by_caller:
                ttnn.deallocate(input_tensor)
        elif input_owned_by_caller:
            # A non-first rank cannot lend the caller's buffer to the forward: the free lives inside the
            # model, not here — tt/layer.py takes the layer input as `residual` and force-frees it after
            # the attention add — so the buffer would come back released and the next chunk would drain
            # into freed memory. Copy, and leave the caller's buffer intact. That copy is the price of the
            # free sitting in the model; a runtime that frees its own input instead (deepseek_v3_d_p)
            # passes the buffer straight through and pays nothing. Moving the layer-0 free out here would
            # remove it for this runtime too.
            x_embd = ttnn.clone(input_tensor)
        else:
            x_embd = input_tensor

        out = self.model.prefill_forward(
            x_embd,
            rot_mats_global=self.rope_indexed[chunk_size],  # per-size indexed rope (persistent; not deallocated)
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
        return out  # logits [1,1,chunk_local,vocab_shard], SP-sharded on seq / TP-sharded on vocab

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the per-layer LayerAck channel (engine-created + owned). ``prefill_chunk`` bumps it
        once per layer (``inject(1)``); the scheduler/driver drains the delta. Called by the engine in
        single-rank request mode when migration or request-mode acks are enabled."""
        assert self.compiled, "Call compile() before set_layer_ack_channel()"

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete

    def kv_migration_base_address(self, kv_caches) -> int:
        """Stage KV base for the runner's device-map / stage-layout gather. The multi-config table
        builder uses each tensor's own ``buffer_address()``; this returns K's base (required hook)."""
        return int(self._resolve_kv(kv_caches).k.buffer_address())

    def build_kv_chunk_table(
        self,
        kv_caches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers: Optional[int] = None,
        stage_layout=None,
    ) -> str:
        """Build + serialize the GPT-OSS multi-config KV chunk address table (k_h0..N, v_h0..N) to
        ``path`` and return it. Issues no comms — the engine publishes to the migration worker.
        Single-rank only (``PREFILL_ENABLE_MIGRATION=1`` is rejected for ``num_ranks>1``). Extra kwargs
        match the DeepSeek/PP runner call site and are ignored for this single-rank GQA path."""
        del first_layer_idx, num_my_layers, stage_layout  # single-rank: whole-model table
        from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        kv = self._resolve_kv(kv_caches)
        c = self.config
        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kv_cache=kv,
            seq_len=c.max_seq_len,
            num_layers=c.num_layers,
            mesh_shape=c.mesh_shape,
            sp_axis=c.sp_axis,
            num_users=c.num_users,
            chunk_size=c.default_chunk_size,
            num_kv_heads=self.hf_config.num_key_value_heads,
            head_dim=self.hf_config.head_dim,
            path=path,
        )

    def read_slot_kv(self, kv_caches, slot: int):
        """Read one slot's KV from device to host: ``[k, v]``, each
        ``[num_layers, num_kv_heads, seq_cache, head_dim]`` in the raw on-device (block-cyclic) layout.
        Used by pairwise migration validation (dst==src). ``DRAM_MEMORY_CONFIG`` on the slice is
        required — the cache is ND-sharded ROUND_ROBIN_1D."""
        kv = self._resolve_kv(kv_caches)
        mesh_device = self.mesh_device
        num_layers = self.config.num_layers

        def _block(tensor):
            s = list(tensor.shape)
            sl = ttnn.slice(
                tensor,
                [slot * num_layers, 0, 0, 0],
                [(slot + 1) * num_layers, s[1], s[2], s[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            block = ttnn.to_torch(
                sl, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
            ).float()  # [num_layers, nkv (=cols), seq_cache, head_dim]
            ttnn.deallocate(sl)
            return block

        return [_block(kv.k), _block(kv.v)]

    def gather_layer(self, slot_id: int, layer_idx: int, n_tokens: int, kv_caches=None, chunk_size=None):
        """Read one layer's device K/V cache back to NATURAL token order (un-rotating the block-cyclic
        SP layout). Returns (k, v) torch tensors in DEVICE convention: K is Meta-RoPE swizzled over the
        (full) head_dim — the caller reconciles vs the HF golden; V is raw. Shapes:
        k, v -> [1, num_kv_heads, n_tokens, head_dim]. No index_k (GQA)."""
        kv = self._resolve_kv(kv_caches)
        sp = self.config.sp_factor
        cols = self.config.tp_factor  # KV head c lives on col c
        nkv = self.hf_config.num_key_value_heads
        slot = slot_id * self.config.num_layers + layer_idx
        # shard-row -> natural global position (inverse of the update_padded_kv_cache writer).
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        p = blockcyclic_positions(sp, chunk_size, self.config.max_seq_len)

        def gather(cache_tensor, col):
            dts = ttnn.get_device_tensors(cache_tensor)
            dev = torch.cat([ttnn.to_torch(dts[r * cols + col])[slot, 0].float() for r in range(sp)], dim=0)
            nat = torch.empty_like(dev)
            nat[p] = dev
            return nat[:n_tokens]

        k = torch.stack([gather(kv.k, c) for c in range(nkv)], dim=0).unsqueeze(0)
        v = torch.stack([gather(kv.v, c) for c in range(nkv)], dim=0).unsqueeze(0)
        return k, v

    def _kv_diag(self, gL, g_k, dev_k, g_v, dev_v, out_dir):
        """Bring-up diagnostic (gated by GPT_OSS_KV_DUMP) to localize a per-position K RoPE error.

        Dumps golden/device K & V (both Meta space, natural position order, [1, nkv, n_tokens,
        head_dim]) to ``out_dir`` and prints a compact per-SP-block + per-head K PCC so a single
        galaxy run reveals WHERE K diverges:
          * uniform low PCC everywhere            => base convention (unlikely; unit test is green)
          * degrades at SP-rank block boundaries  => per-rank global-position offset in the rope
          * scattered specific positions          => block-cyclic cos/sin reorder mismatch
        The dumped .pt tensors are re-loadable for offline per-position analysis on CPU."""

        def _pcc(a, b):
            a = a.reshape(-1).float()
            b = b.reshape(-1).float()
            a = a - a.mean()
            b = b - b.mean()
            d = a.norm() * b.norm()
            return float(torch.dot(a, b) / d) if d > 0 else 1.0

        sp = self.config.mesh_shape[0]
        n_tokens = dev_k.shape[2]
        chunk_local = self.config.default_chunk_size // sp  # SP-rank contiguous block width (one-shot)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "g_k": g_k,
                "dev_k": dev_k,
                "g_v": g_v,
                "dev_v": dev_v,
                "sp": sp,
                "chunk_local": chunk_local,
                "n_tokens": n_tokens,
                "layer": gL,
            },
            out_dir / f"layer_{gL}.pt",
        )
        blocks = []
        for r in range(sp):
            a, b = r * chunk_local, min((r + 1) * chunk_local, n_tokens)
            if a >= n_tokens:
                break
            blocks.append(f"blk{r}[{a}:{b}]={_pcc(g_k[:, :, a:b, :], dev_k[:, :, a:b, :]):.4f}")
        heads = " ".join(f"h{h}={_pcc(g_k[:, h], dev_k[:, h]):.4f}" for h in range(dev_k.shape[1]))
        logger.info(f"[kv-diag L{gL}] K per-SP-block (chunk_local={chunk_local}): {' '.join(blocks)}")
        logger.info(f"[kv-diag L{gL}] K per-head:     {heads}")
        logger.info(f"[kv-diag L{gL}] dumped -> {out_dir / f'layer_{gL}.pt'}")

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
        """PCC the populated KV cache for ``slot_id`` against the golden trace; return the min per-layer
        PCC (K and V). Optional bring-up hook — never called in production serving.

        Golden layout: {trace_dir}/kv_cache/layer_N.safetensors
        with ``key_cache_layer_N`` (post-RoPE K, HF half-split convention) + ``value_cache_layer_N`` (raw
        V), each [1, num_kv_heads, seq_len, head_dim]. GQA => NO index_k. The device K is Meta-RoPE
        swizzled over the full head_dim, so the golden K's rotary slice is permuted HF->Meta first.

        ``n_chunks`` caps the compare to what this run actually wrote (``n_chunks * chunk_size``).
        ``real_len`` further caps to non-pad tokens. ``pt_path_override`` is unsupported
        (trace-dir goldens only) and rejected if set — required keyword for Gate 2b / validation.py.
        """
        from safetensors import safe_open

        from models.common.utility_functions import comp_pcc

        if pt_path_override is not None:
            raise NotImplementedError(
                "GPT-OSS kv_cache_pcc_check has no per-slot .pt golden path; use PREFILL_TRACE_DIR"
            )
        from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir

        raw_trace = trace_dir or os.environ.get("PREFILL_TRACE_DIR")
        assert raw_trace, "kv_cache_pcc_check needs PREFILL_TRACE_DIR or trace_dir="
        trace_dir = resolve_trace_dir(raw_trace)
        token_ids = list(json.load(open(Path(trace_dir) / "metadata.json"))["token_ids"])
        # Only score tokens this run filled (matches MiniMax / avoids comparing past NCHUNKS*chunk_size).
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        n_tokens = min(len(token_ids), n_chunks * chunk_size)
        if real_len is not None:
            n_tokens = min(n_tokens, int(real_len))
        assert n_tokens > 0, f"kv_cache_pcc_check: n_tokens=0 (n_chunks={n_chunks}, chunk_size={chunk_size})"

        head_dim = self.hf_config.head_dim
        rotary_dim = getattr(self.hf_config, "rotary_dim", head_dim)
        half = rotary_dim // 2
        src = list(range(head_dim))
        for m in range(rotary_dim):
            src[m] = half * (m % 2) + (m // 2)  # HF half-split -> Meta interleaved
        src = torch.tensor(src, dtype=torch.long)

        kv_dir = Path(trace_dir) / "kv_cache"
        _dump_env = os.environ.get("GPT_OSS_KV_DUMP", "")
        if _dump_env == "all":
            _dump_set = set(range(first_layer_idx, first_layer_idx + self.config.num_layers))
        else:
            _dump_set = {int(x) for x in _dump_env.split(",") if x.strip()} if _dump_env else set()
        # Per-layer tensor dumps land next to the golden trace by default; GPT_OSS_KV_DUMP_DIR overrides.
        _dump_dir = os.environ.get("GPT_OSS_KV_DUMP_DIR") or (Path(trace_dir) / "kv_dump")
        logger.info(
            f"[kv-pcc] per-layer K / V vs golden ({trace_dir}) over [0,{n_tokens}) ({self.config.num_layers} layers):"
        )
        min_k, min_v = 1.0, 1.0
        for L in range(self.config.num_layers):
            gL = first_layer_idx + L
            dev_k, dev_v = self.gather_layer(
                slot_id=slot_id, layer_idx=L, n_tokens=n_tokens, kv_caches=kv_caches, chunk_size=chunk_size
            )
            with safe_open(str(kv_dir / f"layer_{gL}.safetensors"), framework="pt") as h:
                g_k = h.get_tensor(f"key_cache_layer_{gL}").float()[:, :, :n_tokens, :][..., src]  # HF -> Meta
                g_v = h.get_tensor(f"value_cache_layer_{gL}").float()[:, :, :n_tokens, :]
            pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
            pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
            min_k, min_v = min(min_k, pcc_k), min(min_v, pcc_v)
            logger.info(f"  layer {gL:>2}: K={pcc_k:.5f} V={pcc_v:.5f}")
            if gL in _dump_set:
                self._kv_diag(gL, g_k, dev_k, g_v, dev_v, _dump_dir)
        logger.info(f"[kv-pcc] min PCC across {self.config.num_layers} layers: K={min_k:.5f} V={min_v:.5f}")
        return min(min_k, min_v)

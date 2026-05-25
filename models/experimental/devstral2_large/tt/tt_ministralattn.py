# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Self-attention for Devstral-2 / Ministral3.

HF reference (``Ministral3Attention``)::

    q = q_proj(x); k = k_proj(x); v = v_proj(x)           # (B, S, Hq*D), (B, S, Hkv*D), (B, S, Hkv*D)
    q, k = reshape_to_heads(q, k)                          # (B, Hq, S, D), (B, Hkv, S, D)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q = q * llama4_scale(position_ids)                     # post-RoPE query rescale
    attn = SDPA(q, repeat_kv(k), repeat_kv(v))             # GQA via repeat-along-heads
    out = o_proj(reshape(attn))

TP plan (matches ``Ministral3Config.base_model_tp_plan``):
  - ``q_proj``, ``k_proj``, ``v_proj``: **column-parallel** (each device owns
    ``n_local_heads`` queries and ``n_local_kv_heads`` KV heads).
  - ``o_proj``: **row-parallel** + all-reduce along the TP axis.

Llama-4 query scaling is folded into the cos/sin tables of ``TtRotaryEmbedding`` so the device-side
RoPE op produces the rescaled query directly — no extra multiply.

KV cache layout: ``[batch, n_local_kv_heads, max_seq_len, head_dim]`` on each device, DRAM, no
paging in this baseline implementation. SDPA decode uses
``ttnn.transformer.scaled_dot_product_attention_decode``; prefill uses
``ttnn.transformer.scaled_dot_product_attention``.
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn

from models.experimental.devstral2_large.tt.ccl_helpers import all_reduce_replicate
from models.experimental.devstral2_large.tt.mem_config import (
    get_compute_kernel_config,
    get_compute_kernel_config_hifi4,
    get_linear_program_config,
    get_sdpa_decode_compute_kernel_config,
    get_sdpa_decode_output_mem_config,
    get_sdpa_decode_program_config,
)
from models.experimental.devstral2_large.tt.model_args import Devstral2Args
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import (
    TtRotaryEmbedding,
    permute_split_half_to_interleaved,
)
from models.experimental.devstral2_large.tt.weight_loading import (
    resolve_weight_cache_path,
    upload_matmul_weight,
    upload_page_table,
    upload_paged_kv_cache_buffer,
)

__all__ = ["TtAttention"]

# ``nlp_create_qkv_heads_decode`` + decode RoPE expect height-sharded L1 activations.
_DECODE_QKV_HEADS_MEM_CONFIG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)


def _decode_height_shard_mem_config(batch_size: int, *, height: int, width: int) -> ttnn.MemoryConfig:
    grid_size = ttnn.CoreCoord(8, 8)
    batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(height, width),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _shard_decode_rope_tables(
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    *,
    batch_size: int,
    head_dim: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Match cos/sin layout to decode Q/K from ``nlp_create_qkv_heads_decode``."""
    mem_config = _decode_height_shard_mem_config(batch_size, height=ttnn.TILE_SIZE, width=head_dim)
    return ttnn.interleaved_to_sharded(cos, mem_config), ttnn.interleaved_to_sharded(sin, mem_config)


def _reshape_decode_batch_dim(x: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
    """Decode activations use TILE height padding (often 32); logical batch may be smaller.

    Matches tt_transformers decode: ``reshape(x, (1, 1, batch, H), (1, 1, tile_h, H))``.
    """
    hidden = int(x.shape[-1])
    tile_h = int(x.shape[-2])
    logical = (1, 1, batch_size, hidden)
    if tile_h == batch_size:
        return ttnn.reshape(x, logical)
    return ttnn.reshape(x, logical, (1, 1, tile_h, hidden))


# --- Weight loading ---


def _to_tt_weight(
    w_hf: torch.Tensor,
    mesh_device,
    args: Devstral2Args,
    dtype: ttnn.DataType,
    *,
    shard_dim: Optional[int],
    weight_cache_path: Optional[str] = None,
    cache_key: str,
) -> ttnn.Tensor:
    """Upload HF Linear weight ``(out, in)`` as TTNN ``(in, out)``, optionally TP-sharded."""
    return upload_matmul_weight(
        w_hf,
        mesh_device,
        args,
        dtype=dtype,
        shard_dim=shard_dim,
        weight_cache_path=weight_cache_path,
        cache_key=cache_key,
    )


def _interleave_qkv_for_tp(q_w: torch.Tensor, k_w: torch.Tensor, v_w: torch.Tensor, tp: int) -> torch.Tensor:
    """Arrange ``[Q | K | V]`` HF rows so a contiguous TP-chunk slice puts
    ``(Q_local, K_local, V_local)`` on each device.

    Naive ``cat([q_w, k_w, v_w], dim=0)`` keeps all of Q in the first third etc. — strip-sharding
    that along dim 0 gives some devices all-Q and others all-K/V. Instead, split each of Q, K, V
    into ``tp`` row-chunks and zip them: contiguous chunk ``i`` of the result is exactly device
    ``i``'s ``(Q_local rows ; K_local rows ; V_local rows)``.
    """
    if tp <= 1:
        return torch.cat([q_w, k_w, v_w], dim=0)
    q_chunks = torch.chunk(q_w, tp, dim=0)
    k_chunks = torch.chunk(k_w, tp, dim=0)
    v_chunks = torch.chunk(v_w, tp, dim=0)
    pieces: list[torch.Tensor] = []
    for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks):
        pieces.append(qc)
        pieces.append(kc)
        pieces.append(vc)
    return torch.cat(pieces, dim=0)


# --- Attention module ---


class TtAttention:
    """Inline self-attention with optional KV cache.

    Activations entering / leaving are **replicated** along the TP axis (full ``hidden_size``).
    Internally, QKV outputs are sharded across heads along the TP axis; the o_proj all-reduce
    restores replication.
    """

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        state_dict: dict,
        layer_idx: int,
        tt_ccl,
        rotary_emb: TtRotaryEmbedding,
        *,
        dtype: Optional[ttnn.DataType] = None,
        weight_cache_path: Optional[str] = None,
    ) -> None:
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.rotary_emb = rotary_emb
        self.layer_idx = layer_idx
        self.dtype = dtype or args.weight_dtype

        prefix = args.state_dict_prefix("self_attn", layer_idx)
        q_w = state_dict[prefix + "q_proj.weight"]
        k_w = state_dict[prefix + "k_proj.weight"]
        v_w = state_dict[prefix + "v_proj.weight"]
        o_w = state_dict[prefix + "o_proj.weight"]

        # (1) Permute Q / K head_dim from HF split-half to pairwise-interleaved so the device-side
        # RoPE op (``ttnn.experimental.rotary_embedding_llama``) sees the layout it expects.
        # V is **not** permuted — it never enters RoPE, and SDPA output flows into o_proj in V's
        # native layout.
        q_w = permute_split_half_to_interleaved(q_w, args.head_dim)
        k_w = permute_split_half_to_interleaved(k_w, args.head_dim)

        # (2) Fuse Q, K, V into a single column-parallel matmul, with rows interleaved per
        # TP chunk so a contiguous TP strip-shard puts ``(Q_local | K_local | V_local)`` on each
        # device. With TP=8 on Loudbox this lifts Nt from 48 (Q) / 8 (KV) to 56 (fused) and
        # collapses two dispatches into one.
        qkv_w = _interleave_qkv_for_tp(q_w, k_w, v_w, args.tp)

        wp = resolve_weight_cache_path(weight_cache_path, args)
        pfx = prefix
        # Quantize weights to bfloat8_b for DRAM bandwidth; matmul outputs stay bfloat16 with
        # HiFi4 fidelity. Quantizing outputs to bf8_b compounds across 88 layers and drops PCC.
        self.qkv_proj_weight_dtype = ttnn.bfloat8_b
        self.qkv_proj_output_dtype = ttnn.bfloat16
        self.qkv_proj = _to_tt_weight(
            qkv_w,
            mesh_device,
            args,
            self.qkv_proj_weight_dtype,
            shard_dim=-1,
            weight_cache_path=wp,
            cache_key=f"{pfx}qkv_proj_bfp8",
        )
        self.o_proj_weight_dtype = ttnn.bfloat8_b
        self.o_proj_output_dtype = ttnn.bfloat16
        self.o_proj = _to_tt_weight(
            o_w,
            mesh_device,
            args,
            self.o_proj_weight_dtype,
            shard_dim=-2,
            weight_cache_path=wp,
            cache_key=f"{pfx}o_proj_bfp8",
        )

        # Paged KV cache: [num_total_blocks, n_local_kv_heads, block_size, head_dim]. Logical
        # position p maps to physical block ``page_table[user, p // block_size]`` at slot
        # ``p % block_size``. Chunked prefill writes block-by-block via ``paged_fill_cache``;
        # decode updates a single position via ``paged_update_cache``.
        self.k_cache = upload_paged_kv_cache_buffer(
            num_total_blocks=args.kv_num_total_blocks,
            n_kv_heads=args.n_local_kv_heads,
            block_size=args.kv_block_size,
            head_dim=args.head_dim,
            mesh_device=mesh_device,
            dtype=args.kv_cache_dtype,
            weight_cache_path=wp,
            cache_key=f"{pfx}k_cache_paged",
        )
        self.v_cache = upload_paged_kv_cache_buffer(
            num_total_blocks=args.kv_num_total_blocks,
            n_kv_heads=args.n_local_kv_heads,
            block_size=args.kv_block_size,
            head_dim=args.head_dim,
            mesh_device=mesh_device,
            dtype=args.kv_cache_dtype,
            weight_cache_path=wp,
            cache_key=f"{pfx}v_cache_paged",
        )
        # Page table: ``[batch, blocks_per_user]`` int32 mapping logical block -> physical block.
        # Identical across all attention layers; step 3 (model) will hoist this to model scope to
        # save the per-layer duplication. For now each layer holds its own (tiny) replica.
        self.page_table = upload_page_table(
            batch_size=args.max_batch_size,
            num_blocks_per_user=args.kv_num_blocks_per_user,
            mesh_device=mesh_device,
            weight_cache_path=wp,
        )

        self._compute_kernel_config = get_compute_kernel_config(mesh_device)
        self._compute_kernel_config_hifi4 = get_compute_kernel_config_hifi4(mesh_device)
        self._sdpa_decode_program_config = get_sdpa_decode_program_config(mesh_device)
        self._sdpa_decode_compute_kernel_config = get_sdpa_decode_compute_kernel_config(mesh_device)

    # --- Projections (shared by prefill / decode) ---

    def _act_mem(self, mode: str) -> ttnn.MemoryConfig:
        return self.args.get_activation_mem_config(mode, self.mesh_device)

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        *,
        mode: str,
        kind: str,
        seq_len: int = 1,
        activation: Optional[str] = None,
        output_dtype: Optional[ttnn.DataType] = None,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            dtype=output_dtype if output_dtype is not None else self.args.activation_dtype,
            memory_config=self._act_mem(mode),
            activation=activation,
            program_config=get_linear_program_config(
                self.args,
                self.mesh_device,
                mode=mode,
                kind=kind,
                seq_len=seq_len,
                k=int(weight.shape[-2]),
                n=int(weight.shape[-1]),
            ),
            compute_kernel_config=compute_kernel_config
            if compute_kernel_config is not None
            else self._compute_kernel_config,
        )

    def _project_qkv_prefill(self, x: ttnn.Tensor, *, seq_len: int) -> ttnn.Tensor:
        """Return fused ``[Q | K | V]`` for ``nlp_create_qkv_heads``."""
        return self._linear(
            x,
            self.qkv_proj,
            mode="prefill",
            kind="qkv",
            seq_len=seq_len,
            output_dtype=self.qkv_proj_output_dtype,
            compute_kernel_config=self._compute_kernel_config_hifi4,
        )

    def _project_qkv_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Fused ``[Q | K | V]`` activation for ``nlp_create_qkv_heads_decode``."""
        return self._linear(
            x,
            self.qkv_proj,
            mode="decode",
            kind="qkv",
            output_dtype=self.qkv_proj_output_dtype,
            compute_kernel_config=self._compute_kernel_config_hifi4,
        )

    def _project_o(self, attn_out_flat: ttnn.Tensor, *, mode: str, seq_len: int = 1) -> ttnn.Tensor:
        dense = self._linear(
            attn_out_flat,
            self.o_proj,
            mode=mode,
            kind="o_proj",
            seq_len=seq_len,
            output_dtype=self.o_proj_output_dtype,
            compute_kernel_config=self._compute_kernel_config_hifi4,
        )
        return all_reduce_replicate(
            dense,
            mesh_device=self.mesh_device,
            tt_ccl=self.tt_ccl,
            dim=3,
            cluster_axis=self.args.cluster_axis,
            topology=self.args.ccl_topology,
            memory_config=self.args.get_ccl_output_mem_config(mode, self.mesh_device),
        )

    # --- Prefill: chunk of ``S`` tokens starting at ``start_pos``, populates paged KV cache ---

    def _chunk_page_table(self, start_pos: int, seq_len: int) -> ttnn.Tensor:
        """Slice ``self.page_table`` to the blocks this chunk writes into.

        ``paged_fill_cache`` reads ``page_table[batch_idx]`` to find the physical blocks to fill,
        so the slice must cover exactly the blocks spanned by ``[start_pos, start_pos+seq_len)``.
        Requires ``start_pos`` to be a multiple of ``kv_block_size``.
        """
        block_size = self.args.kv_block_size
        if start_pos % block_size != 0:
            raise ValueError(f"start_pos ({start_pos}) must be a multiple of kv_block_size ({block_size})")
        block_start = start_pos // block_size
        # Ceiling-divide: a partial last chunk still writes a full block (extra slots are masked
        # off by causal attention since the chunk's Q seq_len bounds the active range).
        num_chunk_blocks = (seq_len + block_size - 1) // block_size
        return ttnn.slice(
            self.page_table,
            [0, block_start],
            [self.args.max_batch_size, block_start + num_chunk_blocks],
        )

    def _prefill_sdpa_program_config(self) -> ttnn.SDPAProgramConfig:
        """SDPA program config for chunked prefill.

        ``q_chunk_size`` divides ``chunk_start_idx`` (which is always a multiple of
        ``kv_block_size``) so this is always legal. ``k_chunk_size`` matches the cache's
        physical block size to stream K/V efficiently from paged storage.
        """
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=self.args.kv_block_size,
            k_chunk_size=self.args.kv_block_size,
        )

    def forward_prefill(
        self,
        x: ttnn.Tensor,
        *,
        start_pos: int = 0,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        """One prefill chunk. ``x``: ``(1, 1, S, hidden_size)``, ``S`` a multiple of ``kv_block_size``.

        Writes chunk K/V into the paged cache at logical positions ``[start_pos, start_pos+S)`` and
        runs chunked SDPA so the chunk's Q attends to **all** prior cached K/V positions plus its
        own with the correct block-causal mask.
        """
        S = int(x.shape[-2])
        Hq_local = self.args.n_local_heads
        Hkv_local = self.args.n_local_kv_heads

        act_mem = self._act_mem("prefill")
        qkv = self._project_qkv_prefill(x, seq_len=S)
        # Split fused ``[Q | K | V]`` (..., (Hq_local + 2*Hkv_local)*D) into heads.
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=Hq_local,
            num_kv_heads=Hkv_local,
            transpose_k_heads=False,
            memory_config=act_mem,
        )
        ttnn.deallocate(qkv)

        cos_q, sin_q, cos_k, sin_k = self.rotary_emb.get_prefill_tables(start_pos, S)
        q_heads, k_heads = self.rotary_emb.apply(q_heads, k_heads, cos_q, sin_q, cos_k, sin_k)
        # Free the RoPE prefill slices immediately. They sit in L1 (sliced from L1 source tables)
        # and would otherwise stay alive through SDPA + nlp_concat_heads + o_proj matmul,
        # where a growing CB region eventually collides with their L1 address.
        ttnn.deallocate(cos_q)
        ttnn.deallocate(sin_q)
        ttnn.deallocate(cos_k)
        ttnn.deallocate(sin_k)

        # Write this chunk's K, V into the paged cache. The chunk_page_table is a contiguous
        # slice of the full page_table covering only the blocks this chunk fills.
        chunk_pt = self._chunk_page_table(start_pos, S)
        ttnn.experimental.paged_fill_cache(self.k_cache, k_heads, chunk_pt, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(self.v_cache, v_heads, chunk_pt, batch_idx=user_id)
        ttnn.deallocate(chunk_pt)
        # K/V have been persisted; the fresh head tensors are no longer needed — SDPA reads from
        # the cache via the full page_table.
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # Chunked SDPA: Q chunk at positions ``[start_pos, start_pos+S)`` attends to all cached
        # K/V at ``[0, start_pos+S)`` with a block-causal mask. K/V are read from paged storage.
        # Q/K/V/page_table/chunk_start_idx must be positional (nanobind ``noconvert()`` on tensors).
        # Do not pass ``memory_config`` here; output defaults to interleaved DRAM (see ``test_sdpa_chunked``).
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_heads,
            self.k_cache,
            self.v_cache,
            self.page_table,
            start_pos,
            program_config=self._prefill_sdpa_program_config(),
            compute_kernel_config=self._compute_kernel_config,
        )
        attn = ttnn.to_memory_config(attn, act_mem)
        ttnn.deallocate(q_heads)

        # (B, Hq_local, S, D) -> (1, 1, S, Hq_local*D)
        attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=act_mem)
        ttnn.deallocate(attn)
        out = self._project_o(attn_flat, mode="prefill", seq_len=S)
        ttnn.deallocate(attn_flat)
        return out

    # --- Decode: single token per user, reads + updates KV cache ---

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """``x``: ``(1, 1, tile_h, hidden_size)`` with logical batch ``B`` in dim 2; ``current_pos`` is ``[B]``."""
        batch_size = int(current_pos.shape[0])
        act_mem = self._act_mem("decode")
        Hq_local = self.args.n_local_heads
        Hkv_local = self.args.n_local_kv_heads

        # TILE layout pads height (e.g. 32 for batch=1); do not squeeze to (1, 1, 1, H).
        x = _reshape_decode_batch_dim(x, batch_size)

        xqkv = self._project_qkv_decode(x)
        fused_dim = int(xqkv.shape[-1])
        xqkv = _reshape_decode_batch_dim(xqkv, batch_size)

        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=Hq_local,
            num_kv_heads=Hkv_local,
            memory_config=_DECODE_QKV_HEADS_MEM_CONFIG,
        )
        ttnn.deallocate(xqkv)

        cos_q, sin_q, cos_k, sin_k = self.rotary_emb.get_decode_tables(current_pos)
        cos_q, sin_q = _shard_decode_rope_tables(cos_q, sin_q, batch_size=batch_size, head_dim=self.args.head_dim)
        cos_k, sin_k = _shard_decode_rope_tables(cos_k, sin_k, batch_size=batch_size, head_dim=self.args.head_dim)
        trans_mat = self.rotary_emb.get_sharded_trans_mat(batch_size)
        q_heads = ttnn.experimental.rotary_embedding_llama(q_heads, cos_q, sin_q, trans_mat, is_decode_mode=True)
        k_heads = ttnn.experimental.rotary_embedding_llama(k_heads, cos_k, sin_k, trans_mat, is_decode_mode=True)
        # Mirror forward_prefill: release the sharded RoPE tables before SDPA / o_proj so their L1
        # addresses don't collide with the next program's growing CB region across many decode steps
        # or a follow-up prefill (validate_circular_buffer_region throw on the agent demo).
        ttnn.deallocate(cos_q)
        ttnn.deallocate(sin_q)
        ttnn.deallocate(cos_k)
        ttnn.deallocate(sin_k)

        pos_tt = current_pos
        # Paged decode write: ``paged_update_cache`` consults ``page_table[batch_idx, pos // block]``
        # to find the physical block and writes the new K/V at ``pos % block``.
        ttnn.experimental.paged_update_cache(
            self.k_cache, k_heads, update_idxs_tensor=pos_tt, page_table=self.page_table
        )
        ttnn.experimental.paged_update_cache(
            self.v_cache, v_heads, update_idxs_tensor=pos_tt, page_table=self.page_table
        )
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        sdpa_out_mem = get_sdpa_decode_output_mem_config(self.args, batch_size)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            self.k_cache,
            self.v_cache,
            page_table_tensor=self.page_table,
            cur_pos_tensor=pos_tt,
            scale=self.args.attn_scale,
            program_config=self._sdpa_decode_program_config,
            compute_kernel_config=self._sdpa_decode_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads)
        attn = ttnn.to_memory_config(attn, sdpa_out_mem)

        # (1, B, Hq_local, D) -> (1, 1, B, Hq_local*D)
        attn_flat = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=Hq_local)
        ttnn.deallocate(attn)
        attn_flat = ttnn.to_memory_config(attn_flat, act_mem)
        out = self._project_o(attn_flat, mode="decode", seq_len=1)
        ttnn.deallocate(attn_flat)
        return out

    # --- Public dispatch ---

    def __call__(
        self,
        x: ttnn.Tensor,
        *,
        mode: str = "decode",
        start_pos: int = 0,
        current_pos: Optional[ttnn.Tensor] = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        if mode == "prefill":
            return self.forward_prefill(x, start_pos=start_pos, user_id=user_id)
        if mode == "decode":
            if current_pos is None:
                raise ValueError("decode mode requires current_pos")
            return self.forward_decode(x, current_pos)
        raise ValueError(f"Unknown mode {mode!r}")

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)

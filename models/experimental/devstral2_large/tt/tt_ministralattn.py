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
    upload_kv_cache_buffer,
    upload_matmul_weight,
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


def _interleave_kv_for_tp(k_w: torch.Tensor, v_w: torch.Tensor, tp: int) -> torch.Tensor:
    """Arrange ``[K | V]`` HF rows so a contiguous TP-chunk slice puts ``(K_local, V_local)``
    on each device.

    Naive ``cat([k_w, v_w], dim=0)`` keeps all of K in the first half and all of V in the second
    half — strip-sharding that along dim 0 gives some devices all-K and others all-V. Instead,
    split each of K and V into ``tp`` row-chunks and zip them: contiguous chunk ``i`` of the
    result is exactly device ``i``'s ``(K_local rows ; V_local rows)``.
    """
    if tp <= 1:
        return torch.cat([k_w, v_w], dim=0)
    k_chunks = torch.chunk(k_w, tp, dim=0)
    v_chunks = torch.chunk(v_w, tp, dim=0)
    pieces: list[torch.Tensor] = []
    for kc, vc in zip(k_chunks, v_chunks):
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

        # (2) Fuse K and V into a single column-parallel matmul, with rows interleaved per
        # TP chunk so a contiguous TP strip-shard puts ``(K_local | V_local)`` on each device.
        # The naive ``cat([k_w, v_w], dim=0)`` would give device 0 all-K and device N-1 all-V.
        kv_w = _interleave_kv_for_tp(k_w, v_w, args.tp)

        wp = resolve_weight_cache_path(weight_cache_path, args)
        pfx = prefix
        self.q_proj = _to_tt_weight(
            q_w, mesh_device, args, self.dtype, shard_dim=-1, weight_cache_path=wp, cache_key=f"{pfx}q_proj"
        )
        self.kv_proj = _to_tt_weight(
            kv_w, mesh_device, args, self.dtype, shard_dim=-1, weight_cache_path=wp, cache_key=f"{pfx}kv_proj"
        )
        self.o_proj = _to_tt_weight(
            o_w, mesh_device, args, self.dtype, shard_dim=-2, weight_cache_path=wp, cache_key=f"{pfx}o_proj"
        )

        cache_shape = (
            args.max_batch_size,
            args.n_local_kv_heads,
            args.max_seq_len,
            args.head_dim,
        )
        self.k_cache = upload_kv_cache_buffer(
            cache_shape,
            mesh_device,
            dtype=args.kv_cache_dtype,
            weight_cache_path=wp,
            cache_key=f"{pfx}k_cache",
        )
        self.v_cache = upload_kv_cache_buffer(
            cache_shape,
            mesh_device,
            dtype=args.kv_cache_dtype,
            weight_cache_path=wp,
            cache_key=f"{pfx}v_cache",
        )

        self._compute_kernel_config = get_compute_kernel_config(mesh_device)
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
    ) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            dtype=self.args.activation_dtype,
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
            compute_kernel_config=self._compute_kernel_config,
        )

    def _project_qkv_prefill(self, x: ttnn.Tensor, *, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return ``(q, kv_fused)`` for ``nlp_create_qkv_heads``."""
        q = self._linear(x, self.q_proj, mode="prefill", kind="qkv", seq_len=seq_len)
        kv = self._linear(x, self.kv_proj, mode="prefill", kind="qkv", seq_len=seq_len)
        return q, kv

    def _project_qkv_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Fused QKV activation for ``nlp_create_qkv_heads_decode`` (Q and KV matmuls + concat)."""
        q = self._linear(x, self.q_proj, mode="decode", kind="qkv")
        kv = self._linear(x, self.kv_proj, mode="decode", kind="qkv")
        return ttnn.concat([q, kv], dim=-1, memory_config=self._act_mem("decode"))

    def _project_o(self, attn_out_flat: ttnn.Tensor, *, mode: str, seq_len: int = 1) -> ttnn.Tensor:
        dense = self._linear(attn_out_flat, self.o_proj, mode=mode, kind="o_proj", seq_len=seq_len)
        return all_reduce_replicate(
            dense,
            mesh_device=self.mesh_device,
            tt_ccl=self.tt_ccl,
            dim=3,
            cluster_axis=self.args.cluster_axis,
            topology=self.args.ccl_topology,
            memory_config=self.args.get_ccl_output_mem_config(mode, self.mesh_device),
        )

    # --- Prefill: full sequence, populates KV cache from ``start_pos`` ---

    def forward_prefill(
        self,
        x: ttnn.Tensor,
        *,
        start_pos: int = 0,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        """``x``: ``(1, 1, S, hidden_size)``, replicated across TP. Returns same shape."""
        B = 1
        S = int(x.shape[-2])
        D = self.args.head_dim
        Hq_local = self.args.n_local_heads
        Hkv_local = self.args.n_local_kv_heads

        act_mem = self._act_mem("prefill")
        q, kv = self._project_qkv_prefill(x, seq_len=S)
        # Split into heads. ``input`` = Q (..., Hq_local*D); ``input_kv`` = fused [K|V] (..., 2*Hkv_local*D).
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            q,
            kv,
            num_heads=Hq_local,
            num_kv_heads=Hkv_local,
            transpose_k_heads=False,
            memory_config=act_mem,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(kv)

        cos_q, sin_q, cos_k, sin_k = self.rotary_emb.get_prefill_tables(start_pos, S)
        q_heads, k_heads = self.rotary_emb.apply(q_heads, k_heads, cos_q, sin_q, cos_k, sin_k)

        # Write K/V into the cache for future decode steps. ``ttnn.fill_cache`` writes the entire
        # ``src`` tensor into the cache's ``user_id`` batch slot starting at position 0. Chunked
        # prefill (start_pos > 0) would need ``paged_update_cache`` with a position vector; not
        # supported in this baseline.
        if start_pos != 0:
            raise NotImplementedError("Chunked prefill (start_pos > 0) is not implemented in this baseline.")
        ttnn.fill_cache(self.k_cache, k_heads, user_id)
        ttnn.fill_cache(self.v_cache, v_heads, user_id)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=True,
            scale=self.args.attn_scale,
            compute_kernel_config=self._compute_kernel_config,
            memory_config=act_mem,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

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
        """``x``: ``(1, 1, B, hidden_size)``; ``current_pos`` device int32 tensor ``[B]``."""
        batch_size = int(current_pos.shape[0])
        act_mem = self._act_mem("decode")
        Hq_local = self.args.n_local_heads
        Hkv_local = self.args.n_local_kv_heads

        hidden = int(x.shape[-1])
        # ``nlp_create_qkv_heads_decode`` expects ``(1, seq_len, batch, fused_qkv_dim)``.
        x = ttnn.reshape(x, (1, 1, batch_size, hidden))

        xqkv = self._project_qkv_decode(x)
        fused_dim = int(xqkv.shape[-1])
        xqkv = ttnn.reshape(xqkv, (1, 1, batch_size, fused_dim))

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

        pos_tt = current_pos
        # Separate updates: fused op requires K/V head tensors on non-overlapping core grids, but
        # ``nlp_create_qkv_heads_decode`` places them on the same height-sharded L1 cores.
        ttnn.experimental.paged_update_cache(self.k_cache, k_heads, update_idxs_tensor=pos_tt, page_table=None)
        ttnn.experimental.paged_update_cache(self.v_cache, v_heads, update_idxs_tensor=pos_tt, page_table=None)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        sdpa_out_mem = get_sdpa_decode_output_mem_config(self.args, batch_size)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads,
            self.k_cache,
            self.v_cache,
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

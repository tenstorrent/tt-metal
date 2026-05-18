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
from models.experimental.devstral2_large.tt.model_args import Devstral2Args
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import (
    TtRotaryEmbedding,
    permute_split_half_to_interleaved,
)

__all__ = ["TtAttention"]


# --- Weight loading ---


def _to_tt_weight(
    w_hf: torch.Tensor,
    mesh_device,
    args: Devstral2Args,
    dtype: ttnn.DataType,
    *,
    shard_dim: Optional[int],
) -> ttnn.Tensor:
    """Upload HF Linear weight ``(out, in)`` as TTNN ``(in, out)``, optionally TP-sharded.

    ``shard_dim`` is in the TTNN ``(in, out)`` orientation:
      - ``shard_dim = -1`` → column-parallel (split the output dim ``out``).
      - ``shard_dim = -2`` → row-parallel (split the input dim ``in``).
      - ``shard_dim = None`` → replicate.
    """
    w = w_hf.to(torch.bfloat16).T.contiguous()  # (in, out)
    if shard_dim is None:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        if args.cluster_axis == 1:
            dims = (None, shard_dim)
        else:
            dims = (shard_dim, None)
        mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=args.mesh_shape)
    return ttnn.from_torch(
        w,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
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
        weight_cache_path: Optional[str] = None,  # noqa: ARG002  (reserved for cache hookup)
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

        self.q_proj = _to_tt_weight(q_w, mesh_device, args, self.dtype, shard_dim=-1)
        self.kv_proj = _to_tt_weight(kv_w, mesh_device, args, self.dtype, shard_dim=-1)
        self.o_proj = _to_tt_weight(o_w, mesh_device, args, self.dtype, shard_dim=-2)

        # Per-device KV cache: [batch, n_local_kv_heads, max_seq_len, head_dim], DRAM, zero-init.
        cache_shape = (
            args.max_batch_size,
            args.n_local_kv_heads,
            args.max_seq_len,
            args.head_dim,
        )
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)
        self.k_cache = ttnn.from_torch(
            zeros,
            device=mesh_device,
            dtype=args.kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.v_cache = ttnn.from_torch(
            zeros,
            device=mesh_device,
            dtype=args.kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # --- Projections (shared by prefill / decode) ---

    def _project_qkv(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return ``(q, kv_fused)`` so callers can pass them to ``nlp_create_qkv_heads`` directly."""
        q = ttnn.linear(
            x,
            self.q_proj,
            dtype=self.args.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        kv = ttnn.linear(
            x,
            self.kv_proj,
            dtype=self.args.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        return q, kv

    def _project_o(self, attn_out_flat: ttnn.Tensor) -> ttnn.Tensor:
        dense = ttnn.linear(
            attn_out_flat,
            self.o_proj,
            dtype=self.args.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        # Each device produced a partial sum (rowwise TP); reduce + replicate.
        return all_reduce_replicate(
            dense,
            mesh_device=self.mesh_device,
            tt_ccl=self.tt_ccl,
            dim=3,
            cluster_axis=self.args.cluster_axis,
            topology=self.args.ccl_topology,
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

        q, kv = self._project_qkv(x)
        # Split into heads. ``input`` = Q (..., Hq_local*D); ``input_kv`` = fused [K|V] (..., 2*Hkv_local*D).
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            q,
            kv,
            num_heads=Hq_local,
            num_kv_heads=Hkv_local,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
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
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # (B, Hq_local, S, D) -> (1, 1, S, Hq_local*D)
        attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        out = self._project_o(attn_flat)
        ttnn.deallocate(attn_flat)
        return out

    # --- Decode: single token per user, reads + updates KV cache ---

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos_host: torch.Tensor,
    ) -> ttnn.Tensor:
        """``x``: ``(1, B, 1, hidden_size)``, replicated across TP. ``current_pos_host`` shape ``[B]``."""
        B = self.args.max_batch_size
        D = self.args.head_dim
        Hq_local = self.args.n_local_heads
        Hkv_local = self.args.n_local_kv_heads

        q, kv = self._project_qkv(x)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            q,
            kv,
            num_heads=Hq_local,
            num_kv_heads=Hkv_local,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(kv)

        cos_q, sin_q, cos_k, sin_k = self.rotary_emb.get_decode_tables(current_pos_host)
        q_heads, k_heads = self.rotary_emb.apply(q_heads, k_heads, cos_q, sin_q, cos_k, sin_k)

        # Update K/V cache at ``current_pos`` for each user.
        pos_tt = ttnn.from_torch(
            current_pos_host.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.experimental.paged_update_cache(self.k_cache, k_heads, update_idxs_tensor=pos_tt, page_table=None)
        ttnn.experimental.paged_update_cache(self.v_cache, v_heads, update_idxs_tensor=pos_tt, page_table=None)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=pos_tt,
            scale=self.args.attn_scale,
            compute_kernel_config=self._compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads)

        # (1, B, Hq_local, D) -> (1, 1, B, Hq_local*D)
        attn_flat = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=Hq_local)
        ttnn.deallocate(attn)
        out = self._project_o(attn_flat)
        ttnn.deallocate(attn_flat)
        return out

    # --- Public dispatch ---

    def __call__(
        self,
        x: ttnn.Tensor,
        *,
        mode: str = "decode",
        start_pos: int = 0,
        current_pos_host: Optional[torch.Tensor] = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        if mode == "prefill":
            return self.forward_prefill(x, start_pos=start_pos, user_id=user_id)
        if mode == "decode":
            if current_pos_host is None:
                raise ValueError("decode mode requires current_pos_host")
            return self.forward_decode(x, current_pos_host)
        raise ValueError(f"Unknown mode {mode!r}")

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)

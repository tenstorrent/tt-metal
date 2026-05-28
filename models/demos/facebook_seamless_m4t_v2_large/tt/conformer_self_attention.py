# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 Conformer self-attention block.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::conformer_self_attention_forward``,
which is the multi-head self-attention used inside the W2v-BERT-2.0 Conformer
encoder layers of SeamlessM4T-v2's speech encoder. It is a standard
4-projection MHA (``linear_q``/``linear_k``/``linear_v``/``linear_out``) plus
an additive ``relative_key`` positional-bias term added to the attention
logits BEFORE softmax.

The op sequence intentionally matches HF:

    Q = linear_q(x).view(B, T, H, D).transpose(1, 2)
    K = linear_k(x).view(B, T, H, D).transpose(1, 2)
    V = linear_v(x).view(B, T, H, D).transpose(1, 2)

    scores = (Q @ K^T) / sqrt(D)
    pos_emb[l, r, d] = distance_embedding_weight[clamp(r-l, -L, +R) + L, d]
    rel = einsum("bhld,lrd->bhlr", Q, pos_emb)
    scores = scores + rel / sqrt(D)
    scores += attention_mask                          # additive log-mask
    attn   = softmax(scores, dim=-1)
    out    = (attn @ V).transpose(1, 2).reshape(B, T, H*D)
    out    = linear_out(out)

The fused ``ttnn.transformer.scaled_dot_product_attention`` op does not
support adding an arbitrary positional-bias term, so this block does the
QK^T -> scale -> +bias -> +mask -> softmax -> @V kernel manually using
``ttnn.matmul`` / ``ttnn.add`` / ``ttnn.softmax``.

The relative-position embedding tensor ``pos_emb`` is precomputed on host
at ``__init__`` time (it depends only on ``seq_len``, ``head_dim``,
``left_max_position_embeddings`` and ``right_max_position_embeddings``)
and uploaded to device DRAM in a layout that lets the einsum be expressed
as a single batched 4D matmul:

    Q_reorg  : [1, T_q, B*H,   D  ]      (per-l "batch")
    P_reorg  : [1, T_q,   D, T_k ]      (P^T per l)
    BMM      : [1, T_q, B*H,  T_k]
    -> permute/reshape back to [B, H, T_q, T_k]
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule


def _build_relative_position_embedding(
    distance_embedding_weight,
    seq_len: int,
    left_max_position_embeddings: int,
    right_max_position_embeddings: int,
):
    """Host-side construction of the per-(l, r) positional embedding table.

    Replicates the ``F.embedding(clamp(r-l, -L, +R) + L, distance_embedding_weight)``
    step from the reference, returning a tensor of shape
    ``[seq_len, seq_len, head_dim]``. Uses only attribute access on the
    incoming torch tensor for the gather; no host math kernels touched in the
    hot path (this runs once at construction).
    """
    import torch as _torch  # local import; this file otherwise avoids torch

    L = int(left_max_position_embeddings)
    R = int(right_max_position_embeddings)
    position_ids_l = _torch.arange(seq_len, dtype=_torch.long).view(-1, 1)
    position_ids_r = _torch.arange(seq_len, dtype=_torch.long).view(1, -1)
    distance = position_ids_r - position_ids_l
    distance = distance.clamp(-L, R)
    indices = distance + L  # [seq_len, seq_len], values in [0, L+R]
    # Gather: distance_embedding_weight is [L+R+1, head_dim].
    pos_emb = distance_embedding_weight[indices]  # [seq_len, seq_len, head_dim]
    return pos_emb.contiguous()


class ConformerSelfAttention(LightweightModule):
    """SeamlessM4T-v2 Conformer self-attention with relative-key positional bias.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for seamless-m4t-v2-large).
        num_heads: number of attention heads (16 for -large).
        head_dim: per-head dim; must satisfy ``num_heads * head_dim == embed_dim``.
        seq_len: sequence length (fixed at construction so the positional bias
            table can be precomputed once).
        state_dict: nested mapping ``linear_q/linear_k/linear_v/linear_out`` ->
            ``{"weight": tensor, "bias": tensor}`` with HF-style
            (out_features, in_features) weight orientation; biases are 1D.
        distance_embedding_weight: learned ``[L+R+1, head_dim]`` table for the
            relative-position bias.
        left_max_position_embeddings: ``L`` clamp range on the left side
            (64 in v2-Large).
        right_max_position_embeddings: ``R`` clamp range on the right side
            (8 in v2-Large).
        position_embeddings_type: ``"relative_key"`` (default) or ``None``.
        batch_size: forward batch size. Used to size the broadcast layout of
            the precomputed positional-bias tensor (defaults to 1).
        weight_dtype: storage dtype for the four projection weights/biases AND
            the precomputed positional-bias table.
        weight_memory_config: where to place weights / pos-emb table.
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        seq_len: int,
        state_dict,
        distance_embedding_weight=None,
        left_max_position_embeddings: int = 64,
        right_max_position_embeddings: int = 8,
        position_embeddings_type: str = "relative_key",
        batch_size: int = 1,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.scaling = head_dim**-0.5
        self.inv_sqrt_d = head_dim**-0.5  # second division by sqrt(D) inside the rel-bias term
        self.position_embeddings_type = position_embeddings_type

        # HF stores Linear weight as (out, in); ttnn.linear expects (in, out)
        # so swap the last two dims on load. Biases stay 1D.
        def _load_proj(name):
            w = state_dict[name]["weight"]
            b = state_dict[name].get("bias")
            w_tt = ttnn.from_torch(
                w.transpose(0, 1).contiguous(),
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
            )
            b_tt = None
            if b is not None:
                b_tt = ttnn.from_torch(
                    b.reshape(1, -1),
                    device=device,
                    dtype=weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=weight_memory_config,
                )
            return w_tt, b_tt

        self.q_weight, self.q_bias = _load_proj("linear_q")
        self.k_weight, self.k_bias = _load_proj("linear_k")
        self.v_weight, self.v_bias = _load_proj("linear_v")
        self.o_weight, self.o_bias = _load_proj("linear_out")

        # Precompute the [seq_len, seq_len, head_dim] positional-bias table
        # on host and upload it in the [1, T_q, D, T_k] layout used by the
        # batched einsum matmul. We also pre-scale by 1/sqrt(D) so the kernel
        # only does one matmul + one add per layer-call.
        self._rel_bias_weight = None
        if position_embeddings_type == "relative_key":
            if distance_embedding_weight is None:
                raise ValueError("distance_embedding_weight is required when position_embeddings_type='relative_key'")
            pos_emb_table = _build_relative_position_embedding(
                distance_embedding_weight,
                seq_len=seq_len,
                left_max_position_embeddings=left_max_position_embeddings,
                right_max_position_embeddings=right_max_position_embeddings,
            )
            # pos_emb_table: [T_q, T_k, D] -> [T_q, D, T_k] for the matmul.
            pos_emb_t = pos_emb_table.permute(0, 2, 1).contiguous()  # [T_q, D, T_k]
            # Pre-scale: rel / sqrt(D) folds into the table.
            pos_emb_t = pos_emb_t * (head_dim**-0.5)
            # Add a leading singleton dim to get a 4D tensor [1, T_q, D, T_k].
            pos_emb_t = pos_emb_t.unsqueeze(0).contiguous()
            self._rel_bias_weight = ttnn.from_torch(
                pos_emb_t,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _project_and_split(self, x, weight, bias, batch, seq_len):
        """Project x with linear(weight,bias), reshape to [B, num_heads, S, head_dim]."""
        proj = ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        # proj is [B, S, embed_dim]. -> [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim].
        # Pin the reshape output to L1: tracy shows this reshape (~72us/call x3) at ~14% of
        # the encoder-layer kernel time when materialized to DRAM. Moving to L1 hides the
        # writeback by keeping the head-split tensor resident for the immediate transpose.
        proj = ttnn.reshape(proj, (batch, seq_len, self.num_heads, self.head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        proj = ttnn.transpose(proj, 1, 2)
        return proj

    def _relative_position_bias(self, q):
        """Compute the relative-position bias term added to attention scores.

        Equivalent (in math) to::

            rel = einsum("bhld,lrd->bhlr", q, pos_emb) / sqrt(D)

        where ``pos_emb`` has been pre-scaled by ``1/sqrt(D)`` and pre-permuted
        to ``[1, T_q, D, T_k]``. We reorganize ``q`` from ``[B, H, T_q, D]``
        into ``[1, T_q, B*H, D]`` so that a single batched matmul:

            [1, T_q, B*H, D] @ [1, T_q, D, T_k] -> [1, T_q, B*H, T_k]

        produces ``rel`` with the per-l matmul correctly fanned out across
        heads. Finally we permute / reshape back to ``[B, H, T_q, T_k]``.
        """
        # q: [B, H, T_q, D]. Move T_q forward: -> [T_q, B, H, D].
        # ttnn.permute supports arbitrary 4D permutations.
        q_perm = ttnn.permute(q, (2, 0, 1, 3))  # [T_q, B, H, D]
        # Reshape to [1, T_q, B*H, D].
        q_perm = ttnn.reshape(q_perm, (1, self.seq_len, self.batch_size * self.num_heads, self.head_dim))

        # Batched matmul with broadcast in dim 0 (both have size 1):
        #   [1, T_q, B*H, D] @ [1, T_q, D, T_k] -> [1, T_q, B*H, T_k]
        rel = ttnn.matmul(
            q_perm,
            self._rel_bias_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q_perm)

        # rel: [1, T_q, B*H, T_k] -> permute to [1, B*H, T_q, T_k]
        rel = ttnn.permute(rel, (0, 2, 1, 3))
        # -> [B, H, T_q, T_k]
        rel = ttnn.reshape(rel, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        return rel

    def forward(self, hidden_states, attention_mask=None):
        """Run conformer self-attention.

        Args:
            hidden_states: ttnn tensor of shape ``[B, T, embed_dim]`` in
                TILE_LAYOUT.
            attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T, T]`` representing an additive log-mask.
        Returns:
            ttnn tensor of shape ``[B, T, embed_dim]``.
        """
        batch = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Q, K, V projections + reshape -> [B, H, S, D].
        q = self._project_and_split(hidden_states, self.q_weight, self.q_bias, batch, seq_len)
        k = self._project_and_split(hidden_states, self.k_weight, self.k_bias, batch, seq_len)
        v = self._project_and_split(hidden_states, self.v_weight, self.v_bias, batch, seq_len)

        # K^T: [B, H, S, D] -> [B, H, D, S]
        k_t = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)

        # Content scores = (Q @ K^T) * (1/sqrt(D)).
        scores = ttnn.matmul(
            q,
            k_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(k_t)
        scores = ttnn.multiply(scores, self.scaling)

        # Relative-position bias.
        if self.position_embeddings_type == "relative_key":
            rel = self._relative_position_bias(q)
            scores = ttnn.add(scores, rel)
            ttnn.deallocate(rel)

        ttnn.deallocate(q)

        # Optional additive log-mask.
        if attention_mask is not None:
            scores = ttnn.add(scores, attention_mask)

        attn = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)

        # Context = attn @ V: [B, H, T, D]
        ctx = ttnn.matmul(
            attn,
            v,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        # [B, H, T, D] -> [B, T, H, D] -> [B, T, H*D].
        # Pin the head-merge reshape output to L1 (tracy: ~75us at DRAM, same pattern as
        # the q/k/v split reshapes above).
        ctx = ttnn.transpose(ctx, 1, 2)
        ctx = ttnn.reshape(ctx, (batch, seq_len, self.embed_dim), memory_config=ttnn.L1_MEMORY_CONFIG)

        out = ttnn.linear(
            ctx,
            self.o_weight,
            bias=self.o_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(ctx)
        return out

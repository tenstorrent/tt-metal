# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr Qwen2 LM self-attention block.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`attention_forward` (+ ``_repeat_kv``, ``apply_rotary_pos_emb_lm``)

This is the Qwen2 decoder self-attention (distinct from the bias-free,
block-diagonal vision attention in ``tt/vision_attention.py``):

    GQA  : 12 query heads / 2 KV heads, head_dim 128, hidden 1536.
    bias : q_proj, k_proj, v_proj all carry BIAS; o_proj has NO bias.
    RoPE : 1D rotary (theta 1e6) on Q and K — apply_rotary_pos_emb_lm.
    mask : causal additive mask (0 / -inf), is_causal.

Pipeline (matching the eager reference exactly):

    q = x @ Wq^T + bq ; k = x @ Wk^T + bk ; v = x @ Wv^T + bv
    split via nlp_create_qkv_heads (12 q heads, 2 kv heads, transpose_k=False)
    q,k = rope_1d(q,k)                         # cos/sin broadcast over heads
    k,v = repeat_kv(2 -> 12)                    # GQA expansion
    attn = softmax(q k^T / sqrt(head_dim) + causal_mask) @ v
    out  = concat_heads(attn) @ Wo^T           # output proj, no bias

The fused QKV weight is built host-side as cat([Wq, Wk, Wv]) so a single
``ttnn.linear`` produces the [seq, (nh+2*nkv)*hd] tensor consumed by
``nlp_create_qkv_heads``. cos/sin tables and the causal mask are uploaded at
construction time (analogous to loading weights). forward() runs entirely in
ttnn ops (no host matmul / softmax / numpy / torch.nn.functional).

Reference TTNN technique: models/demos/rednote_hilab_dots.ocr/tt/vision_attention.py
"""
import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtAttention(LightweightModule):
    """dots.ocr Qwen2 LM self-attention (GQA + QKV bias + 1D RoPE, causal).

    Args:
        device: ttnn Device or MeshDevice.
        q_weight/k_weight/v_weight: torch [out, in] projection weights.
        q_bias/k_bias/v_bias: torch [out] projection biases.
        o_weight: torch [hidden, hidden] output projection weight (no bias).
        cos, sin: torch [seq, head_dim] 1D-RoPE tables (theta 1e6).
        attention_mask: torch additive causal mask [1, 1, seq, seq].
        seq_len: sequence length.
        num_heads: 12 query heads.
        num_kv_heads: 2 KV heads.
        head_dim: 128.
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        o_weight,
        cos,
        sin,
        attention_mask,
        seq_len,
        num_heads: int = 12,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.n_rep = num_heads // num_kv_heads
        self.hidden = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        # Fused QKV weight: cat([Wq, Wk, Wv]) along output dim -> [hidden + 2*kv, in].
        # ttnn.linear computes x @ W^T when we store the torch weight transposed
        # (ttnn linear expects [in, out]).
        qkv_w = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [(nh+2nkv)*hd, in]
        self.qkv_weight = ttnn.as_tensor(
            qkv_w.transpose(0, 1).contiguous(),  # [in, (nh+2nkv)*hd]
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )
        qkv_b = torch.cat([q_bias, k_bias, v_bias], dim=0)  # [(nh+2nkv)*hd]
        self.qkv_bias = ttnn.as_tensor(
            qkv_b.reshape(1, -1).contiguous(),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        self.o_weight = ttnn.as_tensor(
            o_weight.transpose(0, 1).contiguous(),  # [hidden, hidden]
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # RoPE cos/sin tables: [seq, head_dim] -> [1, 1, seq, head_dim] broadcast
        # over heads (after q/k are [1, nh, seq, hd]).
        cos = cos.reshape(1, 1, seq_len, head_dim)
        sin = sin.reshape(1, 1, seq_len, head_dim)
        self.cos = ttnn.as_tensor(
            cos,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )
        self.sin = ttnn.as_tensor(
            sin,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # Additive causal mask [1, num_heads, seq, seq].
        mask = attention_mask.reshape(1, 1, seq_len, seq_len)
        mask = mask.expand(1, num_heads, seq_len, seq_len).contiguous()
        self.attn_mask = ttnn.as_tensor(
            mask,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # fp32 compute to match the reference float path.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _rotate_half(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2]) on the last dim."""
        d = self.head_dim
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        neg_x2 = ttnn.neg(x2)
        return ttnn.concat([neg_x2, x1], dim=-1)

    def _apply_rope(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, n_heads, seq, head_dim]; cos/sin broadcast over heads."""
        cos_term = ttnn.mul(x, self.cos)
        rot = self._rotate_half(x)
        sin_term = ttnn.mul(rot, self.sin)
        return ttnn.add(cos_term, sin_term)

    def _repeat_kv(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, num_kv_heads, seq, head_dim] -> [1, num_heads, seq, head_dim].

        Mirrors _repeat_kv: expand each KV head n_rep times along the head dim,
        interleaved as [kv0]*n_rep, [kv1]*n_rep, ... so head h maps to kv h//n_rep.
        """
        if self.n_rep == 1:
            return x
        seq = self.seq_len
        hd = self.head_dim
        nkv = self.num_kv_heads
        # [1, nkv, seq, hd] -> [1, nkv, 1, seq, hd] -> expand n_rep -> reshape.
        x = ttnn.reshape(x, (1, nkv, 1, seq, hd))
        x = ttnn.repeat(x, ttnn.Shape((1, 1, self.n_rep, 1, 1)))
        x = ttnn.reshape(x, (1, nkv * self.n_rep, seq, hd))
        return x

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, hidden] (TILE layout) -> [seq, hidden]."""
        seq = self.seq_len
        nh = self.num_heads
        nkv = self.num_kv_heads
        hd = self.head_dim

        # Fused QKV projection with bias: [seq, hidden] @ [hidden, (nh+2nkv)*hd].
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )  # [seq, (nh + 2*nkv)*hd]

        # GQA head split. The fused weight is cat([Wq, Wk, Wv]) so the columns
        # are contiguous: first nh*hd are Q, next nkv*hd are K, last nkv*hd V.
        # Mirror the reference's view(bsz, q_len, heads, hd).transpose(1, 2).
        q = qkv[:, : nh * hd]
        k = qkv[:, nh * hd : (nh + nkv) * hd]
        v = qkv[:, (nh + nkv) * hd :]

        # -> [1, seq, heads, hd] for rope (cos/sin broadcast over heads).
        # The head-split reshapes are the block's top hotspot (tracy: 38% of
        # kernel time when left DRAM-interleaved); pin their outputs to L1 so
        # the downstream permute/RoPE/repeat_kv chain reads from L1 not DRAM.
        q = ttnn.reshape(q, (1, seq, nh, hd), memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.reshape(k, (1, seq, nkv, hd), memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.reshape(v, (1, seq, nkv, hd), memory_config=ttnn.L1_MEMORY_CONFIG)

        # -> [1, heads, seq, hd] (transpose seq<->heads) for batched attention.
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # 1D RoPE on q and k (cos/sin broadcast over heads).
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # GQA expansion 2 -> 12.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # attn = softmax(q k^T * scale + causal_mask) @ v.  k^T -> [1, nh, hd, seq]
        # Keep the QK^T scores in fp32: real-weight K reaches ±320 (large k_proj
        # bias), so the pre-softmax scores have a wide dynamic range and bf16
        # rounding of the scores is the dominant PCC loss (drops ~0.984 -> 0.998
        # when fp32). fp32_dest_acc already accumulates the dot product in fp32;
        # storing the matmul output fp32 preserves it through scale/mask/softmax.
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        scores = ttnn.matmul(
            q,
            k_t,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.float32,
        )
        scores = ttnn.mul(scores, self.scale)
        scores = ttnn.add(scores, self.attn_mask)
        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)
        attn = ttnn.matmul(
            probs,
            v,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )  # [1, nh, seq, hd]

        # Concat heads: [1, nh, seq, hd] -> [1, seq, nh, hd] -> [seq, hidden].
        # Pin the head-merge reshape output to L1 so the o_proj matmul reads
        # its input from L1 rather than DRAM-interleaved.
        attn = ttnn.permute(attn, (0, 2, 1, 3))
        attn = ttnn.reshape(attn, (seq, nh * hd), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Output projection (no bias): [seq, hidden] @ [hidden, hidden].
        out = ttnn.linear(
            attn,
            self.o_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        return out

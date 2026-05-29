# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr DotsVisionTransformer attention block.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`vision_attention_forward`

Pipeline (matching the eager reference exactly):

    qkv = x @ qkv.weight^T                       # fused QKV, no bias
    q, k, v = split into [seq, num_heads, head_dim]
    q = rope2d(q); k = rope2d(k)                 # 2D vision RoPE (theta 1e4)
    attn = softmax(q k^T / sqrt(head_dim) + block_diag_mask) @ v
    out  = attn @ proj.weight^T                  # output proj, no bias

embed_dim 1536, num_heads 12, head_dim 128. is_causal = False; the attention is
bidirectional but block-diagonal over the cu_seqlens-defined image blocks.

Vision RoPE:
    freqs [seq, head_dim//2] -> cos/sin -> repeat last dim x2 -> [seq, head_dim]
    out = x * cos + rotate_half(x) * sin
    rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])

The cos/sin tables and the additive block-diagonal mask are derived from the
static inputs (rotary_pos_emb freqs and cu_seqlens) and are precomputed on host
at construction time, then uploaded to device — analogous to loading the
RMSNorm gamma weight. The forward() runs entirely with ttnn ops (no host
matmul / softmax / numpy / torch.nn.functional).

Reference TTNN impl this follows: models/demos/qwen25_vl/tt/vision_attention.py
"""
import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _build_rope_tables(rotary_pos_emb: torch.Tensor, head_dim: int):
    """Reproduce apply_rotary_pos_emb_vision's cos/sin expansion on host.

    rotary_pos_emb: [seq, head_dim//2] freqs table.
    Returns cos, sin each [seq, head_dim] (fp32), matching:
        cos = freqs.cos().repeat(..., 2)  # last dim doubled
    """
    freqs = rotary_pos_emb.float()  # [seq, head_dim//2]
    cos = freqs.cos()
    sin = freqs.sin()
    # reference: cos.unsqueeze(1).repeat(1, 1, 2) over a [seq, head_dim//2] -> [seq, 1, head_dim]
    # which is simply concatenating the table with itself along the last dim.
    cos = torch.cat([cos, cos], dim=-1)  # [seq, head_dim]
    sin = torch.cat([sin, sin], dim=-1)  # [seq, head_dim]
    assert cos.shape[-1] == head_dim, (cos.shape, head_dim)
    return cos, sin


def _build_block_diag_mask(cu_seqlens: torch.Tensor, seq_length: int) -> torch.Tensor:
    """Additive block-diagonal attention mask [1, 1, seq, seq].

    0.0 inside a block (token i and j in the same cu_seqlens segment),
    a large negative value across blocks. Bidirectional within each block.
    """
    neg = torch.finfo(torch.float32).min
    mask = torch.full([1, 1, seq_length, seq_length], neg, dtype=torch.float32)
    cu = cu_seqlens.tolist()
    for i in range(1, len(cu)):
        a, b = cu[i - 1], cu[i]
        mask[..., a:b, a:b] = 0.0
    return mask


class TtVisionAttention(LightweightModule):
    """dots.ocr vision attention.

    Args:
        device: ttnn Device or MeshDevice.
        qkv_weight: torch.Tensor [3*dim, dim] (fused QKV, no bias).
        proj_weight: torch.Tensor [dim, dim] (output proj, no bias).
        rotary_pos_emb: torch.Tensor [seq, head_dim//2] vision rope freqs.
        cu_seqlens: int tensor of cumulative seqlens (e.g. [0, 96, 256]).
        seq_length: number of patch tokens.
        num_heads: 12.
        head_dim: 128.
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        qkv_weight,
        proj_weight,
        rotary_pos_emb,
        cu_seqlens,
        seq_length,
        num_heads: int = 12,
        head_dim: int = 128,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_length = seq_length
        self.dim = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        # Fused QKV weight: ttnn.linear computes x @ W^T when we pass the
        # torch weight transposed (ttnn linear expects [in, out]).
        self.qkv_weight = ttnn.as_tensor(
            qkv_weight.transpose(0, 1).contiguous(),  # [dim, 3*dim]
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )
        self.proj_weight = ttnn.as_tensor(
            proj_weight.transpose(0, 1).contiguous(),  # [dim, dim]
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # RoPE cos/sin tables: [seq, head_dim] -> broadcast over heads as
        # [1, seq, 1, head_dim]. Stored bf16 TILE for elementwise on device.
        cos, sin = _build_rope_tables(rotary_pos_emb, head_dim)
        cos = cos.reshape(1, seq_length, 1, head_dim)
        sin = sin.reshape(1, seq_length, 1, head_dim)
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

        # Additive block-diagonal mask [1, num_heads, seq, seq].
        mask = _build_block_diag_mask(cu_seqlens, seq_length)  # [1,1,seq,seq]
        mask = mask.expand(1, num_heads, seq_length, seq_length).contiguous()
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
        """x: [1, seq, num_heads, head_dim]; cos/sin broadcast over heads."""
        cos_term = ttnn.mul(x, self.cos)
        rot = self._rotate_half(x)
        sin_term = ttnn.mul(rot, self.sin)
        return ttnn.add(cos_term, sin_term)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, dim] (TILE layout) -> [seq, dim]."""
        seq = self.seq_length
        nh = self.num_heads
        hd = self.head_dim

        # Fused QKV projection: [seq, dim] @ [dim, 3*dim] -> [seq, 3*dim].
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        # Split into q, k, v each [seq, nh*hd]. The reference reshapes
        # qkv -> [seq, 3, nh, hd]; contiguous QKV ordering means the first
        # nh*hd columns are q, next k, next v.
        # This head-split reshape is the block's dominant op (~33% of kernel
        # time when left DRAM-interleaved); pin the output to L1 so the
        # downstream slice/RoPE chain reads from L1 instead of DRAM.
        qkv = ttnn.reshape(qkv, (seq, 3, nh, hd), memory_config=ttnn.L1_MEMORY_CONFIG)
        q = qkv[:, 0, :, :]  # [seq, nh, hd]
        k = qkv[:, 1, :, :]
        v = qkv[:, 2, :, :]

        # -> [1, seq, nh, hd] for rope (cos/sin broadcast over heads).
        q = ttnn.reshape(q, (1, seq, nh, hd))
        k = ttnn.reshape(k, (1, seq, nh, hd))
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # -> [1, nh, seq, hd] (transpose seq<->heads) for batched attention.
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (1, seq, nh, hd))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # attn = softmax(q k^T * scale + mask) @ v.  k^T -> [1, nh, hd, seq]
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        scores = ttnn.matmul(
            q,
            k_t,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
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

        # -> [seq, nh*hd]. Head-merge reshape; pin to L1 so the output proj
        # reads from L1 instead of a DRAM-interleaved coalesce.
        attn = ttnn.permute(attn, (0, 2, 1, 3))  # [1, seq, nh, hd]
        attn = ttnn.reshape(attn, (seq, nh * hd), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Output projection: [seq, dim] @ [dim, dim] -> [seq, dim].
        out = ttnn.linear(
            attn,
            self.proj_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        return out

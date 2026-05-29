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

Attention is computed with memory-efficient flash attention
(``ttnn.transformer.windowed_scaled_dot_product_attention``) -- the SDPA variant
the cited reference (qwen25_vl vision, ``forward_prefill``) uses -- instead of a
materialized [1, nh, seq, seq] score matrix. This is the same softmax(QK^T/√d)V
math the eager reference computes, but the flash-2 kernel never allocates the
O(seq^2) score buffer or a dense mask: it takes ``cu_seqlens`` and builds the
block-diagonal attention pattern internally, streaming scores in O(seq) memory.

The windowed variant (not the plain ``scaled_dot_product_attention``) is required
here for two reasons: (1) the real vision sequence is NOT tile-aligned -- a
document page grid like [1, 92, 122] gives seq = 11224, which TILE_LAYOUT pads up
to 11232; the cu_seqlens window [0, 11224] makes the kernel mask those padding
rows out, whereas plain SDPA with no mask lets the garbage tail rows pollute every
query's softmax (verified: PCC collapses to ~0.95 at unaligned seq). (2) Plain
SDPA only accepts a full [1, 1, seq, seq] additive mask (a [1,1,1,seq] key-only
broadcast is rejected: ``mask_shape[2] == q_shape[2]``), which at seq 11224 is a
~250 MB dense buffer that OOMs the single p150 -- exactly the failure this rewrite
removes. windowed SDPA needs no dense mask at all. For a single image
cu_seqlens=[0, seq] it is full bidirectional over the valid rows; for multiple
image blocks it reproduces the reference's exact block-diagonal cross-block
masking. HF dots.ocr vision likewise uses flash attention, so this is faithful.

Vision RoPE:
    freqs [seq, head_dim//2] -> cos/sin -> repeat last dim x2 -> [seq, head_dim]
    out = x * cos + rotate_half(x) * sin
    rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])

The cos/sin tables are derived from the static inputs (rotary_pos_emb freqs) and
are precomputed on host at construction time, then uploaded to device --
analogous to loading the RMSNorm gamma weight. ``cu_seqlens`` is uploaded as a
small uint32 ROW_MAJOR tensor (window boundaries). The forward() runs entirely
with ttnn ops (no host matmul / softmax / numpy / torch.nn.functional).

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
        # cos/sin broadcast over the [1, nh, seq, hd] q/k layout that
        # nlp_create_qkv_heads produces (heads at dim 1) -> [1, 1, seq, hd].
        cos, sin = _build_rope_tables(rotary_pos_emb, head_dim)
        cos = cos.reshape(1, 1, seq_length, head_dim)
        sin = sin.reshape(1, 1, seq_length, head_dim)
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

        # Block-diagonal attention boundaries for the windowed flash-SDPA kernel.
        # cu_seqlens is the cumulative window-seqlens vector (e.g. [0, 96, 256]
        # for two blocks, or [0, seq] for a single image -> full bidirectional).
        # Uploaded as a small uint32 ROW_MAJOR tensor; the SDPA kernel builds the
        # block-diagonal mask internally (no O(seq^2) dense mask materialized) and
        # uses the [0, seq] window to mask the TILE_LAYOUT padding tail when seq
        # is not tile-aligned (the real document grids are not).
        cu = cu_seqlens.to(torch.int64).tolist()
        if cu[0] != 0:
            cu = [0] + cu
        if cu[-1] != seq_length:
            cu = cu + [seq_length]
        self._cu_seqlens_list = cu
        self.cu_seqlens_tt = ttnn.from_torch(
            torch.tensor(cu, dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

        # Flash-SDPA chunk sizes: cap the chunk to a tile-aligned size that fits
        # the sequence (small grids use a smaller chunk; large pages use 128).
        def _chunk(seq):
            c = 128
            while c > 32 and c >= seq:
                c //= 2
            return max(c, 32)

        qk_chunk = _chunk(seq_length)
        grid = device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False,
            q_chunk_size=qk_chunk,
            k_chunk_size=qk_chunk,
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

        # At small grids the head-split / head-merge reshapes are pinned to L1
        # (the dominant op at validation grids ~256). But at the model's real
        # document resolution (seq in the thousands) an L1-interleaved
        # [seq, 3, nh, hd] buffer is hundreds of MB and overflows L1 -- so above
        # a tile-friendly threshold these intermediates live in DRAM. Flash SDPA
        # already removed the O(seq^2) score buffer; this keeps the surrounding
        # reshapes bounded too, letting full-resolution pages run.
        reshape_mem = ttnn.L1_MEMORY_CONFIG if seq <= 1024 else ttnn.DRAM_MEMORY_CONFIG

        # Fused QKV projection: [seq, dim] @ [dim, 3*dim] -> [seq, 3*dim].
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=reshape_mem,
        )

        # Fused QKV-head split: nlp_create_qkv_heads does reshape + slice +
        # transpose in ONE op, returning q/k/v each as [1, nh, seq, hd] -- the
        # layout flash SDPA wants. This replaces a manual
        # reshape->slice->reshape->permute chain that round-tripped ~38% of
        # vision device-time through DRAM at seq 11224 (tracy: reshape 23.7% +
        # slice 6.9% + transpose 7.2%). Mirrors qwen25_vl vision. head_dim 128 is
        # tile-aligned so the sub-tile-head workaround does not apply. The fused
        # QKV weight lays out q|k|v as contiguous nh*hd blocks (== the reference's
        # reshape(seq,3,nh,hd) ordering), which is what the op expects.
        qkv = ttnn.reshape(qkv, (1, 1, seq, 3 * nh * hd))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=nh,
            num_kv_heads=nh,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # each [1, nh, seq, hd]
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # Memory-efficient flash attention with block-diagonal windowing over
        # cu_seqlens. Same softmax(q k^T * scale + block_diag_mask) @ v math as
        # the reference, but the kernel never materializes the [1, nh, seq, seq]
        # score matrix or a dense mask -- it builds the block-diagonal pattern
        # internally (and masks the TILE_LAYOUT padding tail via the [0, seq]
        # window) and streams in O(seq) memory (flash-2 chunked). This is the SDPA
        # variant qwen25_vl vision uses; plain scaled_dot_product_attention is not
        # usable here (no mask -> garbage tail rows pollute softmax at unaligned
        # seq; dense [1,1,seq,seq] mask -> ~250 MB OOM at seq 11224).
        attn = ttnn.transformer.windowed_scaled_dot_product_attention(
            q,
            k,
            v,
            self.cu_seqlens_tt,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config,
            program_config=self.sdpa_program_config,
        )  # [1, nh, seq, hd]

        # Fused head-merge: nlp_concat_heads ([1, nh, seq, hd] -> [1, 1, seq,
        # nh*hd]) in one op instead of permute + reshape.
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, (seq, nh * hd), memory_config=reshape_mem)

        # Output projection: [seq, dim] @ [dim, dim] -> [seq, dim].
        out = ttnn.linear(
            attn,
            self.proj_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=reshape_mem,
        )
        return out

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MoonViT attention block.

Per `MoonVitEncoderLayer.attention_qkvpacked` in modeling_kimi_k25.py:
  xqkv = wqkv(x)                         # [..., L, 3*hidden]
  xq, xk, xv = unbind(xqkv.view(..., 3, num_heads, head_dim), dim=-3)
  xq, xk     = apply_rope(xq, xk, freqs_cis)
  attn       = sdpa(q, k, v, cu_seqlens=cu_seqlens)
  out        = wo(attn.flatten(-2))

Both `wqkv` and `wo` are bias-less (attn_bias=False in __init__). Scale is
1/sqrt(head_dim). Attention is non-causal and block-diagonal — each image
attends only within itself, using cu_seqlens to mark sequence boundaries.

Implementation notes:
  * `head_dim = 72` is not tile-aligned. ttnn.experimental.rotary_embedding_llama
    requires a tile-multiple head_dim, so we pad to 96 across wqkv output,
    Q/K/V, and wo input. Padded slots hold zeros, which:
      - leave Q/K/V untouched at pad positions (zero contribution to QK^T),
      - rotate identically with cos=1, sin=0 in the padded posemb slots,
      - multiply zero-rows of wo (zero contribution to output).
    Scale stays 1/sqrt(72) — the real head_dim — because padded dims add 0
    to the dot product.
  * Our `Rope2DSetup.get_cos_sin` already produces tensors in the
    "consecutive-pair" / ttnn rotary_embedding_llama layout (each pair of
    head_dim slots shares one cos/sin), so no convert_rope_style_*_to_meta
    is needed (step 6 verified this).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import get_rot_transformation_mat


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


def _next_tile_multiple(x: int, tile: int = ttnn.TILE_SIZE) -> int:
    return ((x + tile - 1) // tile) * tile


def _pad_wqkv_weight(
    weight: torch.Tensor,  # HF nn.Linear weight: [3*hidden, hidden]
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pad the wqkv weight's head_dim from `head_dim` to `padded_head_dim`.

    Returns a tensor in ttnn-linear convention: [hidden_in, 3*num_heads*padded_head_dim].
    """
    out_dim, in_dim = weight.shape
    assert out_dim == 3 * num_heads * head_dim, f"wqkv weight out dim {out_dim} != 3 * {num_heads} * {head_dim}"

    # HF layout interprets the output dim as (3, num_heads, head_dim).
    # Reshape, pad head_dim, then flatten back.
    w = weight.detach().to(torch.bfloat16)
    w = w.view(3, num_heads, head_dim, in_dim)  # (3, H, 72, in)
    w = F.pad(w, (0, 0, 0, padded_head_dim - head_dim))  # pad head_dim dim: (3, H, 96, in)
    w = w.reshape(3 * num_heads * padded_head_dim, in_dim)  # (3*H*96, in)
    # ttnn.linear takes weight as [in, out].
    return w.transpose(0, 1).contiguous()


def _pad_wqkv_bias(
    bias: torch.Tensor,  # HF nn.Linear bias: [3*hidden]
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pad the wqkv bias to match the head_dim-padded weight."""
    assert bias.shape == (
        3 * num_heads * head_dim,
    ), f"wqkv bias shape {tuple(bias.shape)} != ({3*num_heads*head_dim},)"
    b = bias.detach().to(torch.bfloat16).view(3, num_heads, head_dim)
    b = F.pad(b, (0, padded_head_dim - head_dim))  # pad last dim
    return b.reshape(3 * num_heads * padded_head_dim).contiguous()


def _pad_wo_weight(
    weight: torch.Tensor,  # HF nn.Linear weight: [hidden, hidden]
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pad the wo weight's input dim from num_heads*head_dim to num_heads*padded_head_dim.

    Returns ttnn-linear convention: [num_heads*padded_head_dim, hidden_out].
    """
    out_dim, in_dim = weight.shape
    assert in_dim == num_heads * head_dim, f"wo weight in dim {in_dim} != {num_heads} * {head_dim}"
    w = weight.detach().to(torch.bfloat16)
    # ttnn convention: [in, out]. Start from HF [out, in] -> transpose -> [in, out].
    w = w.transpose(0, 1).contiguous()  # (in=H*72, out)
    w = w.view(num_heads, head_dim, out_dim)  # (H, 72, out)
    w = F.pad(w, (0, 0, 0, padded_head_dim - head_dim))  # pad head_dim dim: (H, 96, out)
    w = w.reshape(num_heads * padded_head_dim, out_dim)  # (H*96, out)
    return w.contiguous()


class MoonVisionAttention(LightweightModule):
    """QKV-packed attention with 2D RoPE and cu_seqlens windowed SDPA."""

    def __init__(
        self,
        mesh_device,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        wqkv_weight: torch.Tensor,
        wo_weight: torch.Tensor,
        wqkv_bias: Optional[torch.Tensor] = None,
        wo_bias: Optional[torch.Tensor] = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.padded_head_dim = _next_tile_multiple(self.head_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert (
            num_heads * head_dim == hidden_size
        ), f"num_heads * head_dim ({num_heads}*{head_dim}={num_heads*head_dim}) != hidden_size {hidden_size}"

        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _is_mesh_device(mesh_device) else None

        wqkv_padded = _pad_wqkv_weight(wqkv_weight, num_heads, head_dim, self.padded_head_dim)
        self.wqkv = ttnn.as_tensor(
            wqkv_padded,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        if wqkv_bias is not None:
            wqkv_bias_padded = _pad_wqkv_bias(wqkv_bias, num_heads, head_dim, self.padded_head_dim)
            wqkv_bias_4d = wqkv_bias_padded.view(1, 1, 1, -1).contiguous()
            self.wqkv_bias = ttnn.as_tensor(
                wqkv_bias_4d,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        else:
            self.wqkv_bias = None

        wo_padded = _pad_wo_weight(wo_weight, num_heads, head_dim, self.padded_head_dim)
        self.wo = ttnn.as_tensor(
            wo_padded,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        if wo_bias is not None:
            assert wo_bias.shape == (hidden_size,), f"wo bias shape {tuple(wo_bias.shape)} != ({hidden_size},)"
            wo_bias_4d = wo_bias.detach().to(torch.bfloat16).view(1, 1, 1, -1).contiguous()
            self.wo_bias = ttnn.as_tensor(
                wo_bias_4d,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        else:
            self.wo_bias = None

        # Transformation matrix for ttnn rotary_embedding_llama, sized to padded head_dim.
        trans_mat_pt = get_rot_transformation_mat(self.padded_head_dim)
        self.transformation_mat = ttnn.as_tensor(
            trans_mat_pt,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_layer: torch.nn.Module,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dtype=ttnn.bfloat16,
    ) -> "MoonVisionAttention":
        """Construct from one HF MoonVitEncoderLayer reference (uses .wqkv and .wo).

        The HF MoonVitEncoderLayer constructor defaults `attn_bias=False`, but
        the actual Kimi-VL checkpoint sets it True — biases are pulled when present.
        """
        assert hasattr(ref_layer, "wqkv") and hasattr(
            ref_layer, "wo"
        ), f"expected MoonVitEncoderLayer-like module with .wqkv and .wo, got {type(ref_layer).__name__}"
        return cls(
            mesh_device=mesh_device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            wqkv_weight=ref_layer.wqkv.weight.data,
            wo_weight=ref_layer.wo.weight.data,
            wqkv_bias=ref_layer.wqkv.bias.data if ref_layer.wqkv.bias is not None else None,
            wo_bias=ref_layer.wo.bias.data if ref_layer.wo.bias is not None else None,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # cos/sin staging — caller hands us host tensors, we pad and push to device.

    def stage_cos_sin(self, cos_pt: torch.Tensor, sin_pt: torch.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Pad host cos/sin from head_dim=72 to padded_head_dim=96 and push to device.

        Padding strategy: cos=1, sin=0 in the extra slots. That's the identity
        rotation, which is what the padded Q/K head_dim slots see (they hold zeros).
        """
        assert (
            cos_pt.shape[-1] == self.head_dim and sin_pt.shape[-1] == self.head_dim
        ), f"cos/sin head_dim {cos_pt.shape[-1]} != self.head_dim {self.head_dim}"
        # Pad on the last dim with cos=1, sin=0.
        cos_padded = F.pad(cos_pt, (0, self.padded_head_dim - self.head_dim), value=1.0)
        sin_padded = F.pad(sin_pt, (0, self.padded_head_dim - self.head_dim), value=0.0)
        # Shape to (1, 1, L, padded_head_dim) for ttnn.
        cos_4d = cos_padded.view(1, 1, *cos_padded.shape).to(torch.bfloat16).contiguous()
        sin_4d = sin_padded.view(1, 1, *sin_padded.shape).to(torch.bfloat16).contiguous()

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if _is_mesh_device(self.device) else None
        cos_tt = ttnn.from_torch(
            cos_4d,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        sin_tt = ttnn.from_torch(
            sin_4d,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return cos_tt, sin_tt

    def stage_cu_seqlens(self, cu_seqlens_pt: torch.Tensor) -> ttnn.Tensor:
        """Push cu_seqlens to device as uint32 row-major (the layout ttnn windowed SDPA expects)."""
        cu = cu_seqlens_pt.to(torch.int32).contiguous()
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if _is_mesh_device(self.device) else None
        return ttnn.from_torch(
            cu,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

    # ------------------------------------------------------------------
    # Forward

    def forward(
        self,
        x: ttnn.Tensor,  # (1, 1, L, hidden_size)
        cu_seqlens: ttnn.Tensor,  # uint32 row-major, length (num_seqs+1)
        cos: ttnn.Tensor,  # (1, 1, L, padded_head_dim) — already padded
        sin: ttnn.Tensor,  # (1, 1, L, padded_head_dim)
        memory_config: Optional["ttnn.MemoryConfig"] = None,
    ) -> ttnn.Tensor:
        # QKV projection (bias when checkpoint has it).
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            bias=self.wqkv_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Split into Q/K/V heads.
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # RoPE on Q and K.
        q_heads = ttnn.experimental.rotary_embedding_llama(
            q_heads, cos, sin, self.transformation_mat, is_decode_mode=False
        )
        k_heads = ttnn.experimental.rotary_embedding_llama(
            k_heads, cos, sin, self.transformation_mat, is_decode_mode=False
        )

        # Variable-length attention with cu_seqlens.
        attn = ttnn.transformer.windowed_scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            cu_seqlens,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # Concat heads back into (1, 1, L, num_heads * padded_head_dim).
        attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        # Output projection.
        out = ttnn.linear(
            attn_flat,
            self.wo,
            bias=self.wo_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_flat)
        return out

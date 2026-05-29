# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of one dots.ocr DotsVisionTransformer block.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`vision_block_forward`

DotsVisionBlock (pre-norm, residual):

    h = h + attn(norm1(h))
    h = h + mlp(norm2(h))

This is the FIRST composite block. It does NOT re-implement the leaf maths --
it imports and composes the already-verified leaf modules:

    TtVisionRMSNorm  (x2, eps 1e-5)   -- tt/vision_rmsnorm.py
    TtVisionAttention (fused QKV, 2D vision RoPE, block-diagonal mask) -- tt/vision_attention.py
    TtVisionMLP       (SwiGLU FFN, no bias) -- tt/vision_mlp.py

The cu_seqlens block-diagonal mask + 2D RoPE cos/sin tables are threaded into
TtVisionAttention exactly as the standalone attention block did (precomputed on
host at construction time, uploaded to device like the norm gamma weight). The
forward() runs entirely with ttnn ops (no host-side matmul / softmax / activation).

embed_dim 1536, num_heads 12, head_dim 128, intermediate 4224, rms_norm_eps 1e-5,
use_bias False.

The model dir name (rednote_hilab_dots.ocr) contains a dot, so the sibling leaf
modules cannot be imported via the normal dotted package path -- they are loaded
by file path with importlib (the same convention the tests use).

Reference TTNN impl this follows: models/demos/qwen25_vl/tt/vision_block.py
"""
import importlib.util
import os

import ttnn
from models.common.lightweightmodule import LightweightModule

_TT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_leaf(module_name: str, file_name: str, symbol: str):
    """Import a sibling leaf module by file path (dir name has a dot)."""
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_TT_DIR, file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, symbol)


TtVisionRMSNorm = _load_leaf("dots_tt_vision_rmsnorm", "vision_rmsnorm.py", "TtVisionRMSNorm")
TtVisionAttention = _load_leaf("dots_tt_vision_attention", "vision_attention.py", "TtVisionAttention")
TtVisionMLP = _load_leaf("dots_tt_vision_mlp", "vision_mlp.py", "TtVisionMLP")


class TtVisionBlock(LightweightModule):
    """dots.ocr vision transformer block (pre-norm residual).

    Composes the three verified leaf modules. Args mirror the block golden's
    flat state_dict (norm1/attn.qkv/attn.proj/norm2/mlp.fc1/fc2/fc3 weights).

    Args:
        device: ttnn Device or MeshDevice.
        norm1_weight: torch.Tensor [dim] -- pre-attention RMSNorm gamma.
        qkv_weight: torch.Tensor [3*dim, dim] -- fused QKV proj (no bias).
        proj_weight: torch.Tensor [dim, dim] -- attention output proj (no bias).
        norm2_weight: torch.Tensor [dim] -- pre-MLP RMSNorm gamma.
        fc1_weight: torch.Tensor [intermediate, dim] -- gate proj (no bias).
        fc3_weight: torch.Tensor [intermediate, dim] -- up proj (no bias).
        fc2_weight: torch.Tensor [dim, intermediate] -- down proj (no bias).
        rotary_pos_emb: torch.Tensor [seq, head_dim//2] vision rope freqs.
        cu_seqlens: int tensor of cumulative seqlens (block-diagonal mask).
        seq_length: number of patch tokens.
        num_heads: 12.
        head_dim: 128.
        eps: RMSNorm epsilon (1e-5).
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        norm1_weight,
        qkv_weight,
        proj_weight,
        norm2_weight,
        fc1_weight,
        fc3_weight,
        fc2_weight,
        rotary_pos_emb,
        cu_seqlens,
        seq_length,
        num_heads: int = 12,
        head_dim: int = 128,
        eps: float = 1e-5,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        dim = num_heads * head_dim

        self.norm1 = TtVisionRMSNorm(
            device=device,
            dim=dim,
            weight=norm1_weight,
            eps=eps,
            weight_dtype=dtype,
            weight_memory_config=weight_memory_config,
        )
        self.attn = TtVisionAttention(
            device=device,
            qkv_weight=qkv_weight,
            proj_weight=proj_weight,
            rotary_pos_emb=rotary_pos_emb,
            cu_seqlens=cu_seqlens,
            seq_length=seq_length,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            weight_memory_config=weight_memory_config,
        )
        self.norm2 = TtVisionRMSNorm(
            device=device,
            dim=dim,
            weight=norm2_weight,
            eps=eps,
            weight_dtype=dtype,
            weight_memory_config=weight_memory_config,
        )
        self.mlp = TtVisionMLP(
            device=device,
            fc1_weight=fc1_weight,
            fc3_weight=fc3_weight,
            fc2_weight=fc2_weight,
            dtype=dtype,
            weight_memory_config=weight_memory_config,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, dim] (TILE layout) -> [seq, dim].

        h = h + attn(norm1(h)); h = h + mlp(norm2(h)).

        Both residual adds are pinned to L1. Tracy on the traced composite path
        showed these two ttnn.add ops are the only block-internal ops that land
        DRAM-interleaved (~27 us each at the default) -- every leaf sub-op was
        already L1-pinned during the per-leaf optimization passes. Inheriting L1
        here keeps the whole block resident in L1 between the attn/mlp output and
        the residual.
        """
        # L1-resident residuals at validation grids; DRAM at the real document
        # resolution where the [seq, dim] residual overflows L1.
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[0] <= 1024 else ttnn.DRAM_MEMORY_CONFIG

        attn_out = self.attn(self.norm1(x))
        x = ttnn.add(x, attn_out, memory_config=mem)

        mlp_out = self.mlp(self.norm2(x))
        x = ttnn.add(x, mlp_out, memory_config=mem)
        return x

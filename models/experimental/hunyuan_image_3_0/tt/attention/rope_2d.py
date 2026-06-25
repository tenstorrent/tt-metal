# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of Hunyuan 2D RoPE.
#
# Design
# ------
# Hunyuan's 2D RoPE is unique: cos/sin tables are built on the CPU by
# build_batch_2d_rope() (our ref implementation) and vary per forward pass
# because image patch coordinates depend on the input image size.
# We therefore:
#   1. Build cos/sin on CPU (ref build_batch_2d_rope — unchanged).
#   2. Reshape to [1, 1, seq_len, head_dim] and upload to device with
#      ttnn.from_torch() each prefill.
#   3. Apply split-half RoPE on device via apply_rotary_pos_emb_vision_tt from
#      tt_transformers (same rotate_half math as Hunyuan ref).
#
# References
# ----------
#   models/tt_transformers/tt/multimodal/mistral_24b/vision_attention.py
#   models/experimental/hunyuan_image_3_0/ref/attention/rope_2d.py

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.tt_transformers.tt.multimodal.mistral_24b.vision_attention import (
    apply_rotary_pos_emb_vision_tt,
)


class HunyuanTtRoPE2D(LightweightModule):
    """
    Hunyuan 2D RoPE — prefill path, single device.

    Usage per forward pass
    ----------------------
    1.  Call prepare_cos_sin() once on the CPU to build the cos/sin tables
        for the current sequence (depends on image size / token layout).
    2.  Call forward(q, k, cos_tt, sin_tt) to apply RoPE on device.

    Args:
        device:    TTNN device.
        head_dim:  Per-head dimension (default 128).

    Note on rotate_half implementation
    -----------------------------------
    Hunyuan uses split-half RoPE: rotate_half(x) = cat([-x[...,D/2:], x[...,:D/2]], dim=-1).
    ttnn.experimental.rotary_embedding_llama uses an interleaved-pair rotation matrix
    (32×32 tile), which is not equivalent for head_dim=128. We reuse the split-half
    apply path from tt_transformers Mistral vision attention instead.
    """

    def __init__(self, device, head_dim: int = 128):
        super().__init__()
        self.device = device
        self.head_dim = head_dim

    def prepare_cos_sin(
        self,
        seq_len: int,
        image_infos=None,
        base: int = 10000,
    ):
        """
        Build 2D RoPE cos/sin tables on CPU and upload to device.

        Args:
            seq_len:     Sequence length.
            image_infos: List[List[Tuple[slice, (h, w)]]] or None for text-only.
                         Outer list is batch; each inner list has one entry per
                         image region in that sample.
            base:        RoPE theta base (default 10000).

        Returns:
            cos_tt: ttnn.Tensor [1, 1, seq_len, head_dim] on device, TILE_LAYOUT.
            sin_tt: ttnn.Tensor [1, 1, seq_len, head_dim] on device, TILE_LAYOUT.
        """
        cos, sin = build_batch_2d_rope(
            seq_len=seq_len,
            n_elem=self.head_dim,
            image_infos=image_infos,
            base=base,
        )
        cos = cos[0].unsqueeze(0).unsqueeze(0)
        sin = sin[0].unsqueeze(0).unsqueeze(0)

        cos_tt = ttnn.from_torch(
            cos,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin_tt = ttnn.from_torch(
            sin,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cos_tt, sin_tt

    def slice_cos_sin(self, cos_tt: ttnn.Tensor, sin_tt: ttnn.Tensor, position: int):
        """Return cos/sin for a single sequence position (decode step)."""
        return (
            ttnn.slice(cos_tt, [0, 0, position, 0], [1, 1, position + 1, self.head_dim]),
            ttnn.slice(sin_tt, [0, 0, position, 0], [1, 1, position + 1, self.head_dim]),
        )

    @staticmethod
    def _cos_sin_for_vision_apply(cos_tt: ttnn.Tensor, sin_tt: ttnn.Tensor):
        """Adapt [1, 1, S, D] tables to the [1, S, D] shape expected by vision RoPE."""
        if len(cos_tt.shape) == 4 and cos_tt.shape[0] == 1 and cos_tt.shape[1] == 1:
            cos_tt = ttnn.squeeze(cos_tt, 1)
            sin_tt = ttnn.squeeze(sin_tt, 1)
        return cos_tt, sin_tt

    def forward(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        cos_tt: ttnn.Tensor,
        sin_tt: ttnn.Tensor,
    ):
        """
        Apply 2D RoPE to Q and K.

        Args:
            q:      [B, num_heads,    S, head_dim] TILE_LAYOUT on device.
            k:      [B, num_kv_heads, S, head_dim] TILE_LAYOUT on device.
            cos_tt: [1, 1, S, head_dim] from prepare_cos_sin().
            sin_tt: [1, 1, S, head_dim] from prepare_cos_sin().

        Returns:
            q_rot, k_rot — same shapes as inputs.
        """
        cos_tt, sin_tt = self._cos_sin_for_vision_apply(cos_tt, sin_tt)
        return apply_rotary_pos_emb_vision_tt(q, k, cos_tt, sin_tt)

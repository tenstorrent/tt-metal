# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN transformer: the MiniMax-H3 DiT with every block's attention swapped for `OpenVDNHybridAttention`, as the
model repo's `apply_hybrid_attention_transform` does (all 50 blocks; the two token-refiner blocks stay dense).

Reference: OpenVDN/vdn-minimax-h3 @e262cb57, `VDNMiniMaxH3Transformer3DModel` (`stage-*/diffusers/modeling_vdn_h3.py`):
same forward and outputs as the base class, plus the packed layout, which the caller sets with `set_layout` before
each request.
"""

from __future__ import annotations

from ..minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from .hybrid_attention_openvdn import HybridLayout, OpenVDNHybridAttention


class OpenVDNTransformer3DModel(MiniMaxH3Transformer3DModel):
    def __init__(
        self,
        *,
        radius: int = 1,
        chunk: int = 5,
        anchor_frames: str = "both",
        enable_text_state: bool = True,
        is_fsdp: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(is_fsdp=is_fsdp, **kwargs)
        for block in self.transformer_blocks:
            dense = block.attn
            block.attn = OpenVDNHybridAttention(
                radius=radius,
                chunk=chunk,
                anchor_frames=anchor_frames,
                enable_text_state=enable_text_state,
                hidden_size=dense.hidden_size,
                num_heads=dense.num_heads,
                head_dim=dense.head_dim,
                rotary_dim=dense.rotary_dim,
                qk_norm_eps=dense.qk_norm_eps,
                mesh_device=dense.mesh_device,
                ccl_manager=dense.ccl_manager,
                parallel_config=dense.parallel_config,
                is_fsdp=is_fsdp,
                kv_gather_capacity=dense.kv_gather_capacity,
                use_persistent_ccl_buffers=dense.use_persistent_ccl_buffers,
            )

    def set_layout(self, layout: HybridLayout | None) -> None:
        for block in self.transformer_blocks:
            block.attn.layout = layout

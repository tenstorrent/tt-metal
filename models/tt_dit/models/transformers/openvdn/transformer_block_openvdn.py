# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN transformer block: the MiniMax-H3 block (AdaLN-modulated attention and SwiGLU feed-forward) with its
attention swapped for `OpenVDNHybridAttention`, as the model repo's `hybrid_transform` does to each diffusers block.
Everything else is the shared H3 block, unchanged. Set `block.attn.layout` before each forward."""

from __future__ import annotations

from ..minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from .hybrid_attention_openvdn import OpenVDNHybridAttention

# The block keywords the H3 block forwards to its attention.
_ATTENTION_KWARGS = (
    "hidden_size",
    "num_heads",
    "head_dim",
    "rotary_dim",
    "qk_norm_eps",
    "mesh_device",
    "ccl_manager",
    "parallel_config",
    "is_fsdp",
    "kv_gather_capacity",
    "use_persistent_ccl_buffers",
    "dynamic_length",
    "norm_row_capacity",
    "agmm_mode",
)


class OpenVDNTransformerBlock(MiniMaxH3TransformerBlock):
    def __init__(
        self,
        *,
        radius: int = 1,
        chunk: int = 5,
        anchor_frames: str = "both",
        enable_text_state: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.attn = OpenVDNHybridAttention(
            radius=radius,
            chunk=chunk,
            anchor_frames=anchor_frames,
            enable_text_state=enable_text_state,
            **{k: kwargs[k] for k in _ATTENTION_KWARGS if k in kwargs},
        )

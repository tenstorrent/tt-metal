# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Decoder layer for Devstral-2 / Ministral3.

HF reference (``Ministral3DecoderLayer``)::

    h = self_attn(input_layernorm(x), ...) + x
    out = mlp(post_attention_layernorm(h)) + h
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn

from models.experimental.devstral2_large.tt.model_args import Devstral2Args
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import TtRotaryEmbedding
from models.experimental.devstral2_large.tt.tt_ministralattn import TtAttention
from models.experimental.devstral2_large.tt.tt_ministralmlp import TtMLP
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import TtRMSNorm

__all__ = ["TtDecoderLayer"]


class TtDecoderLayer:
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
        weight_cache_path: Optional[str] = None,
    ) -> None:
        self.args = args
        self.layer_idx = layer_idx
        prefix = args.state_dict_prefix("", layer_idx)
        self.input_layernorm = TtRMSNorm(args, mesh_device, state_dict, prefix + "input_layernorm.weight", dtype=dtype)
        self.post_attention_layernorm = TtRMSNorm(
            args, mesh_device, state_dict, prefix + "post_attention_layernorm.weight", dtype=dtype
        )
        self.self_attn = TtAttention(
            args,
            mesh_device,
            state_dict,
            layer_idx,
            tt_ccl,
            rotary_emb,
            dtype=dtype,
            weight_cache_path=weight_cache_path,
        )
        self.mlp = TtMLP(
            args,
            mesh_device,
            state_dict,
            layer_idx,
            tt_ccl,
            dtype=dtype,
            weight_cache_path=weight_cache_path,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        *,
        mode: str = "decode",
        start_pos: int = 0,
        current_pos_host: Optional[torch.Tensor] = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        # Attention sub-block.
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(
            h,
            mode=mode,
            start_pos=start_pos,
            current_pos_host=current_pos_host,
            user_id=user_id,
        )
        h = ttnn.add(h, residual)
        ttnn.deallocate(residual)

        # MLP sub-block.
        residual = h
        h2 = self.post_attention_layernorm(h)
        h2 = self.mlp(h2)
        out = ttnn.add(h2, residual)
        ttnn.deallocate(residual)
        ttnn.deallocate(h2)
        return out

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)

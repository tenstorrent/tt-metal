# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decoder layer for Devstral-2 / Ministral3.

HF reference (``Ministral3DecoderLayer``)::

    h = self_attn(input_layernorm(x), ...) + x
    out = mlp(post_attention_layernorm(h)) + h
"""

from __future__ import annotations

from typing import Optional, Sequence

import ttnn

from models.experimental.devstral2_123B_instruct.tt.mem_config import (
    get_decode_width_sharded_activation_mem_config,
    get_prefill_width_sharded_activation_mem_config,
    use_width_sharded_decode_norm_matmul,
    use_width_sharded_prefill_norm_matmul,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args
from models.experimental.devstral2_123B_instruct.tt.tt_ministral_rotary_emb import TtRotaryEmbedding
from models.experimental.devstral2_123B_instruct.tt.tt_ministralattn import TtAttention
from models.experimental.devstral2_123B_instruct.tt.tt_ministralmlp import TtMLP
from models.experimental.devstral2_123B_instruct.tt.tt_ministralrmsnorm import TtRMSNorm

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
        self.input_layernorm = TtRMSNorm(
            args,
            mesh_device,
            state_dict,
            prefix + "input_layernorm.weight",
            dtype=dtype,
            weight_cache_path=weight_cache_path,
        )
        self.post_attention_layernorm = TtRMSNorm(
            args,
            mesh_device,
            state_dict,
            prefix + "post_attention_layernorm.weight",
            dtype=dtype,
            weight_cache_path=weight_cache_path,
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
        current_pos: Optional[ttnn.Tensor] = None,
        user_id: int = 0,
        chunk_start_idx_tensor: Optional[ttnn.Tensor] = None,
        chunk_page_table: Optional[ttnn.Tensor] = None,
        prefill_rope_tables: Optional[Sequence[ttnn.Tensor]] = None,
    ) -> ttnn.Tensor:
        mesh_device = self.self_attn.mesh_device
        act_mem = self.args.get_activation_mem_config(mode, mesh_device)
        seq_len = max(1, int(x.shape[-2]))
        ws_norm_out_mem = act_mem
        if use_width_sharded_prefill_norm_matmul(self.args, mode, seq_len):
            ws_norm_out_mem = get_prefill_width_sharded_activation_mem_config(seq_len, self.args.hidden_size)
        elif use_width_sharded_decode_norm_matmul(self.args, mode):
            ws_norm_out_mem = get_decode_width_sharded_activation_mem_config(self.args.hidden_size)
        input_norm_out_mem = ws_norm_out_mem

        # Attention sub-block.
        residual = x
        h = self.input_layernorm(x, memory_config=input_norm_out_mem, mode=mode)
        h = self.self_attn(
            h,
            mode=mode,
            start_pos=start_pos,
            current_pos=current_pos,
            user_id=user_id,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
            chunk_page_table=chunk_page_table,
            prefill_rope_tables=prefill_rope_tables,
        )
        h = ttnn.add(h, residual, memory_config=act_mem)
        ttnn.deallocate(residual)

        # MLP sub-block.
        residual = h
        h2 = self.post_attention_layernorm(h, memory_config=ws_norm_out_mem, mode=mode)
        h2 = self.mlp(h2, mode=mode)
        out = ttnn.add(h2, residual, memory_config=act_mem)
        ttnn.deallocate(residual)
        ttnn.deallocate(h2)
        return out

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)

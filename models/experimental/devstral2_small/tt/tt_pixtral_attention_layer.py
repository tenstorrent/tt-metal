# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstral2_small.devstral_utils.pixtral_seq_chunk import (
    vision_rms_norm_block_shard_eligible,
    vision_rms_norm_block_shard_memcfg,
    vision_rms_norm_prepare_block_shard_input,
    vision_seq_memcfg,
)
from models.experimental.devstral2_small.tt.tt_pixtralattn import TtMistralImageAttention
from models.experimental.devstral2_small.tt.tt_pixtralmlp import MistralTTVisionMLP
from models.experimental.devstral2_small.tt.tt_pixtralnorm import TtPixtralRMSNorm


class TtPixtralAttentionLayer(LightweightModule):
    """TT composition of HF ``PixtralAttentionLayer``: attention_norm -> attention + residual -> ffn_norm -> mlp + residual."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()
        self.attention_norm = TtPixtralRMSNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            eps=1e-5,
            weight_key=f"{state_dict_prefix}attention_norm.weight",
            dtype=dtype,
        )
        self.attention = TtMistralImageAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}attention.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )
        self.ffn_norm = TtPixtralRMSNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            eps=1e-5,
            weight_key=f"{state_dict_prefix}ffn_norm.weight",
            dtype=dtype,
        )
        self.mlp = MistralTTVisionMLP(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            state_dict_prefix=f"{state_dict_prefix}feed_forward.",
            tt_ccl=tt_ccl,
        )

    def forward(self, hidden_states: ttnn.Tensor, attention_mask=None, position_embeddings=None) -> ttnn.Tensor:
        seq_len = int(hidden_states.shape[-2])
        hidden = int(hidden_states.shape[-1])
        layer_mem = vision_seq_memcfg(seq_len, hidden)

        residual = hidden_states
        x = self.attention_norm(hidden_states)
        x = self.attention(x, position_embeddings=position_embeddings)

        if vision_rms_norm_block_shard_eligible(seq_len, hidden, 8, 8):
            # Sweep winner (8x8, block_h=4, block_w=4, subblock_w=4): run the post-attention
            # residual add BLOCK-SHARDED so its output feeds the block-sharded ffn_norm with
            # NO interleaved->sharded reshard between them (ffn_norm's internal
            # vision_rms_norm_prepare_block_shard_input becomes a no-op) and the BinaryNg itself
            # runs sharded. The residual is kept block-sharded through the MLP residual add
            # (so that add is sharded too and no mid-block reshard is introduced), then
            # stay block-sharded across transformer layers (no sharded_to_interleaved per layer).
            bs_mem = vision_rms_norm_block_shard_memcfg(seq_len, hidden, 8, 8)
            a_bs = vision_rms_norm_prepare_block_shard_input(residual, seq_len, hidden)
            x_bs = vision_rms_norm_prepare_block_shard_input(x, seq_len, hidden)
            if x_bs is not x:  # prepare returns the same tensor when already block-sharded
                ttnn.deallocate(x)
            residual = ttnn.add(a_bs, x_bs, memory_config=bs_mem)
            if a_bs is not hidden_states:
                ttnn.deallocate(hidden_states)
            ttnn.deallocate(a_bs)
            ttnn.deallocate(x_bs)

            x = self.ffn_norm(residual)  # block-sharded input -> prepare is a no-op
            x = self.mlp(x)
            x_bs = vision_rms_norm_prepare_block_shard_input(x, seq_len, hidden)
            if x_bs is not x:
                ttnn.deallocate(x)
            out = ttnn.add(residual, x_bs, memory_config=bs_mem)
            ttnn.deallocate(residual)
            ttnn.deallocate(x_bs)
            return out

        ttnn.add(residual, x, output_tensor=residual, memory_config=layer_mem)
        ttnn.deallocate(x)

        x = self.ffn_norm(residual)
        x = self.mlp(x)
        ttnn.add(residual, x, output_tensor=residual, memory_config=layer_mem)
        ttnn.deallocate(x)
        return residual


__all__ = ["TtPixtralAttentionLayer"]

"""
One encoder block of the Janus-Pro-7B vision tower: pre-norm attention and pre-norm MLP,
each into a residual add.

HF reference: `vision_model.encoder.layers[i]` (`ModelArgs.reference_vision_encoder_block`).
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.janus_pro.tt.janus_pro_image_attention import TtJanusProImageAttention
from models.experimental.janus_pro.tt.janus_pro_image_mlp import TtJanusProImageFeedForward
from models.experimental.janus_pro.tt.janus_pro_layernorm import TtJanusProLayerNorm


class TtJanusProImageTransformerBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        tt_ccl,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        residual_dtype=None,
    ):
        super().__init__()

        # `None` leaves the adds inheriting their input format. Both layer norms take their output
        # format from the residual, so this also sets what qkv and c_fc multicast as in0.
        self.residual_dtype = residual_dtype

        self.ln_1 = TtJanusProLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_1.",
            configuration=configuration,
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.attn = TtJanusProImageAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}attn.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )

        self.ln_2 = TtJanusProLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_2.",
            configuration=configuration,
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.mlp = TtJanusProImageFeedForward(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=configuration,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}mlp.",
            weight_cache_path=weight_cache_path,
        )

    def forward(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"
        batch_size = x_11SH.shape[0]

        # Both norms hand their shard straight to the projection that reads it: the 8x6 grid and the
        # block width in tiles are what qkv's and c_fc's 2D configs already want for in0, so no
        # unshard sits between them. They stay bfloat16 -- LoFi truncates to the same mantissa either
        # way, so narrowing first would only lose bits.
        attn_out = self.attn(self.ln_1(x_11SH, out_sharded=True), mask=mask)

        # Align x_11SH shape with attn_output
        x_11SH = ttnn.reshape(x_11SH, [batch_size, 1, seq_len, -1])

        res = ttnn.add(x_11SH, attn_out, dtype=self.residual_dtype)

        mlp_out = self.mlp(self.ln_2(res, out_sharded=True))
        out = ttnn.add(res, mlp_out, dtype=self.residual_dtype)

        ttnn.deallocate(mlp_out)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(res)
        return out

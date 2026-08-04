"""
This is the ImageTransformer block for Gemma-3-4b-it.
We have reused the TtLlamaImageTransformerBlock with incorporating the
TtGemmaImageAttention and TtGemmaImageFeedForward
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.multimodal.gemma3.tt.gemma_image_attention import TtGemmaImageAttention
from models.demos.multimodal.gemma3.tt.gemma_image_mlp import TtGemmaImageFeedForward
from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm


class TtGemmaImageTransformerBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        tt_ccl,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.vision_dim

        self.ln_1 = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_1.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.attn = TtGemmaImageAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}attn.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )

        self.ln_2 = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_2.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.mlp = TtGemmaImageFeedForward(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=configuration,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}mlp.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def forward(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"
        batch_size = x_11SH.shape[0]

        attn_out = self.attn(self.ln_1(x_11SH), mask=mask)

        # Align x_11SH shape with attn_out
        x_11SH = ttnn.reshape(x_11SH, [batch_size, 1, seq_len, -1])

        res = ttnn.add(x_11SH, attn_out)

        mlp_out = self.mlp(self.ln_2(res))
        out = ttnn.add(res, mlp_out)

        ttnn.deallocate(mlp_out)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(res)
        return out

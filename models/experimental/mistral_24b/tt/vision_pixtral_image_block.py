# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_24b.tt.rmsnorm import RMSNorm

from models.experimental.mistral_24b.tt.vision_attention import TtMistralImageAttention as TtLlamaImageAttention
from models.experimental.mistral_24b.tt.vision_mlp import MistralTTVisionMLP as MLP

try:
    from tracy import signpost
except ImportError:

    def signpost(*args, **kwargs):
        pass


"""
This file implements the pixtral image block specific for the Mistral-Small-3.1-24B-Instruct-2503 model.
"""


class TtPixtralImageTransformerBlock(LightweightModule):
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
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.configuration = configuration
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.vision_dim

        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_key="attention_norm",
            weight_dtype=dtype,
            is_distributed=False,
            simplified_rms=True,
        )

        self.attention = TtLlamaImageAttention(
            mesh_device,
            tt_ccl,
            state_dict,
            state_dict_prefix=f"{state_dict_prefix}attention.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )

        self.ffn_norm = RMSNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_key="ffn_norm",
            weight_dtype=dtype,
            is_distributed=False,
            simplified_rms=True,
        )

        self.mlp = MLP(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=f"{state_dict_prefix}feed_forward.",
            dtype=dtype,
        )

    def forward(self, x_input, position_embeddings=None):
        mode = "prefill"
        # attention norm Input and result replicated
        signpost("Mistral24B::VisionBlock::AttentionNorm::Start")
        attn_norm_res = self.attention_norm(x_input, mode=mode)
        signpost("Mistral24B::VisionBlock::AttentionNorm::End")
        # attention Input and results replicated
        signpost("Mistral24B::Attention::Start")
        attn_out = self.attention(attn_norm_res, position_embeddings=position_embeddings)
        res = ttnn.add(x_input, attn_out)
        ffn_norm_res = self.ffn_norm(res, mode=mode)
        signpost("Mistral24B::VisionBlock::FFNNorm::End")
        signpost("Mistral24B::VisionMLP::Start")
        mlp_out = self.mlp(ffn_norm_res)
        signpost("Mistral24B::VisionMLP::End")
        signpost("Mistral24B::VisionBlock::FFNResidual::Start")
        out = ttnn.add(res, mlp_out)
        signpost("Mistral24B::VisionBlock::FFNResidual::End")
        return out

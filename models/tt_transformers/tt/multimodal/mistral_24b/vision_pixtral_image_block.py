# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.multimodal.mistral_24b.rmsnorm import RMSNorm
from models.tt_transformers.tt.multimodal.mistral_24b.vision_attention import (
    TtMistralImageAttention as TtLlamaImageAttention,
)
from models.tt_transformers.tt.multimodal.mistral_24b.vision_mlp import MistralTTVisionMLP as MLP


class TtPixtralImageTransformerBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
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
            sharded_program_config=configuration.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
            sharded_output_config=configuration.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
        )

        self.attention = TtLlamaImageAttention(
            mesh_device,
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
            sharded_program_config=configuration.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
            sharded_output_config=configuration.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
        )

        self.mlp = MLP(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=f"{state_dict_prefix}feed_forward.",
            dtype=dtype,
        )

    def forward(self, x_input, mask=None, position_embeddings=None):
        mode = "prefill"
        attn_out = self.attention(
            self.attention_norm(x_input, mode=mode), position_embeddings=position_embeddings, mask=mask
        )
        res = ttnn.add(x_input, attn_out, use_legacy=True)
        mlp_out = self.mlp(self.ffn_norm(res, mode=mode))
        out = ttnn.add(res, mlp_out)
        return out

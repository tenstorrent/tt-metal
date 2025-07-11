# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm as RMSNorm

from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.experimental.mistral_24b.tt.vision_attention import TtMistralImageAttention as TtLlamaImageAttention
from models.experimental.mistral_24b.tt.vision_mlp import MistralTTVisionMLP as MLP


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

        inner_rms = RMSNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_key="attention_norm",
            weight_dtype=dtype,
            is_distributed=configuration.is_distributed_norm,
        )
        self.attention_norm = DistributedNorm(inner_rms, configuration, TG=configuration.is_galaxy)

        self.attention = TtLlamaImageAttention(
            mesh_device,
            state_dict,
            state_dict_prefix=f"{state_dict_prefix}attention.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )

        ffn_rms = RMSNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_key="ffn_norm",
            weight_dtype=dtype,
            is_distributed=configuration.is_distributed_norm,
        )

        self.ffn_norm = DistributedNorm(ffn_rms, configuration, TG=configuration.is_galaxy)

        self.mlp = MLP(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            weight_cache_path=configuration.weight_cache_path(dtype),
            state_dict_prefix=f"{state_dict_prefix}feed_forward.",
            dtype=dtype,
        )

    def forward(self, x_input, mask=None, position_embeddings=None):
        mode = "prefill"
        attn_out = self.attention(
            self.attention_norm(x_input, mode=mode), position_embeddings=position_embeddings, mask=mask
        )
        res = ttnn.add(x_input, attn_out)
        mlp_out = self.mlp(self.ffn_norm(res, mode=mode))
        out = ttnn.add(res, mlp_out)
        return out

"""
This is the vision block used in the Qwen-VL-7B architecture
consisting of RMSnorm and self-attention layer followed by an MLP layer.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.qwen25_vl.tt.mlp import QwenTTVisionMLP
from models.experimental.qwen25_vl.tt.rmsnorm import RMSNorm
from models.experimental.qwen25_vl.tt.attention import TtQwen2_5_VLVisionSdpaAttention


class TtQwen2_5_VLVisionBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        dtype,
        model_args,
        weight_cache_path=None,
        state_dict_prefix=None,
    ):
        super().__init__()

        self.norm1 = RMSNorm(
            device=mesh_device,
            dim=1280,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_key="norm1",
            weight_dtype=dtype,
            is_distributed=False,
            sharded_program_config=model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
            sharded_output_config=model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
        )

        self.norm2 = RMSNorm(
            device=mesh_device,
            dim=1280,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_key="norm2",
            weight_dtype=dtype,
            is_distributed=False,
            sharded_program_config=model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
            sharded_output_config=model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
        )

        self.attn = TtQwen2_5_VLVisionSdpaAttention(
            mesh_device,
            state_dict,
            state_dict_prefix=f"{state_dict_prefix}attn.",
            # weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
            configuration=model_args,
        )

        self.mlp = QwenTTVisionMLP(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}feed_forward.",
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
        )

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = ttnn.add(
            hidden_states,
            self.attn(
                self.norm1(hidden_states),
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            ),
        )

        hidden_states = ttnn.add(hidden_states, self.mlp(self.norm2(hidden_states)))

        return hidden_states

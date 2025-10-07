import math

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.grok.tt.attention import Attention
from models.demos.grok.tt.expert_mlp import ExpertMLP
from models.demos.grok.tt.mlp import MLP
from models.demos.grok.tt.moe import TtMoE


class Decoder(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        args,
        layer_num,
        dtype,
        transformation_mats,
        paged_attention_config=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype
        self.model_config = args.get_model_config()

        self.attention = Attention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
        )
        self.shared_mlp = MLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )

        self.experts = ExpertMLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            args=args,
            layer_num=layer_num,
            dtypes={
                "w1": ttnn.bfloat8_b,
                "w2": ttnn.bfloat8_b,
                "w3": ttnn.bfloat8_b,
            },
        )
        self.moe = TtMoE(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            experts=self.experts,
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )

        # We have 4 norms: pre-attn, post-attn, pre-moe, post-moe

    def forward(self, hidden_states, current_pos, rot_mats, page_table=None):
        # Pre-attn norm
        residual = hidden_states
        # hidden_states = self.pre_attn_norm(hidden_states)

        # Attention
        attn_out = self.attention.forward(hidden_states, current_pos, rot_mats, page_table=page_table)

        # Post-attn norm
        # attn_out = self.post_attn_norm(attn_out)
        # Residual connection
        hidden_states = ttnn.add(residual, attn_out)
        residual_memory_config = hidden_states.memory_config()

        # Pre-MoE norm
        residual = hidden_states
        # hidden_states = self.pre_moe_norm(hidden_states)

        # MLP
        mlp_in = ttnn.to_memory_config(hidden_states, self.model_config["MLP_ACT_MEMCFG"])
        shared_mlp_out = self.shared_mlp.forward(mlp_in)
        shared_mlp_out = ttnn.to_memory_config(shared_mlp_out, residual_memory_config)

        # MoE
        moe_in = ttnn.to_memory_config(hidden_states, self.model_config["MOE_INPUT_MEMCFG"])
        moe_out = self.moe.forward(moe_in)
        moe_out = ttnn.to_memory_config(moe_out, residual_memory_config)
        moe_out = ttnn.add(moe_out, shared_mlp_out)
        moe_out = ttnn.div(moe_out, math.sqrt(2))

        # Post-MoE norm
        # moe_out = self.post_moe_norm(moe_out)
        hidden_states = ttnn.add(residual, moe_out)

        return hidden_states

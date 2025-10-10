import math

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.grok.tt.attention import Attention
from models.demos.grok.tt.distributed_norm import DistributedNorm
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
        deallocate_torch=False,
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
            deallocate_torch=deallocate_torch,
        )

        self.experts = ExpertMLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            args=args,
            layer_num=layer_num,
            dtypes={
                "w1": ttnn.bfloat4_b,
                "w2": ttnn.bfloat4_b,
                "w3": ttnn.bfloat4_b,
            },
            deallocate_torch=deallocate_torch,
        )
        self.moe = TtMoE(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            experts=self.experts,
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )

        # We have 4 norms: pre-attn, post-attn, pre-moe, post-moe
        self.pre_attn_norm = DistributedNorm(
            RMSNorm(
                device=self.mesh_device,
                dim=self.args.dim,
                eps=1e-5,
                state_dict={
                    f"model.layers.{layer_num}.pre_attn_norm.weight": state_dict[
                        f"model.layers.{layer_num}.pre_attn_norm.weight"
                    ]
                },
                # weight_cache_path=weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=f"model.layers.{layer_num}.pre_attn_norm",
                is_distributed=True,
                ccl_topology=ttnn.Topology.Ring,
                tt_ccl=tt_ccl,
            ),
            self.args,
            tt_ccl=tt_ccl,
        )
        self.post_attn_norm = DistributedNorm(
            RMSNorm(
                device=self.mesh_device,
                dim=self.args.dim,
                eps=1e-5,
                state_dict={
                    f"model.layers.{layer_num}.post_attn_norm.weight": state_dict[
                        f"model.layers.{layer_num}.post_attn_norm.weight"
                    ]
                },
                # weight_cache_path=weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=f"model.layers.{layer_num}.post_attn_norm",
                is_distributed=True,
                ccl_topology=ttnn.Topology.Ring,
                tt_ccl=tt_ccl,
            ),
            self.args,
            tt_ccl=tt_ccl,
        )
        self.pre_moe_norm = DistributedNorm(
            RMSNorm(
                device=self.mesh_device,
                dim=self.args.dim,
                eps=1e-5,
                state_dict={
                    f"model.layers.{layer_num}.pre_moe_norm.weight": state_dict[
                        f"model.layers.{layer_num}.pre_moe_norm.weight"
                    ]
                },
                # weight_cache_path=weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=f"model.layers.{layer_num}.pre_moe_norm",
                is_distributed=True,
                ccl_topology=ttnn.Topology.Ring,
                tt_ccl=tt_ccl,
            ),
            self.args,
            tt_ccl=tt_ccl,
        )
        self.post_moe_norm = DistributedNorm(
            RMSNorm(
                device=self.mesh_device,
                dim=self.args.dim,
                eps=1e-5,
                state_dict={
                    f"model.layers.{layer_num}.post_moe_norm.weight": state_dict[
                        f"model.layers.{layer_num}.post_moe_norm.weight"
                    ]
                },
                # weight_cache_path=weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=f"model.layers.{layer_num}.post_moe_norm",
                is_distributed=True,
                ccl_topology=ttnn.Topology.Ring,
                tt_ccl=tt_ccl,
            ),
            self.args,
            tt_ccl=tt_ccl,
        )

    def forward(self, hidden_states, current_pos, rot_mats, page_table=None):
        # Pre-attn norm
        residual = hidden_states
        input_memory_config = hidden_states.memory_config()

        hidden_states = self.pre_attn_norm(hidden_states, mode="decode")

        # Attention
        attn_out = self.attention.forward(hidden_states, current_pos, rot_mats, page_table=page_table)

        # Post-attn norm
        # attn_out = ttnn.typecast(attn_out, ttnn.bfloat16)
        attn_out = self.post_attn_norm(attn_out, mode="decode")

        # Residual connection
        hidden_states = ttnn.add(residual, attn_out)
        hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)
        residual_memory_config = hidden_states.memory_config()

        # Pre-MoE norm
        residual = hidden_states
        # hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)
        hidden_states = self.pre_moe_norm(hidden_states, mode="decode")

        # MLP
        mlp_in = ttnn.to_memory_config(hidden_states, self.model_config["MLP_ACT_MEMCFG"])
        shared_mlp_out = self.shared_mlp.forward(mlp_in)
        shared_mlp_out = ttnn.to_memory_config(shared_mlp_out, residual_memory_config)

        # MoE
        moe_in = ttnn.to_memory_config(hidden_states, self.model_config["MOE_INPUT_MEMCFG"])
        moe_out = self.moe.forward(moe_in)
        # moe_out = self.moe.forward_batch_1_tp_32(moe_in)
        moe_out = ttnn.to_memory_config(moe_out, residual_memory_config)
        moe_out = ttnn.add(moe_out, shared_mlp_out)
        moe_out = ttnn.div(moe_out, math.sqrt(2))

        # Post-MoE norm
        moe_out = ttnn.typecast(moe_out, ttnn.bfloat16)
        moe_out = self.post_moe_norm(moe_out, mode="decode")

        hidden_states = ttnn.add(residual, moe_out)
        # hidden_states = ttnn.to_memory_config(hidden_states, input_memory_config)

        return hidden_states

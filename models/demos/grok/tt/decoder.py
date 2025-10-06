import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.grok.tt.attention import Attention
from models.demos.grok.tt.experts import ExpertMLP
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
        self.state_dict = state_dict
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype
        self.model_config = args.get_model_config()

        self.attention = Attention(
            mesh_device,
            tt_ccl,
            state_dict,
            weight_cache_path,
            args,
            layer_num,
            dtype,
            transformation_mats,
            paged_attention_config,
        )
        self.shared_mlp = MLP(mesh_device, tt_ccl, state_dict, weight_cache_path, args, layer_num, dtype)

        self.moe = TtMoE(mesh_device, tt_ccl, state_dict, weight_cache_path, args, layer_num, dtype)
        self.experts = ExpertMLP(mesh_device, tt_ccl, state_dict, weight_cache_path, args, layer_num, dtype)

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

        # Pre-MoE norm
        residual = hidden_states
        # hidden_states = self.pre_moe_norm(hidden_states)

        # MLP
        shared_mlp_out = self.shared_mlp.forward(hidden_states)
        # MoE
        moe_out = self.moe.forward(hidden_states)
        moe_out = ttnn.add(moe_out, shared_mlp_out)  # need to add: divide by sqrt(2)

        # Post-MoE norm
        # moe_out = self.post_moe_norm(moe_out)
        hidden_states = ttnn.add(residual, moe_out)

        return hidden_states

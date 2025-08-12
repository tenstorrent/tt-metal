import ttnn
from models.experimental.stable_diffusion_35_large.tt.substate import substate

from .attention import Attention
from .mlp import MLP
from .rms_norm import RMSNorm


class DecoderLayer:
    def __init__(self, mesh_device, hf_config, state_dict, layer_idx, ccl_manager, dtype=ttnn.bfloat16):
        self.input_layernorm = RMSNorm(mesh_device, hf_config, substate(state_dict, "input_layernorm"))
        self.post_attention_layernorm = RMSNorm(
            mesh_device, hf_config, substate(state_dict, "post_attention_layernorm")
        )
        self.mlp = MLP(mesh_device, hf_config, substate(state_dict, "mlp"), ccl_manager, dtype=dtype)

        self.attention_type = hf_config.layer_types[layer_idx]

        self.self_attn = Attention(mesh_device, hf_config, substate(state_dict, "self_attn"), layer_idx, ccl_manager)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            x=hidden_states,
            mask=attention_mask,
            rope_stuff=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
        hidden_states = residual + hidden_states
        return hidden_states

from models.experimental.stable_diffusion_35_large.tt.substate import substate

from .mlp import MLP
from .rms_norm import RMSNorm


class DecoderLayer:
    def __init__(self, mesh_device, hf_config, state_dict, layer_idx, ccl_manager):
        self.input_layernorm = RMSNorm(mesh_device, hf_config, substate(state_dict, "input_layernorm"))
        self.post_attention_layernorm = RMSNorm(
            mesh_device, hf_config, substate(state_dict, "post_attention_layernorm")
        )
        self.mlp = MLP(mesh_device, hf_config, substate(state_dict, "mlp"), ccl_manager)

        self.attention_type = hf_config.layer_types[layer_idx]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        cache_position=None,
        position_embeddings=None,
    ):
        # residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        # hidden_states, _ = self.self_attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     cache_position=cache_position,
        #     position_embeddings=position_embeddings,
        #     **kwargs,
        # )
        # hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
        hidden_states = residual + hidden_states
        return hidden_states

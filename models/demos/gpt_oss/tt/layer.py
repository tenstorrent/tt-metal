import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.experimental.stable_diffusion_35_large.tt.substate import substate

from .attention import Attention
from .mlp import MLP
from .rms_norm import RMSNorm


class DecoderLayer:
    def __init__(
        self, mesh_device, hf_config, state_dict, layer_idx, ccl_manager, dtype=ttnn.bfloat16, tensor_cache_path=None
    ):
        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
        )
        self.mlp = MLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            ccl_manager,
            dtype=dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
        )

        self.attention_type = hf_config.layer_types[layer_idx]

        self.self_attn = Attention(
            mesh_device,
            hf_config,
            substate(state_dict, "self_attn"),
            layer_idx,
            ccl_manager,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
        )

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

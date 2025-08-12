import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.experimental.stable_diffusion_35_large.tt.substate import substate

from .layer import DecoderLayer
from .rms_norm import RMSNorm


class Model:
    def __init__(self, mesh_device, hf_config, state_dict, ccl_manager, dtype=ttnn.bfloat16, tensor_cache_path=None):
        self.mesh_device = mesh_device
        embedding_weight = substate(state_dict, "model.embed_tokens")["weight"]
        embedding_weight = embedding_weight.unsqueeze(0).unsqueeze(0)
        self.embedding_weight = ttnn.as_tensor(
            embedding_weight,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "model.embed_tokens.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layers = [
            DecoderLayer(
                mesh_device,
                hf_config,
                substate(state_dict, f"model.layers.{layer_idx}"),
                layer_idx,
                ccl_manager,
                dtype=dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, f"model.layers.{layer_idx}"),
            )
            for layer_idx in range(hf_config.num_hidden_layers)
        ]
        self.norm = RMSNorm(mesh_device, hf_config, substate(state_dict, "model.norm"))
        self.lm_head_weight = ttnn.as_tensor(
            substate(state_dict, "lm_head")["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        input_ids,
        attention_masks,
        position_embeddings,
    ):
        input_embeds = ttnn.embedding(input_ids, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        hidden_states = input_embeds
        for decoder_layer in self.layers:
            mask = attention_masks[decoder_layer.attention_type]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = ttnn.matmul(hidden_states, self.lm_head_weight)
        return hidden_states

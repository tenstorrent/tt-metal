# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .attention import Attention, AttentionConfig
from .attention_configs import GPTOSSAttentionProgramConfig
from .mlp import MLP
from .rms_norm import RMSNorm


class DecoderLayer:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=None,
        create_kv_cache=True,
        transformation_mats=None,
    ):
        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
        )
        self.mlp = MLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            ccl_manager,
            dtype=dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
            mesh_config=mesh_config,
        )

        self.attention_type = hf_config.layer_types[layer_idx]

        # Create attention configuration
        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            max_seq_len=hf_config.max_position_embeddings,
            sliding_window=hf_config.sliding_window,
        )

        # Create attention program config
        attention_program_config = GPTOSSAttentionProgramConfig()

        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attention_program_config,
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=transformation_mats,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
            create_kv_cache=create_kv_cache,
        )
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
    ):
        # Input state: row-sharded (4), column-replicated (8)
        mem = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, 32),  # hidden_size_per_device_distributed_ln//num_cores_ln),
            core_grid=ttnn.CoreGrid(y=3, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        hidden_states = ttnn.to_memory_config(hidden_states, mem)
        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)

        # Attention (includes 2D sharding all-reduces)
        # Output: row-sharded (4), column-replicated (8)
        hidden_states = self.self_attn(
            hidden_states_post_norm,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
        )
        print(residual.memory_config())
        print(hidden_states.memory_config())
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=residual)
        residual = hidden_states

        # Second RMSNorm
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        # hidden_states_post_norm = hidden_states
        # All-gather along ROWS before MLP (MoE router needs full hidden states)
        # Input: row-sharded (4), col-replicated (8) → Output: fully replicated
        num_rows = self.mesh_config.mesh_shape[0]
        if num_rows > 1:
            hidden_states_post_norm = ttnn.all_gather(
                hidden_states_post_norm,
                dim=-1,
                topology=ttnn.Topology.Linear,
                cluster_axis=0,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            hidden_states_post_norm = ttnn.reshape(hidden_states_post_norm, (1, 1, 2944), (1, 32, 2944))
            hidden_states_post_norm = hidden_states_post_norm[:, :, :2880]
        print("before moe", hidden_states_post_norm.shape)

        # MLP with MoE (expects fully replicated input)
        # Output: fully replicated after EP + TP all-reduces
        hidden_states, _ = self.mlp(hidden_states_post_norm)  # diff with llama: router scores
        print("after moe", hidden_states.shape)
        # # Residual add (both have same sharding now)
        # hidden_states = ttnn.to_memory_config(hidden_states, mem)
        hidden_states = hidden_states[:, :, :, :736]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=residual)
        return hidden_states

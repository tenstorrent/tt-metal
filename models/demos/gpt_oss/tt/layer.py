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
        max_seq_len=1024,
        max_local_batch_size=1,
        users_row_sharded=False,
        use_throughput_experts=False,
    ):
        # Enable distributed RMS norm when activations are WIDTH sharded across TP dimension
        # This avoids all_gather by computing local stats and gathering them instead
        # Reference: llama3_70b_galaxy uses this pattern for TP > 1
        use_distributed_norm = (mesh_config is not None and mesh_config.tp > 1) or users_row_sharded

        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            mesh_config=mesh_config,
            is_distributed=use_distributed_norm,
            ccl_manager=ccl_manager,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            mesh_config=mesh_config,
            is_distributed=use_distributed_norm,
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
            use_throughput_experts=use_throughput_experts,
        )

        self.attention_type = hf_config.layer_types[layer_idx]
        self.layer_idx = layer_idx

        # Create attention configuration
        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            sliding_window=hf_config.sliding_window,
            max_seq_len=max_seq_len,
            max_local_batch_size=max_local_batch_size,
            users_row_sharded=users_row_sharded,
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

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        user_id=0,
    ):
        # hidden_states: [1, 1, tokens/num_rows, hidden_size/num_columns]
        # residual: [1, 1, tokens/num_rows, hidden_size/num_columns]

        # CRITICAL FIX: Create an explicit copy of residual to prevent buffer corruption
        # from async operations like allgather that may share underlying memory.
        # Without this clone, the allgather on hidden_states_post_norm can corrupt
        # the residual tensor if they share the same underlying buffer.
        residual = ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states_post_norm = self.input_layernorm(hidden_states)

        # All-gather over TP axis before attention (skip if norm already gathered)
        if self.self_attn.mesh_config.tp > 1 and not self.input_layernorm.output_is_gathered:
            hidden_states_post_norm = self.self_attn.mesh_config.allgather(
                hidden_states_post_norm,
                self.self_attn.ccl_manager,
                axis=self.self_attn.mesh_config.tp_axis,
                dim=3,
            )

        hidden_states = self.self_attn(
            hidden_states_post_norm,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            user_id=user_id,
        )
        hidden_states_post_norm.deallocate(True)

        # after reduce scatter at end of attn: [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        # CRITICAL FIX: Create an explicit copy of residual to prevent buffer corruption
        # from async operations like allgather that may share underlying memory.
        residual = ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)

        # All-gather over TP axis before MLP (skip if norm already gathered)
        if self.self_attn.mesh_config.tp > 1 and not self.post_attention_layernorm.output_is_gathered:
            hidden_states_post_norm = self.self_attn.mesh_config.allgather(
                hidden_states_post_norm,
                self.self_attn.ccl_manager,
                axis=self.self_attn.mesh_config.tp_axis,
                dim=3,
            )
        hidden_states = self.mlp(hidden_states_post_norm, is_decode=is_decode)  # diff with llama: router scores
        hidden_states_post_norm.deallocate(True)

        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        return hidden_states

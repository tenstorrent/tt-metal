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
        self.hidden_size = hf_config.hidden_size

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
    ):
        # Input state: row-sharded (4), column-replicated (8)
        # Calculate per-device sizes dynamically
        num_rows = self.mesh_config.mesh_shape[0]

        # Calculate padded hidden size (must be divisible by num_rows * TILE_SIZE)
        shard_chunk_size = num_rows * ttnn.TILE_SIZE
        if self.hidden_size % shard_chunk_size != 0:
            padded_hidden_size = ((self.hidden_size + shard_chunk_size - 1) // shard_chunk_size) * shard_chunk_size
        else:
            padded_hidden_size = self.hidden_size
        per_device_hidden = padded_hidden_size // num_rows if num_rows > 1 else padded_hidden_size
        batch_size = int(hidden_states.shape[0])
        padded_seq_len = int(hidden_states.shape[-2])
        seq_len = int(position_idx.shape[-1]) if position_idx is not None else padded_seq_len
        residual_mem_config = hidden_states.memory_config()
        is_prefill = seq_len > 1

        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)
        if is_prefill:
            hidden_states_post_norm = ttnn.to_memory_config(hidden_states_post_norm, ttnn.DRAM_MEMORY_CONFIG)

        # Attention (includes 2D sharding all-reduces)
        # Output: row-sharded (4), column-replicated (8)
        hidden_states = self.self_attn(
            hidden_states_post_norm,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
        )
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=residual)
        residual = hidden_states

        # Second RMSNorm
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        if is_prefill:
            hidden_states_post_norm = ttnn.to_memory_config(hidden_states_post_norm, ttnn.DRAM_MEMORY_CONFIG)

        # All-gather along ROWS before MLP (MoE router needs full hidden states)
        # Input: row-sharded (4), col-replicated (8) → Output: fully replicated
        if num_rows > 1:
            if is_prefill:
                hidden_states_post_norm = ttnn.to_memory_config(hidden_states_post_norm, ttnn.DRAM_MEMORY_CONFIG)
            hidden_states_post_norm = ttnn.all_gather(
                hidden_states_post_norm,
                dim=-1,
                topology=ttnn.Topology.Linear,
                cluster_axis=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if is_prefill else ttnn.L1_MEMORY_CONFIG,
            )
            hidden_states_post_norm = ttnn.reshape(
                hidden_states_post_norm,
                (batch_size, seq_len, padded_hidden_size),
                (batch_size, padded_seq_len, padded_hidden_size),
            )
            hidden_states_post_norm = hidden_states_post_norm[:, :, : self.hidden_size]
        else:
            hidden_states_post_norm = ttnn.reshape(
                hidden_states_post_norm,
                (batch_size, seq_len, per_device_hidden),
                (batch_size, padded_seq_len, per_device_hidden),
            )
            if is_prefill:
                hidden_states_post_norm = ttnn.to_memory_config(hidden_states_post_norm, ttnn.DRAM_MEMORY_CONFIG)

        # MLP with MoE (expects fully replicated input)
        # Output: fully replicated after EP + TP all-reduces
        hidden_states, _ = self.mlp(hidden_states_post_norm)  # diff with llama: router scores

        mlp_seq_len = int(hidden_states.shape[-2])
        if num_rows > 1:
            local_hidden = padded_hidden_size // num_rows
            current_shape = tuple(int(dim) for dim in hidden_states.shape)
            if current_shape[-1] > local_hidden:
                hidden_states = ttnn.slice(
                    hidden_states,
                    (0, 0, 0, 0),
                    (current_shape[0], current_shape[1], current_shape[2], local_hidden),
                )
            hidden_states = ttnn.reshape(
                hidden_states,
                (batch_size, seq_len, local_hidden),
                (batch_size, mlp_seq_len, local_hidden),
            )
            hidden_states = ttnn.reshape(
                hidden_states,
                (batch_size, 1, seq_len, local_hidden),
                (batch_size, mlp_seq_len, local_hidden),
            )
            hidden_states = ttnn.to_memory_config(hidden_states, residual_mem_config)
        else:
            hidden_states = ttnn.reshape(
                hidden_states,
                (batch_size, seq_len, self.hidden_size),
                (batch_size, mlp_seq_len, self.hidden_size),
            )
            if padded_hidden_size > self.hidden_size:
                hidden_states = ttnn.pad(
                    hidden_states,
                    [(0, 0), (0, 0), (0, padded_hidden_size - self.hidden_size)],
                    0,
                )
            hidden_states = ttnn.reshape(
                hidden_states,
                (batch_size, 1, seq_len, padded_hidden_size),
                (batch_size, mlp_seq_len, padded_hidden_size),
            )
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=residual)
        return hidden_states

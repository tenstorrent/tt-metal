# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .attention import Attention, AttentionConfig
from .attention_configs import GPTOSSAttentionProgramConfig
from .fused_decode import residual_memory_config
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
        tokens_per_device=32,
    ):
        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            mesh_config=mesh_config,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            mesh_config=mesh_config,
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
            tokens_per_device=tokens_per_device,
        )

        self.attention_type = hf_config.layer_types[layer_idx]

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
            fused_decode=self.mlp.indexed_decode,
        )
        self.mesh_device = mesh_device
        # Fused decode (one token per device, TP over mesh columns): the residual stream stays width-sharded in L1
        # between the fused all-reduces (fused_decode.py) and the MoE computes only the routed experts.
        self.fused_decode = self.mlp.indexed_decode
        if self.fused_decode:
            self.residual_memory_config = residual_memory_config(mesh_device, hf_config.hidden_size)
            ccl_manager.get_decode_all_reduce(hf_config.hidden_size, mesh_config.tp_axis)

    def _decode_forward(self, hidden_states, position_embeddings, position_idx, page_table, kv_cache):
        """One decode token: sharded norm -> attention (+ fused all-reduce) -> residual add -> sharded norm -> MoE
        (+ fused all-reduce) -> residual add. Returns the BF16 residual stream in the width-sharded decode layout."""
        if hidden_states.memory_config() != self.residual_memory_config:
            # First layer: the embedding output enters the decode residual layout.
            embeddings = hidden_states
            hidden_states = ttnn.to_memory_config(embeddings, self.residual_memory_config)
            embeddings.deallocate(True)
        residual = hidden_states
        attn_in = self.input_layernorm.forward_sharded(hidden_states)
        attn_out = self.self_attn(
            attn_in,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
        )
        attn_in.deallocate(True)
        hidden_states = ttnn.add(residual, attn_out, memory_config=self.residual_memory_config, dtype=ttnn.bfloat16)
        attn_out.deallocate(True)
        residual.deallocate(True)

        residual = hidden_states
        mlp_in = self.post_attention_layernorm.forward_sharded(hidden_states)
        mlp_out = self.mlp(mlp_in, is_decode=True)
        mlp_in.deallocate(True)
        hidden_states = ttnn.add(residual, mlp_out, memory_config=self.residual_memory_config, dtype=ttnn.bfloat16)
        mlp_out.deallocate(True)
        residual.deallocate(True)
        return hidden_states

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        user_id=0,
        batch_size=1,
    ):
        if is_decode and self.fused_decode:
            return self._decode_forward(hidden_states, position_embeddings, position_idx, page_table, kv_cache)

        seqlen = hidden_states.shape[-2]
        if seqlen > 32 * 1024:
            # Reallocate hidden states to prevent memory fragmentation.
            hidden_states = ttnn.move(hidden_states)

        # hidden_states: [1, 1, tokens/num_rows, hidden_size/num_columns]
        # residual: [1, 1, tokens/num_rows, hidden_size/num_columns]
        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)

        # additional all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]
        # hidden_states_post_norm: [1, 1, tokens/num_rows, hidden_size]
        hidden_states = self.self_attn(
            hidden_states_post_norm,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            user_id=user_id,
            batch_size=batch_size,
        )
        hidden_states_post_norm.deallocate(True)

        # after reduce scatter at end of attn: [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        # another all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]

        hidden_states = self.mlp(hidden_states_post_norm, is_decode=is_decode)  # diff with llama: router scores
        hidden_states_post_norm.deallocate(True)

        # TODO: replace all_reduce at end of MLP with reduce_scatter so we get [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        return hidden_states

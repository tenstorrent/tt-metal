# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.minimax_m3.utils.general_utils import get_cache_file_name
from models.demos.minimax_m3.utils.substate import substate

from .attention import Attention, AttentionConfig
from .attention_configs import MiniMaxM3AttentionProgramConfig
from .dense_mlp import DenseMLP
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
        mesh_config=None,
        transformation_mats=None,
        max_seq_len=1024,
        max_local_batch_size=1,
        users_row_sharded=False,
        expert_weight_dtype=ttnn.bfloat4_b,
        use_ep_moe=False,
        ep_seq_len_per_chip=1024,
        sequence_parallel=False,
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
        # Hybrid dense/MoE schedule (M3): layers with moe_layer_freq[idx]==0 are a plain dense
        # SwiGLU MLP (mlp.{gate,up,down}_proj); the rest are MoE (block_sparse_moe.*). If
        # moe_layer_freq is absent, default to MoE.
        moe_layer_freq = getattr(hf_config, "moe_layer_freq", None)
        self.is_dense = (
            moe_layer_freq is not None and layer_idx < len(moe_layer_freq) and moe_layer_freq[layer_idx] == 0
        )
        if self.is_dense:
            self.mlp = DenseMLP(
                mesh_device,
                hf_config,
                substate(state_dict, "mlp"),
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
            )
        else:
            self.mlp = MLP(
                mesh_device,
                hf_config,
                # MiniMax-M3 names the MoE block 'block_sparse_moe'.
                substate(state_dict, "block_sparse_moe"),
                ccl_manager,
                dtype=dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
                mesh_config=mesh_config,
                expert_weight_dtype=expert_weight_dtype,
                use_ep_moe=use_ep_moe,
                ep_seq_len_per_chip=ep_seq_len_per_chip,
            )

        # MiniMax-M3 lists per-layer attention types in `attn_type_list` (all 1 =
        # full attention) via attn_type_list. Fall back gracefully.
        attn_types = getattr(hf_config, "attn_type_list", None) or getattr(hf_config, "layer_types", None)
        self.attention_type = attn_types[layer_idx] if attn_types is not None else 1

        # M3 MSA: layers with sparse_attention_freq[layer_idx]==1 (layers 3-59) run block-sparse
        # attention; layers 0-2 (==0) stay dense. sparse_attention_config may be a dict or an object.
        sparse_cfg = getattr(hf_config, "sparse_attention_config", None)
        if isinstance(sparse_cfg, dict):
            freq = sparse_cfg.get("sparse_attention_freq") if sparse_cfg.get("use_sparse_attention") else None
        else:
            freq = getattr(sparse_cfg, "sparse_attention_freq", None) if sparse_cfg is not None else None
        is_sparse = bool(freq[layer_idx]) if freq is not None and layer_idx < len(freq) else False

        # MSA hyperparams from sparse_attention_config (dict or object); defaults match M3.
        def _sc(name, default):
            if sparse_cfg is None:
                return default
            return sparse_cfg.get(name, default) if isinstance(sparse_cfg, dict) else getattr(sparse_cfg, name, default)

        msa_block_size = _sc("sparse_block_size", 128)
        msa_topk_blocks = _sc("sparse_topk_blocks", 16)
        msa_index_dim = _sc("sparse_index_dim", 128)

        # Create attention configuration
        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            rotary_dim=getattr(hf_config, "rotary_dim", hf_config.head_dim),
            rms_norm_eps=hf_config.rms_norm_eps,
            use_qk_norm=getattr(hf_config, "use_qk_norm", True),
            use_gemma_norm=getattr(hf_config, "use_gemma_norm", False),
            sliding_window=getattr(hf_config, "sliding_window", None),
            max_seq_len=max_seq_len,
            max_local_batch_size=max_local_batch_size,
            users_row_sharded=users_row_sharded,
            is_sparse=is_sparse,
            msa_block_size=msa_block_size,
            msa_topk_blocks=msa_topk_blocks,
            msa_index_dim=msa_index_dim,
            sequence_parallel=sequence_parallel,
        )

        # Create attention program config
        attention_program_config = MiniMaxM3AttentionProgramConfig()

        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attention_program_config,
            layer_idx=layer_idx,
            transformation_mats=transformation_mats,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
        )
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        position_idx=None,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
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
            kv_cache=kv_cache,
            user_id=user_id,
            batch_size=batch_size,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
        hidden_states_post_norm.deallocate(True)

        # after reduce scatter at end of attn: [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        # another all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]

        hidden_states = self.mlp(hidden_states_post_norm)
        hidden_states_post_norm.deallocate(True)

        # TODO: replace all_reduce at end of MLP with reduce_scatter so we get [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        return hidden_states

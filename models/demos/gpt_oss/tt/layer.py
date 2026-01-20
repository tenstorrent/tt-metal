# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .attention import Attention, AttentionConfig
from .attention_configs import GPTOSSAttentionProgramConfig
from .mlp import MLP
from .rms_norm import RMSNorm

# Debug mode: Set DEBUG_LAYER_OPS=1 to enable per-operation logging
DEBUG_LAYER_OPS = os.getenv("DEBUG_LAYER_OPS", "0") == "1"


def _debug_tensor_stats(name, tensor, layer_idx=None):
    """Log tensor statistics for debugging."""
    if not DEBUG_LAYER_OPS:
        return

    import torch
    from loguru import logger

    # Get first device tensor for inspection
    device_tensors = ttnn.get_device_tensors(tensor)
    t = ttnn.to_torch(device_tensors[0]).float()

    prefix = f"[L{layer_idx}]" if layer_idx is not None else "[LAYER]"
    stats = f"shape={list(t.shape)}, min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}, std={t.std().item():.4f}"

    # Check for extreme values
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    max_abs = t.abs().max().item()

    if has_nan or has_inf:
        stats += f" ⚠️ NaN={has_nan}, Inf={has_inf}"
    if max_abs > 1000:
        stats += f" ⚠️ EXTREME"

    logger.debug(f"{prefix} {name}: {stats}")


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
        )
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx

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
        _debug_tensor_stats("input", hidden_states, self.layer_idx)

        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)
        _debug_tensor_stats("after_input_layernorm", hidden_states_post_norm, self.layer_idx)

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
        )
        hidden_states_post_norm.deallocate(True)
        _debug_tensor_stats("after_self_attn", hidden_states, self.layer_idx)

        # after reduce scatter at end of attn: [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        _debug_tensor_stats("after_attn_residual", hidden_states, self.layer_idx)

        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        _debug_tensor_stats("after_post_attn_layernorm", hidden_states_post_norm, self.layer_idx)
        # another all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]

        # Set layer index for expert debug logging (both regular and throughput experts)
        from models.demos.gpt_oss.tt.experts import prefill as expert_prefill
        from models.demos.gpt_oss.tt.experts_throughput import decode as throughput_decode

        expert_prefill._current_layer_idx = self.layer_idx
        throughput_decode._current_layer_idx = self.layer_idx

        hidden_states = self.mlp(hidden_states_post_norm, is_decode=is_decode)  # diff with llama: router scores
        hidden_states_post_norm.deallocate(True)
        _debug_tensor_stats("after_mlp", hidden_states, self.layer_idx)

        # TODO: replace all_reduce at end of MLP with reduce_scatter so we get [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        _debug_tensor_stats("after_mlp_residual", hidden_states, self.layer_idx)

        return hidden_states

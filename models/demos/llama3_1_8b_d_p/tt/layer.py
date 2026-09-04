# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B decoder layer. Copied from ``gpt_oss_d_p/tt/layer.py``.

    input_layernorm -> Attention -> residual add -> post_attention_layernorm -> MLP -> residual add

Every layer is identical: full-causal GQA attention plus a dense SwiGLU MLP. There is no MoE, no
shared expert, no sliding/full alternation and no ``layer_types`` dispatch — the spec's
``layer_schedule`` is null, so ``layer_idx`` only ever selects the KV cache slot row, never a
different block.
"""

import ttnn
from models.demos.llama3_1_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama3_1_8b_d_p.utils.substate import substate

from .attention import Attention, AttentionConfig, ProgramConfig
from .mlp import MLP
from .rms_norm import RMSNorm


def build_attention_config(hf_config, *, max_seq_len, sequence_parallel=False) -> AttentionConfig:
    """AttentionConfig from an HF-shaped config. Shared by every layer (never mutated per layer)."""
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    return AttentionConfig(
        hidden_size=hf_config.hidden_size,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rotary_dim=head_dim,  # full rotary
        rms_norm_eps=hf_config.rms_norm_eps,
        sequence_parallel=sequence_parallel,
    )


class DecoderLayer:
    """One Llama decoder layer on the target mesh."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        tensor_cache_path=None,
        mesh_config=None,
        transformation_mats=None,
        max_seq_len=1024,
        attn_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat4_b,
        sequence_parallel=False,
    ):
        """
        Args:
            state_dict: this layer's substate — ``input_layernorm.*``, ``self_attn.*``,
                ``post_attention_layernorm.*``, ``mlp.*``. Empty dict => cache-only load.
            attn_weight_dtype / mlp_weight_dtype: the spec asks for bfp4 on both; tt_transformers'
                accuracy path uses bfp8 for WQKV/WO. The two disagree (spec known_risks), so the
                default here is the conservative bfp8 for attention and the spec's bfp4 for the MLP,
                and both are overridable so the difference can be measured rather than argued.
        """
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx

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
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            weight_dtype=mlp_weight_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
        )

        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=build_attention_config(hf_config, max_seq_len=max_seq_len, sequence_parallel=sequence_parallel),
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=ProgramConfig(),
            layer_idx=layer_idx,
            transformation_mats=transformation_mats,
            weight_dtype=attn_weight_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
        )

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
        """hidden_states / residual: ``[1, 1, tokens_per_sp_row, hidden]``."""
        seqlen = hidden_states.shape[-2]
        if seqlen > 32 * 1024:
            # Reallocate to prevent DRAM fragmentation at long context.
            hidden_states = ttnn.move(hidden_states)

        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states_post_norm,
            rope_mats=position_embeddings,
            kv_cache=kv_cache,
            user_id=user_id,
            batch_size=batch_size,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
        hidden_states_post_norm.deallocate(True)
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_post_norm)
        hidden_states_post_norm.deallocate(True)
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        return hidden_states

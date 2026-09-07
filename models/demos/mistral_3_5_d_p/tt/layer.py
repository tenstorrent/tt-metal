# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 decoder layer. Adapted from ``gpt_oss_d_p/tt/layer.py``.

    input_layernorm -> Attention -> residual add -> post_attention_layernorm -> MLP -> residual add

This is the block-level composition D2 freezes the signatures of: the layer is nothing but four
named blocks and two residual adds, and every block below it has its own ``*_vs_ref`` PCC test.

Unlike the gpt-oss donor there is no MoE branch and no per-layer type dispatch: EVERY Mistral layer
is the same dense-GQA + dense-SwiGLU pair (``sliding_window`` is null, ``moe_layer_freq`` does not
exist), so all 88 layers are built from one code path with one config.
"""

import ttnn
from models.demos.mistral_3_5_d_p.utils.general_utils import get_cache_file_name
from models.demos.mistral_3_5_d_p.utils.substate import substate

from .attention import Attention, AttentionConfig, ProgramConfig
from .mlp import MLP
from .rms_norm import RMSNorm

# Above this per-chunk token count the residual stream is re-homed before the layer runs, to keep
# long-context prefill from fragmenting DRAM (donor behaviour, kept).
_MOVE_HIDDEN_SEQ_THRESHOLD = 32 * 1024


class DecoderLayer:
    """One Mistral-Medium-3.5 decoder layer: norm -> attention -> add -> norm -> MLP -> add."""

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
        sequence_parallel=False,
        program_config=None,
        weight_dtype=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device
            hf_config: HF text config (dims, eps, rope parameters)
            state_dict: this layer's HF sub-state (``self_attn.*``, ``mlp.*``, the two norms), with
                q/k already Meta-swizzled. Empty dict -> cache-only load.
            layer_idx: global layer index — also this layer's offset inside each user's cache slot
            ccl_manager: Communication manager
            dtype: activation dtype for the residual stream
            tensor_cache_path: Optional path for weight caching
            mesh_config: Mesh parallelization config
            transformation_mats: ``{"prefill": tensor}`` RoPE transformation matrices
            max_seq_len: cache capacity in tokens, threaded into AttentionConfig
            max_local_batch_size: users packed per device (1 for bring-up)
            sequence_parallel: take the SP cache-backed ring attention path
            program_config: Optional ProgramConfig override (defaults to this model's)
            weight_dtype: override the spec's weight dataformats for attention and the MLP. None
                (the default) uses the spec. Exists for bring-up A/Bs — the measured cost of the
                spec's bfloat8_b weights on per-layer KV PCC at depth is recorded in README.md, and
                that number came from flipping this.
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

        # Dense SwiGLU on every layer. HF names the block `mlp` with mlp.{gate,up,down}_proj.
        self.mlp = MLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
            weight_dtype=weight_dtype,
        )

        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            max_seq_len=max_seq_len,
            rotary_dim=hf_config.head_dim,  # full rotary
            rms_norm_eps=hf_config.rms_norm_eps,
            sequence_parallel=sequence_parallel,
        )
        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=program_config or ProgramConfig(),
            layer_idx=layer_idx,
            transformation_mats=transformation_mats,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
            weight_dtype=weight_dtype,
        )

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
        """hidden_states / residual: [1, 1, tokens/sp, hidden_size] (SP-sharded on the seq dim,
        replicated across TP)."""
        if hidden_states.shape[-2] > _MOVE_HIDDEN_SEQ_THRESHOLD:
            # Reallocate the residual to keep long-context prefill from fragmenting DRAM.
            hidden_states = ttnn.move(hidden_states)

        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)

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

        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states_post_norm)
        hidden_states_post_norm.deallocate(True)

        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        return hidden_states

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS decoder layer. Mirrors ``minimax_m3/tt/layer.py``.

    input_layernorm -> Attention -> residual add -> post_attention_layernorm -> MLP -> residual add

Unlike M3 there is NO dense-layer branch and NO shared expert: EVERY layer is a MoE layer
(router + expert-parallel routed experts). Sliding-window vs full-causal attention is selected per
layer from ``hf_config.layer_types[layer_idx]`` (see attention/__init__.py:Attention).
"""

import os

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.substate import substate

_DELTA_PROBE = os.environ.get("GPT_OSS_DELTA_PROBE", "") != ""


def _delta_stats(tag, layer_idx, t):
    """Bring-up probe (GPT_OSS_DELTA_PROBE): log per-layer L2 norm / mean-abs / signed-mean of a
    residual delta from device(0)'s shard. A growing signed-mean = a directional bias accumulating in
    that sublayer's output (the fingerprint of the per-layer logic error hitting V harder than K)."""
    try:
        import torch  # noqa

        from loguru import logger

        d0 = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        logger.info(
            f"[delta-probe L{layer_idx:>2}] {tag}: L2={d0.norm():.3f}  mean|x|={d0.abs().mean():.4f}  "
            f"signed_mean={d0.mean():.5f}  max|x|={d0.abs().max():.3f}"
        )
    except Exception as e:  # never let the probe break a run
        from loguru import logger

        logger.warning(f"[delta-probe] failed at L{layer_idx} {tag}: {e}")

from .attention import Attention, AttentionConfig, ProgramConfig
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
        expert_weight_dtype=ttnn.bfloat4_b,
        use_ep_moe=True,
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

        # Every GPT-OSS layer is MoE (no dense branch, no shared expert). HF names the block `mlp`
        # with `mlp.router.*` + `mlp.experts.*`.
        self.mlp = MLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            ccl_manager,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
            mesh_config=mesh_config,
            expert_weight_dtype=expert_weight_dtype,
            use_ep_moe=use_ep_moe,
            ep_seq_len_per_chip=ep_seq_len_per_chip,
            layer_idx=layer_idx,
        )

        # Sliding vs full is selected inside Attention from hf_config.layer_types[layer_idx].
        layer_types = getattr(hf_config, "layer_types", None)

        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            max_seq_len=max_seq_len,
            # Keep the sliding-window SIZE on the config; Attention nulls it out for full layers.
            sliding_window=getattr(hf_config, "sliding_window", 128),
            rotary_dim=getattr(hf_config, "rotary_dim", hf_config.head_dim),  # full rotary
            rms_norm_eps=hf_config.rms_norm_eps,
            sequence_parallel=sequence_parallel,
        )

        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=ProgramConfig(),
            layer_idx=layer_idx,
            layer_types=layer_types,
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

        # hidden_states / residual: [1, 1, tokens/num_rows, hidden_size]
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

        if _DELTA_PROBE:
            _delta_stats("attn_out", self.layer_idx, hidden_states)

        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states_post_norm)
        hidden_states_post_norm.deallocate(True)

        if _DELTA_PROBE:
            _delta_stats("moe_out ", self.layer_idx, hidden_states)

        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        return hidden_states

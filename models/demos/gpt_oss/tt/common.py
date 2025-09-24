# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific implementation of create_tt_model that's compatible with tt_transformers
"""

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
):
    """
    GPT-OSS version of create_tt_model that matches tt_transformers interface
    """
    from models.demos.gpt_oss.tt.model import Model
    from models.demos.gpt_oss.tt.model_config import ModelArgs

    # Create GPT-OSS ModelArgs
    gpt_oss_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    # Note: num_layers parameter is intentionally not used to preserve full model architecture

    # Avoid loading state_dict for every DP model
    if not state_dict:
        state_dict = gpt_oss_model_args.load_state_dict()

    # Create GPT-OSS model using transformer-compatible constructor
    model = Model.create_transformer_compatible(
        args=gpt_oss_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=gpt_oss_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    # Extract tt_kv_cache like tt_transformers does
    # For GPT-OSS, layer_past points to the actual KV cache tensors [k_cache, v_cache]
    # KV cache is needed regardless of paged attention for proper generation
    tt_kv_cache = []
    for layer in model.layers:
        # GPT-OSS uses self_attn instead of attention
        tt_kv_cache.append(layer.self_attn.layer_past)

    return gpt_oss_model_args, model, tt_kv_cache, state_dict

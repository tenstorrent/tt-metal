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
    mesh_config=None,
    create_kv_cache=True,
):
    """
    GPT-OSS version of create_tt_model that matches tt_transformers interface
    Uses clean MeshConfig abstraction for optimal device parallelization
    """
    from models.demos.gpt_oss.config import MeshConfig
    from models.demos.gpt_oss.tt.model import Model
    from models.demos.gpt_oss.tt.model_config import ModelArgs

    # Use provided mesh_config or create optimal MeshConfig for the mesh shape
    if mesh_config is None:
        mesh_config = MeshConfig(mesh_device.shape, tp=mesh_device.shape[1], ep=mesh_device.shape[0])

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
        weight_cache_path=str(gpt_oss_model_args.weight_cache_path(dtype)),
        paged_attention_config=paged_attention_config,
        mesh_config=mesh_config,  # Pass explicit MeshConfig
        create_kv_cache=create_kv_cache,
    )

    # Extract tt_kv_cache like tt_transformers does
    tt_kv_cache = []
    if create_kv_cache:
        for layer in model.layers:
            # GPT-OSS uses self_attn instead of attention
            tt_kv_cache.append(layer.self_attn.layer_past)

    return gpt_oss_model_args, model, tt_kv_cache, state_dict

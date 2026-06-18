# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2 specific implementation of create_tt_model that's compatible with tt_transformers
"""

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig


def create_tt_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    optimizations=None,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
    mesh_config=None,
    create_kv_cache=True,
    users_row_sharded=False,
    use_throughput_experts=False,
):
    """
    MiniMax-M2 version of create_tt_model that matches tt_transformers interface
    Uses clean MeshConfig abstraction for optimal device parallelization
    """
    from models.demos.minimax_m3.config import MeshConfig
    from models.demos.minimax_m3.tt.model import Model
    from models.demos.minimax_m3.tt.model_config import ModelArgs

    # Use provided mesh_config or create optimal MeshConfig for the mesh shape
    if mesh_config is None:
        from models.demos.minimax_m3.config import ModeConfig

        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))

    # Create MiniMax-M2 ModelArgs
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    # Override num_layers if provided (useful for quick testing with fewer layers)
    if num_layers is not None:
        model_args.hf_config.num_hidden_layers = num_layers
        model_args.n_layers = num_layers

    # Avoid loading state_dict for every DP model. An empty dict is intentional
    # when --skip-model-load is used.
    if state_dict is None:
        state_dict = model_args.load_state_dict(
            weights_path=model_args.model_path,
            dummy_weights=model_args.dummy_weights,
            convert_to_meta_format=True,
        )

    # Create MiniMax-M2 model using transformer-compatible constructor
    model = Model.create_transformer_compatible(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        tensor_cache_path=str(model_args.weight_cache_path(dtype)),
        paged_attention_config=paged_attention_config,
        mesh_config=mesh_config,  # Pass explicit MeshConfig
        create_kv_cache=create_kv_cache,
        users_row_sharded=users_row_sharded,
        use_throughput_experts=use_throughput_experts,
    )

    # Extract tt_kv_cache like tt_transformers does
    tt_kv_cache = []
    if create_kv_cache:
        for layer in model.layers:
            # MiniMax-M2 uses self_attn instead of attention
            tt_kv_cache.append(layer.self_attn.layer_past)

    return model_args, model, tt_kv_cache, state_dict

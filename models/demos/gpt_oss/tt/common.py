# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific implementation of create_tt_model that's compatible with tt_transformers
"""

from loguru import logger

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
    GPT-OSS version of create_tt_model that matches tt_transformers interface
    Uses clean MeshConfig abstraction for optimal device parallelization
    """
    from models.demos.gpt_oss.config import MeshConfig
    from models.demos.gpt_oss.tt.model import Model
    from models.demos.gpt_oss.tt.model_config import ModelArgs

    # Use provided mesh_config or create optimal MeshConfig for the mesh shape
    if mesh_config is None:
        from models.demos.gpt_oss.config import ModeConfig

        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))

    # Create GPT-OSS ModelArgs
    gpt_oss_model_args = ModelArgs(
        mesh_device,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    # Override num_layers if provided (useful for quick testing with fewer layers)
    if num_layers is not None:
        gpt_oss_model_args.hf_config.num_hidden_layers = num_layers
        gpt_oss_model_args.n_layers = num_layers

    # Decide whether the HF weights are still needed on host. When the ttnn weight cache for
    # this (model, dtype, mesh shape) was already fully built on a previous run, ttnn.as_tensor
    # loads every weight from disk and the state_dict is never read -- so skip the expensive
    # from_pretrained host load entirely. This is what spares the e2e demo the prefill host-OOM
    # (#48509) on warm-cache runs, without relying on the manual --skip-model-load flag.
    #
    # state_dict is None  -> decide here (warm cache => {} skip, else cold load).
    # state_dict == {}     -> explicit skip (--skip-model-load) or a prior DP model already skipped.
    # state_dict populated -> reuse across DP models (avoid reloading for every submesh).
    loaded_real_weights = False
    if state_dict is None:
        if not gpt_oss_model_args.dummy_weights and gpt_oss_model_args.weight_cache_is_complete(dtype):
            logger.info("Warm ttnn weight cache detected -- skipping HF state_dict load.")
            state_dict = {}
        else:
            state_dict = gpt_oss_model_args.load_state_dict(
                weights_path=gpt_oss_model_args.model_path,
                dummy_weights=gpt_oss_model_args.dummy_weights,
                convert_to_meta_format=True,
            )
            loaded_real_weights = bool(state_dict) and not gpt_oss_model_args.dummy_weights

    # Create GPT-OSS model using transformer-compatible constructor
    model = Model.create_transformer_compatible(
        args=gpt_oss_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        tensor_cache_path=str(gpt_oss_model_args.weight_cache_path(dtype)),
        paged_attention_config=paged_attention_config,
        mesh_config=mesh_config,  # Pass explicit MeshConfig
        create_kv_cache=create_kv_cache,
        users_row_sharded=users_row_sharded,
        use_throughput_experts=use_throughput_experts,
    )

    # If this run populated the cache from a cold host load, record completion so future runs
    # can skip the load. Only for full-model builds (a num_layers override produces a partial
    # cache that must not satisfy the completeness check).
    if loaded_real_weights and num_layers is None:
        gpt_oss_model_args.mark_weight_cache_complete(dtype)

    # Extract tt_kv_cache like tt_transformers does
    tt_kv_cache = []
    if create_kv_cache:
        for layer in model.layers:
            # GPT-OSS uses self_attn instead of attention
            tt_kv_cache.append(layer.self_attn.layer_past)

    return gpt_oss_model_args, model, tt_kv_cache, state_dict

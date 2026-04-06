# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 model creation utility — matches tt_transformers interface.

Usage:
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device, max_batch_size=1, max_seq_len=8192,
    )
"""

import os

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


def create_tt_model(
    mesh_device,
    max_batch_size=1,
    max_seq_len=8192,
    dtype=ttnn.bfloat16,
    state_dict=None,
    num_layers=None,
    mesh_config=None,
    paged_attention_config=None,
    create_kv_cache=True,
    model_path=None,
):
    """
    Create Gemma4 model with all weights loaded to device.

    Returns:
        (model_args, model, tt_kv_cache, state_dict)
    """
    model_path = model_path or os.getenv("GEMMA4_MODEL_PATH", "/proj_sw/user_dev/gemma4/gemma-4-26B-A4B-it")

    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)
    # Store the real HF text config for RoPE creation (Gemma4TextRotaryEmbedding needs it)
    hf_text_config = getattr(hf_config, "text_config", hf_config)
    model_args._hf_text_config = hf_text_config

    if num_layers is not None:
        model_args.num_hidden_layers = num_layers

    if mesh_config is None:
        is_mesh = hasattr(mesh_device, "shape")
        if is_mesh:
            mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        else:
            mesh_config = MeshConfig((1, 1), decode=ModeConfig(tp=1))

    is_mesh = hasattr(mesh_device, "shape")
    num_devices = mesh_device.get_num_devices() if is_mesh else 1
    if is_mesh and num_devices > 1:
        num_links = 1 if num_devices <= 2 else 4
        ccl_manager = CCLManager(mesh_device, num_links=num_links)
    else:
        ccl_manager = None

    if state_dict is None:
        state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)

    tensor_cache_path = str(model_args.weight_cache_path(model_path, dtype))

    model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=dtype,
        tensor_cache_path=tensor_cache_path,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=max_batch_size,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        create_kv_cache=create_kv_cache,
    )

    return model_args, model, model.tt_kv_cache, state_dict

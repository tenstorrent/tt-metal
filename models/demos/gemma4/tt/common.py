# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
from models.demos.gemma4.tt.assistant.model import Gemma4AssistantModel
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs, Gemma4ModelArgs
from models.demos.gemma4.tt.precision import Gemma4Precision
from loguru import logger
from models.common.weight_cache import (
    build_cached_state_dict,
    mark_weight_cache_complete,
    weight_cache_is_complete,
)

# Weights gemma4 consumes on the HOST (not just via ttnn.as_tensor) and that therefore must be
# loaded for real even on a warm cache (see #45400 follow-up analysis of models/demos/gemma4/tt):
#  - token embedding: F.embedding(tokens, _embed_weight_cpu)      (model.py:1218/1238/1421)
#  - per-layer-input embed/proj/norm (E2B/E4B):                    (model.py:615-635)
#  - per-layer learned scalar read via .item():                   (layer.py:122-123)
# Everything else flows through ttnn.as_tensor(cache_file_name=...) and is placeholder-safe.
_GEMMA4_HOST_WEIGHT_SUFFIXES = (
    "embed_tokens.weight",
    "embed_tokens_per_layer.weight",
    "per_layer_model_projection.weight",
    "per_layer_projection_norm.weight",
    ".layer_scalar",
)


def _gemma4_is_host_weight(key):
    return any(key.endswith(s) for s in _GEMMA4_HOST_WEIGHT_SUFFIXES)


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
    bounded_sliding_kv_cache: bool = False,
):
    """
    Create Gemma4 model with all weights loaded to device.

    Returns:
        (model_args, model, tt_kv_cache, state_dict)
    """
    model_path = (
        model_path
        or os.getenv("HF_MODEL")
        or os.getenv("GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it")
    )

    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)
    model_args.model_cache_path = model_args.resolve_model_cache_path(model_path)
    # Store the real HF text config for RoPE creation (Gemma4TextRotaryEmbedding needs it)
    hf_text_config = getattr(hf_config, "text_config", hf_config)
    model_args._hf_text_config = hf_text_config

    if num_layers is not None:
        model_args.num_hidden_layers = num_layers

    if mesh_config is None:
        is_mesh = hasattr(mesh_device, "shape")
        num_devices = mesh_device.get_num_devices() if is_mesh else 1
        if is_mesh and num_devices > 1:
            mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        else:
            mesh_config = MeshConfig((1, 1), decode=ModeConfig(tp=1))

    is_mesh = hasattr(mesh_device, "shape")
    num_devices = mesh_device.get_num_devices() if is_mesh else 1
    if is_mesh and num_devices > 1:
        # num_links=None -> arch default (2 on Blackhole) so the per-layer TP
        # all-reduces (the dominant ~31% of prefill device time) use full
        # inter-device bandwidth.
        ccl_manager = CCLManager(mesh_device)
    else:
        ccl_manager = None

    # Warm ttnn cache => skip the full HF weight load and build from .tensorbin. Hybrid: the few
    # host-consumed weights (token embedding, per-layer scalars/PLI) are served real from the
    # sidecar, the rest as dataless placeholders. Generalizes PR #50550 to gemma4 (#45400).
    cache_dir = model_args.weight_cache_path(dtype)
    cache_identity = dict(
        model_name=os.path.basename(str(model_path).rstrip("/")) or "gemma4",
        n_layers=model_args.num_hidden_layers,
        mesh_shape=tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1),
    )
    loaded_real_weights = False
    if state_dict is None:
        if num_layers is None and weight_cache_is_complete(cache_dir, **cache_identity):
            logger.info("Warm ttnn weight cache detected -- skipping HF state_dict load (gemma4 hybrid).")
            state_dict = build_cached_state_dict(cache_dir)
        else:
            state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
            loaded_real_weights = bool(state_dict)

    tensor_cache_path = str(cache_dir)

    # Resolve per-module dtype overrides from precision_overrides.json. The
    # mesh shape is the worker grid (rows x cols); a 1x1 mesh on a multi-device
    # system still gets the 1x1 entry.
    mesh_shape = tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    precision = Gemma4Precision.load(model_path, mesh_shape)

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
        precision=precision,
        bounded_sliding_kv_cache=bounded_sliding_kv_cache,
    )

    # After a full cold build, record completion (+ capture host-consumed weights to the sidecar)
    # so future runs can skip the HF load.
    if loaded_real_weights and num_layers is None:
        mark_weight_cache_complete(cache_dir, state_dict, is_host_weight=_gemma4_is_host_weight, **cache_identity)

    return model_args, model, model.tt_kv_cache, state_dict


def create_assistant_model(
    mesh_device,
    target_model,
    mesh_config,
    ccl_manager,
    dtype=ttnn.bfloat16,
    assistant_path=None,
    state_dict=None,
    max_local_batch_size=1,
):
    """Create the Gemma4 it-assistant drafter, sharing the target's mesh/CCL.

    The drafter cross-attends into ``target_model``'s KV caches and reuses its
    RoPE caches + raw token embedding, so it must be built from the same target
    instance used for decoding.

    Returns:
        (assistant_args, assistant_model)
    """
    assistant_path = assistant_path or os.getenv("GEMMA4_ASSISTANT_MODEL")
    if not assistant_path:
        raise ValueError(
            "No assistant model path. Set GEMMA4_ASSISTANT_MODEL (e.g. google/gemma-4-31B-it-assistant) "
            "or pass assistant_path=."
        )

    hf_config = Gemma4AssistantArgs.load_hf_config(assistant_path)
    assistant_args = Gemma4AssistantArgs.from_hf_config(hf_config)
    assistant_args.model_cache_path = assistant_args.resolve_model_cache_path(assistant_path)

    if assistant_args.backbone_hidden_size != target_model.hidden_size:
        raise ValueError(
            f"Assistant backbone_hidden_size ({assistant_args.backbone_hidden_size}) != target hidden_size "
            f"({target_model.hidden_size}). The assistant must match its target model."
        )
    if getattr(target_model, "bounded_sliding_kv_cache", False):
        raise NotImplementedError(
            "Speculative decoding requires the target to use unbounded sliding KV caches "
            "(bounded_sliding_kv_cache=False); the drafter cross-attention reads absolute cache positions."
        )

    if state_dict is None:
        state_dict = Gemma4AssistantArgs.load_state_dict(assistant_path, dummy_weights=False)

    tensor_cache_path = str(assistant_args.weight_cache_path(dtype))

    model = Gemma4AssistantModel(
        mesh_device=mesh_device,
        assistant_args=assistant_args,
        target_model=target_model,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=dtype,
        tensor_cache_path=tensor_cache_path,
        mesh_config=mesh_config,
        max_local_batch_size=max_local_batch_size,
    )
    return assistant_args, model

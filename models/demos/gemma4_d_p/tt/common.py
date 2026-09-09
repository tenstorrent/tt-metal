# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Construct the standalone Gemma4 Galaxy prefill model."""

import os

from loguru import logger

import ttnn
from models.common.weight_cache import build_cached_state_dict, mark_weight_cache_complete, weight_cache_is_complete
from models.demos.gemma4_d_p.config import MeshConfig, validate_galaxy_mesh
from models.demos.gemma4_d_p.tt.ccl import CCLManager
from models.demos.gemma4_d_p.tt.model import Gemma4Model
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision

# Host weights required for embedding construction and learned layer scalars.
_GEMMA4_HOST_WEIGHT_SUFFIXES = (
    "embed_tokens.weight",
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
    model_path=None,
    prefill_chunk_size=None,
    ring_kv_caches=None,
    force_rebuild=False,
):
    """
    Create Gemma4 model with all weights loaded to device.

    Returns:
        (model_args, model, tt_kv_cache, state_dict)
    """
    mesh_config = mesh_config or MeshConfig(mesh_device.shape)
    validate_galaxy_mesh(mesh_device.shape)
    if tuple(mesh_device.shape) != mesh_config.mesh_shape:
        raise ValueError("mesh_config must match the device mesh")
    if prefill_chunk_size is None:
        prefill_chunk_size = min(8192, max_seq_len)
    if max_seq_len <= 0 or prefill_chunk_size <= 0:
        raise ValueError("sequence and chunk lengths must be positive")
    if prefill_chunk_size % (mesh_config.prefill.sp * ttnn.TILE_SIZE) or max_seq_len % prefill_chunk_size:
        raise ValueError("prefill chunks must divide max_seq_len and contain whole CP-local tiles")
    if prefill_chunk_size < 1024 * mesh_config.prefill.sp:
        raise ValueError("prefill chunk size must cover the sliding window on each CP rank")

    model_path = model_path or os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "google/gemma-4-31B-it")

    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)
    model_args.model_cache_path = model_args.resolve_model_cache_path(model_path)
    # Store the real HF text config for RoPE creation (Gemma4TextRotaryEmbedding needs it)
    hf_text_config = getattr(hf_config, "text_config", hf_config)
    model_args._hf_text_config = hf_text_config

    if num_layers is not None:
        model_args.num_hidden_layers = num_layers

    ccl_manager = CCLManager(mesh_device)

    # Warm ttnn cache => skip the full HF weight load and build from .tensorbin. Hybrid: the few
    # host-consumed weights (token embedding and layer scalars) are served real from the
    # sidecar, the rest as dataless placeholders. Generalizes PR #50550 to gemma4 (#45400).
    # Qualify the cache by mesh geometry BEFORE resolving cache_dir: ttnn.as_tensor
    # reloads tensorbins as-is and ignores mesh_mapper, so a TP=4 cache built on
    # MeshShape([2,4]) must not be reused on [1,4] (QB2). Setting cluster_shape here
    # keeps the warm-cache marker (cache_dir) and the tensorbin path on the same
    # directory instead of letting them diverge.
    _worker_mesh = tuple(mesh_device.shape)
    model_args.cluster_shape = _worker_mesh
    cache_dir = model_args.weight_cache_path(dtype)
    # Resolved early so it can key the cache identity: gemma4 embeds each module's dtype in its
    # tensorbin FILENAME (attention/shared_mlp *_{dtype} suffixes), so an edit to
    # precision_overrides.json changes which files a build needs. Without the precision in the
    # variant, a marker seeded under the old overrides would certify a warm build whose files do
    # not exist -- and as_tensor would persist placeholders for them. (#45400 review, finding B2)
    _precision_for_variant = Gemma4Precision.load(model_path, _worker_mesh)
    cache_identity = dict(
        model_name=os.path.basename(str(model_path).rstrip("/")) or "gemma4",
        n_layers=model_args.num_hidden_layers,
        mesh_shape=_worker_mesh,
        build_variant={
            "prefill_cache_layout": 1,
            "global_projection": "qk",
            "precision": {k: str(v) for k, v in sorted(_precision_for_variant._overrides.items())},
        },
    )
    loaded_real_weights = False
    if state_dict is None:
        if not force_rebuild and num_layers is None and weight_cache_is_complete(cache_dir, **cache_identity):
            logger.info("Warm ttnn weight cache detected -- skipping HF state_dict load (gemma4 hybrid).")
            state_dict = build_cached_state_dict(
                cache_dir, args=model_args, build_variant=cache_identity["build_variant"]
            )
        else:
            state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
            loaded_real_weights = bool(state_dict)

    tensor_cache_path = str(cache_dir)

    # Per-module dtype overrides from precision_overrides.json, resolved once
    # above so the cache identity and the model share one value.
    precision = _precision_for_variant

    model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=dtype,
        tensor_cache_path=tensor_cache_path,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        prefill_chunk_size=prefill_chunk_size,
        max_local_batch_size=max_batch_size,
        num_layers=num_layers,
        precision=precision,
        ring_kv_caches=ring_kv_caches,
    )

    # After a full cold build, record completion (+ capture host-consumed weights to the sidecar)
    # so future runs can skip the HF load.
    if loaded_real_weights and num_layers is None:
        mark_weight_cache_complete(cache_dir, state_dict, is_host_weight=_gemma4_is_host_weight, **cache_identity)

    return model_args, model, model.tt_kv_cache, state_dict

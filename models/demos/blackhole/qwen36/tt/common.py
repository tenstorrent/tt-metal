# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model construction entry point (gpt_oss/gemma4 `create_tt_model` convention)."""
import os

from loguru import logger

from models.common.weight_cache import build_cached_state_dict, mark_weight_cache_complete, weight_cache_is_complete
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def create_tt_model(
    mesh_device,
    max_batch_size=1,
    max_seq_len=2048,
    n_layers=None,
    layer_indices=None,
    hf_model=None,
):
    """Build the Qwen3.5-9B model. Returns (args, model, state_dict).

    HF_MODEL (env var) is the single source of truth. `hf_model`, if given, sets it.
    `layer_indices` runs ONLY the listed checkpoint layers (profiling); it takes precedence
    over `n_layers` (first-N truncation). See Qwen36Model.from_pretrained for details.
    """
    if hf_model is not None:
        os.environ["HF_MODEL"] = hf_model

    args = Qwen36ModelArgs(
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if layer_indices is not None:
        layer_indices = list(layer_indices)
        assert layer_indices, "layer_indices must be non-empty"
        assert all(
            0 <= i < len(args.attention_type_list) for i in layer_indices
        ), f"layer_indices {layer_indices} out of range [0, {len(args.attention_type_list)})"
        args.layer_indices = layer_indices
        args.n_layers = len(layer_indices)
    elif n_layers is not None:
        args.n_layers = n_layers
        args.attention_type_list = args.attention_type_list[:n_layers]

    # Warm ttnn cache => skip the HF from_pretrained load for the (text) weights; they build from
    # .tensorbin. Pure placeholder is safe: the vision tower uses a SEPARATE live HF reference model
    # (tt/vision/model.py), so it never sees this state_dict. Partial win -- the vision reference
    # load is not skipped (follow-up). Only for a full build (layer truncation => partial cache). (#45400)
    cache_path = args.weight_cache_path()
    full_build = n_layers is None and layer_indices is None
    cache_identity = dict(
        model_name=args.model_name,
        n_layers=args.n_layers,
        mesh_shape=tuple(args.mesh_device.shape),
    )
    loaded_real_weights = False
    if (
        full_build
        and not getattr(args, "dummy_weights", False)
        and weight_cache_is_complete(cache_path, **cache_identity)
    ):
        logger.info("Warm ttnn weight cache detected -- skipping HF state_dict load (text weights).")
        state_dict = build_cached_state_dict(cache_path)
    else:
        logger.info("Loading + remapping weights via Qwen36ModelArgs.load_state_dict()...")
        state_dict = args.load_state_dict()
        loaded_real_weights = bool(state_dict) and not getattr(args, "dummy_weights", False)

    model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=cache_path)

    if loaded_real_weights and full_build:
        mark_weight_cache_complete(cache_path, state_dict, **cache_identity)

    return args, model, state_dict

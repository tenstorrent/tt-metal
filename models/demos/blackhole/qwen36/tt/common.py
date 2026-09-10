# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model construction entry point (gpt_oss/gemma4 `create_tt_model` convention)."""
import os

from loguru import logger

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

    # NOTE: the warm-ttnn-cache HF-load skip is DISABLED for qwen3.6.
    # Its Gated-DeltaNet loader consumes conv weights on the host without a cache_file_name --
    # gdn/weights.py::load_conv_weight does ttnn.from_torch(state_dict[name], ...) for q/k/v_conv in
    # every DeltaNet layer, and gdn/tp.py derives taps the same way -- so a dataless placeholder
    # feeds those layers garbage while the HF load is skipped. This is the same failure that made
    # the vision demo emit token soup, on the text path. Re-enabling needs those conv weights either
    # cache-backed or captured to the sidecar via an is_host_weight predicate. (#45400 review)
    cache_path = args.weight_cache_path()
    logger.info("Loading + remapping weights via Qwen36ModelArgs.load_state_dict()...")
    state_dict = args.load_state_dict()

    model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=cache_path)

    return args, model, state_dict

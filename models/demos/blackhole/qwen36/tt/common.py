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
    hf_model=None,
):
    """Build the Qwen3.5-9B model. Returns (args, model, state_dict).

    HF_MODEL (env var) is the single source of truth. `hf_model`, if given, sets it.
    """
    if hf_model is not None:
        os.environ["HF_MODEL"] = hf_model

    args = Qwen36ModelArgs(
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if n_layers is not None:
        args.n_layers = n_layers
        args.attention_type_list = args.attention_type_list[:n_layers]

    logger.info("Loading + remapping weights via Qwen36ModelArgs.load_state_dict()...")
    state_dict = args.load_state_dict()
    cache_path = args.weight_cache_path()
    model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=cache_path)
    return args, model, state_dict

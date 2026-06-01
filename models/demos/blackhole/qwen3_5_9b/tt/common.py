# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model construction entry point (gpt_oss/gemma4 `create_tt_model` convention)."""
import os

from loguru import logger

from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model


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

    args = Qwen35ModelArgs(
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if n_layers is not None:
        args.n_layers = n_layers
        args.attention_type_list = args.attention_type_list[:n_layers]

    logger.info("Loading + remapping weights via Qwen35ModelArgs.load_state_dict()...")
    state_dict = args.load_state_dict()
    cache_path = args.weight_cache_path()
    model = Qwen35Model(args, state_dict, mesh_device, weight_cache_path=cache_path)
    return args, model, state_dict

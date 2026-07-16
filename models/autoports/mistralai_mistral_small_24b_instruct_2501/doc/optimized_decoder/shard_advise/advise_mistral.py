# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ttnn-advise capture target for the rewritten Mistral dense decoder block."""

from __future__ import annotations

import os
import sys

import torch

import ttnn  # Import the advisor environment's package before appending tt-metal.

MODEL_ROOT = os.environ.get("SHARD_ADVISE_MODEL_ROOT", "/home/mvasiljevic/tt-metal")
HF_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
LAYER_IDX = 20
BATCH = 32
MAX_CACHE_LEN = 128


def _build(device):
    if MODEL_ROOT not in sys.path:
        sys.path.append(MODEL_ROOT)

    from transformers import AutoConfig

    from models.autoports.mistralai_mistral_small_24b_instruct_2501.tests.test_functional_decoder import (
        _synthetic_state,
    )
    from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.optimized_decoder import OptimizedDecoder

    config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    state = _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=MAX_CACHE_LEN,
    )

    hidden = ttnn.from_torch(
        torch.zeros((1, BATCH, 1, config.hidden_size), dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cache_shape = (BATCH, config.num_key_value_heads, MAX_CACHE_LEN, config.head_dim)
    key_cache = ttnn.from_torch(
        torch.zeros(cache_shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    value_cache = ttnn.from_torch(
        torch.zeros(cache_shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return decoder, hidden, key_cache, value_cache


_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None


def decode(hidden):
    return _DECODER.decode_forward(hidden, _KEY_CACHE, _VALUE_CACHE, current_pos=18)


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE
    _DECODER, hidden, _KEY_CACHE, _VALUE_CACHE = _build(device)
    return (hidden,)

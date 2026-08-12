# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor target for one rewritten TP4 Mistral rank-local dense block.

The advisor does not model mesh CCLs.  This capture therefore presents the
exact per-rank dense shapes to the compiler: full hidden residual, 8 local Q
heads, 2 local KV heads, and an 8192-wide local MLP.  The leading/trailing
public-layout reshape in OptimizedDecoder is ignored when applying the report;
the optimized multichip runtime keeps the rewritten internal residual layout
between layers.
"""

from __future__ import annotations

import copy
import os
import sys

import torch

import ttnn  # Import the advisor environment's package before appending tt-metal.

MODEL_ROOT = os.environ.get("SHARD_ADVISE_MODEL_ROOT", "/home/mvasiljevic/tt-metal")
HF_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
LAYER_IDX = 20
BATCH = 32
MAX_CACHE_LEN = 128
LOCAL_Q_HEADS = 8
LOCAL_KV_HEADS = 2
LOCAL_INTERMEDIATE = 8192


def _build(device):
    if MODEL_ROOT not in sys.path:
        sys.path.append(MODEL_ROOT)

    from transformers import AutoConfig

    from models.autoports.mistralai_mistral_small_24b_instruct_2501.tests.test_functional_decoder import (
        _synthetic_state,
    )
    from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.optimized_decoder import OptimizedDecoder

    global_config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    config = copy.deepcopy(global_config)
    config.num_attention_heads = LOCAL_Q_HEADS
    config.num_key_value_heads = LOCAL_KV_HEADS
    config.intermediate_size = LOCAL_INTERMEDIATE
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
    cache_shape = (BATCH, LOCAL_KV_HEADS, MAX_CACHE_LEN, config.head_dim)
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

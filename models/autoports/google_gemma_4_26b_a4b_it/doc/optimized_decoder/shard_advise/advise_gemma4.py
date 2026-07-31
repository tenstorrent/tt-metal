# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for Gemma-4's dense attention + dense MLP.

The routed expert block is intentionally outside this capture because TTIR has
no ``sparse_matmul`` operation.  The captured function is the real optimized
decoder's dense decode subgraph at the real hidden/head/intermediate shapes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = Path(os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")).resolve()
if str(TT_METAL_ROOT) not in sys.path:
    # Append: prepending tt-metal would shadow the advisor environment's ttnn.
    sys.path.append(str(TT_METAL_ROOT))

from models.autoports.google_gemma_4_26b_a4b_it.tests.synthetic_weights import synthetic_layer_state_dict
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    FULL_BLOCK_SIZE,
    FULL_HEAD_DIM,
    FULL_NUM_KV_HEADS,
    HIDDEN_SIZE,
    SLIDING_BLOCK_SIZE,
    SLIDING_HEAD_DIM,
    SLIDING_NUM_KV_HEADS,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

LAYER_IDX = int(os.environ.get("GEMMA4_ADVISE_LAYER", "0"))
BATCH = int(os.environ.get("GEMMA4_ADVISE_BATCH", "1"))

_DECODER = None
_KWARGS = None


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _as_tt(device, value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.as_tensor(
        value,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build(device):
    config_path = TT_METAL_ROOT / "models/demos/gemma4/configs/gemma-4-26B-A4B-it/config.json"
    cfg = _namespace(json.loads(config_path.read_text())["text_config"])
    layer_type = cfg.layer_types[LAYER_IDX]
    # The advisor executes setup before interception, so real CPU storage is
    # required. Values are deterministic synthetic samples at canonical shapes;
    # layout advice depends on shapes/dtypes, not checkpoint values.
    state = synthetic_layer_state_dict(LAYER_IDX)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=LAYER_IDX,
        mesh_device=device,
    )

    torch.manual_seed(20260729 + BATCH)
    hidden = torch.randn(1, BATCH, HIDDEN_SIZE, dtype=torch.bfloat16)
    head_dim = SLIDING_HEAD_DIM if layer_type == "sliding_attention" else FULL_HEAD_DIM
    cos = torch.ones(1, 1, BATCH, head_dim, dtype=torch.bfloat16)
    sin = torch.zeros(1, 1, BATCH, head_dim, dtype=torch.bfloat16)
    shared_physical = layer_type == "sliding_attention"
    if shared_physical:
        cache_shape = (4, SLIDING_NUM_KV_HEADS, SLIDING_BLOCK_SIZE, SLIDING_HEAD_DIM)
        blocks = 4
    else:
        cache_shape = (2, FULL_NUM_KV_HEADS, FULL_BLOCK_SIZE, FULL_HEAD_DIM)
        blocks = 2
    kwargs = {
        "position_cos": _as_tt(device, cos),
        "position_sin": _as_tt(device, sin),
        "current_pos": _as_tt(
            device,
            torch.full((BATCH,), 32, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": _as_tt(
            device,
            torch.arange(blocks, dtype=torch.int32).view(1, blocks).expand(BATCH, -1).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "kv_cache": (
            _as_tt(device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
            _as_tt(device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        ),
        "cache_position_modulo": None,
    }
    return decoder, _as_tt(device, hidden.unsqueeze(1)), kwargs


def decode(hidden):
    """Trace the optimized decoder's dense attention and dense MLP subgraph."""
    attn_in = _DECODER._rms_norm(hidden, _DECODER.weights.input_ln)
    attn_out = _DECODER._attention_decode(attn_in, **_KWARGS)
    dense_in = _DECODER._rms_norm(attn_out, _DECODER.weights.pre_ff_ln)
    return _DECODER._dense_mlp(dense_in)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, hidden, _KWARGS = _build(device)
    return (hidden,)

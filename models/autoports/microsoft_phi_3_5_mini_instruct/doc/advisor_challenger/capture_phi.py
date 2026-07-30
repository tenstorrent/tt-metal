# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 shard-advisor capture of the shipped Phi-3.5 optimized decoder."""

from __future__ import annotations

import sys

import torch

import ttnn

ROOT = "/home/mvasiljevic/tt-metal"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _page_table,
    _positions,
    _synthetic_state,
    _to_tt_decode,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
    OptimizationPolicy,
    OptimizedDecoder,
)

BATCH = 32
_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None
_PAGE_TABLE = None
_POSITIONS = None


class CaptureDecoder(OptimizedDecoder):
    """Tracer-safe equivalent of the shipped decode RoPE path.

    The runtime implementation queries the q/k tensors' assigned memory config.
    During advisor analysis that placement is intentionally unknown. The qkv
    head-split explicitly requests L1 height sharding, so restore that declared
    phase config directly after the DRAM rotate-half sequence.
    """

    def _decode_rope(self, query, key, current_positions, *, use_long_rope):
        cos_table = self.long_cos if use_long_rope else self.short_cos
        sin_table = self.long_sin if use_long_rope else self.short_sin
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, 1, self.batch, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, self.batch, self.head_dim])
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        return (
            ttnn.to_memory_config(query, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
            ttnn.to_memory_config(key, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
        )


def _build(device):
    config = _config()
    policy = OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
    )
    decoder = CaptureDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_context=128,
        optimization_policy=policy,
    )
    hidden = torch.randn(
        BATCH,
        1,
        config.hidden_size,
        generator=torch.Generator().manual_seed(20260730),
    ).to(torch.bfloat16)
    return (
        decoder,
        _to_tt_decode(hidden, device),
        decoder.create_paged_kv_cache(),
        _page_table(BATCH, 128, device, permute=True),
        _positions([0] * BATCH, device),
    )


def decode(hidden):
    return _DECODER.decode_forward(
        hidden,
        key_cache=_KEY_CACHE,
        value_cache=_VALUE_CACHE,
        page_table=_PAGE_TABLE,
        current_positions=_POSITIONS,
        use_long_rope=False,
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE, _PAGE_TABLE, _POSITIONS
    _DECODER, hidden, caches, _PAGE_TABLE, _POSITIONS = _build(device)
    _KEY_CACHE, _VALUE_CACHE = caches
    return (hidden,)

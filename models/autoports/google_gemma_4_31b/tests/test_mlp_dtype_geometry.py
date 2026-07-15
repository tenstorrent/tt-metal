# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import replace
from pathlib import Path

from models.autoports.google_gemma_4_31b.tt.multichip_decoder import (
    MLP_BFP8_PACKED_GATE_UP_BLOCK_W_MAX,
    _TPOptimizedSharedMLP,
)
from models.autoports.google_gemma_4_31b.tt.precision import load_precision_config

MODEL_DIR = Path("models/autoports/google_gemma_4_31b")
CONFIG_DIR = MODEL_DIR / "doc/datatype_sweep/configs"
BLACKHOLE_L1_BYTES = 1_572_864
BFP4_TILE_BYTES_ALIGNED = 576
BFP8_TILE_BYTES_ALIGNED = 1_088
BF16_TILE_BYTES = 2_048
PACKED_GATE_UP_N_TILES_PER_DRAM_READER = 42
WEIGHT_CB_BUFFERING_DEPTH = 3
FAILING_STATIC_CB_BYTES = 1_937_280
STATIC_CB_BASE_BYTES = 111_360


def _effective_block_width(config_name: str) -> int:
    resolved = load_precision_config(CONFIG_DIR / config_name)
    mlp = object.__new__(_TPOptimizedSharedMLP)
    # Match MultichipDecoder.from_state_dict's model-specific TP4 geometry.
    mlp.policy = replace(
        resolved.default_decoder_policy,
        decode_num_cores=14,
        gate_up_in0_block_w=12,
        down_in0_block_w=12,
    )
    return mlp.packed_gate_up_in0_block_w


def _packed_gate_up_static_cb_bytes(block_width: int, weight_tile_bytes: int) -> int:
    return (
        STATIC_CB_BASE_BYTES
        + block_width * 2 * BF16_TILE_BYTES
        + PACKED_GATE_UP_N_TILES_PER_DRAM_READER * block_width * WEIGHT_CB_BUFFERING_DEPTH * weight_tile_bytes
        + PACKED_GATE_UP_N_TILES_PER_DRAM_READER * BFP8_TILE_BYTES_ALIGNED
        + PACKED_GATE_UP_N_TILES_PER_DRAM_READER * BF16_TILE_BYTES
    )


def test_bfp8_packed_gate_up_uses_largest_l1_safe_k_block():
    baseline_width = _effective_block_width("baseline_bfp8attn_bfp4mlp_lofi_bf16lm.json")
    bfp8_width = _effective_block_width("mlp_bfp8_lofi.json")

    assert baseline_width == 12
    assert bfp8_width == MLP_BFP8_PACKED_GATE_UP_BLOCK_W_MAX == 6
    assert max(divisor for divisor in range(1, baseline_width) if baseline_width % divisor == 0) == bfp8_width

    failing_weight_cb = (
        PACKED_GATE_UP_N_TILES_PER_DRAM_READER * baseline_width * WEIGHT_CB_BUFFERING_DEPTH * BFP8_TILE_BYTES_ALIGNED
    )
    assert failing_weight_cb == 1_645_056 > BLACKHOLE_L1_BYTES

    assert _packed_gate_up_static_cb_bytes(baseline_width, BFP8_TILE_BYTES_ALIGNED) == FAILING_STATIC_CB_BYTES
    assert _packed_gate_up_static_cb_bytes(baseline_width, BFP4_TILE_BYTES_ALIGNED) == 1_163_136
    assert _packed_gate_up_static_cb_bytes(bfp8_width, BFP8_TILE_BYTES_ALIGNED) == 1_090_176
    assert _packed_gate_up_static_cb_bytes(bfp8_width, BFP8_TILE_BYTES_ALIGNED) < BLACKHOLE_L1_BYTES


def test_dtype_cap_is_consumed_only_by_packed_gate_up_program():
    source = inspect.getsource(_TPOptimizedSharedMLP.__call__)
    packed_branch = source.index('if self.policy.mlp_gate_up_topology == "packed"')
    packed_program = source.index("packed_program = self._decode_program_config(", packed_branch)
    capped_width = source.index("in0_block_w=self.packed_gate_up_in0_block_w", packed_program)
    packed_linear = source.index("packed_sharded = ttnn.linear(", capped_width)
    assert packed_program < capped_width < packed_linear

    # Separate gate/up and down geometry retain their existing policy values.
    assert "in0_block_w=self.policy.gate_up_in0_block_w" in source[:packed_branch]
    assert "in0_block_w=self.policy.down_in0_block_w" in source

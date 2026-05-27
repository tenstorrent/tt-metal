# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for SRAM hot expert allocation infrastructure.

Covers:
  - ``SramCompressedExpertSlots`` dataclass + ``is_dram_flags`` routing helper.
  - ``compute_expert_l1_bytes`` predictor (assigner + uniform + BSPM paths).
  - ``build_sram_hot_expert_config`` routing-frequency ranking.

Multi-device end-to-end coverage of ``prepare_compressed_sram_slots`` lives in
``test_decoder_block.py`` (TP=8).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensorAssigner
from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp
from models.demos.deepseek_v3_b1.weights.sram_slots import SramCompressedExpertSlots
from models.demos.deepseek_v3_b1.weights.transforms.sram_experts import (
    SramExpertCoreGrids,
    build_sram_hot_expert_config,
    compute_expert_l1_bytes,
)

TILE_W = 32


def _make_assigner(formats=None):
    if formats is None:
        formats = ["bfp8", "bfp4"]
    return CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=formats)


def test_sram_slot_structure():
    """SramCompressedExpertSlots fields and is_dram_flags helper."""
    slots = SramCompressedExpertSlots(
        num_slots=3,
        slot_experts=[10, 42, 100],
        gate_proj=[None, None, None],  # type: ignore
        up_proj=[None, None, None],  # type: ignore
        down_proj=[None, None, None],  # type: ignore
    )
    assert slots.num_slots == 3
    assert slots.slot_experts == [10, 42, 100]

    flags = slots.is_dram_flags(256)
    assert len(flags) == 256
    assert flags[10] == 0
    assert flags[42] == 0
    assert flags[100] == 0
    assert flags[0] == 1
    assert flags[255] == 1
    assert sum(f == 0 for f in flags) == 3


K = 256
N = 256


def test_compute_expert_l1_bytes():
    """compute_expert_l1_bytes returns the max per-core sum across projections.

    Uses a mixed-precision assignment from CompressedTensorAssigner to verify
    the tighter computation (sum-per-core, then max) is <= the conservative
    estimate (sum of independent per-projection maxes).
    """
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    core_grids = SramExpertCoreGrids.uniform(core_grid)
    num_cores = 8
    assigner = _make_assigner(formats=["bfp8", "bfp4", "bfp2"])

    def _quantize_fn(x, fmt_str):
        mant_map = {"bfp8": 7, "bfp4": 3, "bfp2": 1, "bfp0": 0}
        return quantize_dequantize_bfp(x, mant_map[fmt_str])

    torch.manual_seed(123)
    weights = [torch.randn(K, N), torch.randn(K, N), torch.randn(K, N)]
    assignments = []
    shapes = []
    for w in weights:
        result = assigner.assign(w, _quantize_fn)
        assignments.append(result.assignment)
        shapes.append(tuple(w.shape))

    computed = compute_expert_l1_bytes(assignments, shapes, core_grids)
    assert computed > 0, "L1 bytes must be positive"

    # Conservative estimate: sum of independent per-projection maxes
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import compute_shard_page_mapping
    from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import bfp_tile_packed_size

    _MANT_BITS = {0: 7, 1: 3, 2: 1, 3: 0}
    tile_byte_lut = np.array(
        [bfp_tile_packed_size(_MANT_BITS[c], TILE_W) for c in range(4)],
        dtype=np.int64,
    )

    conservative = 0
    for assignment, (Kd, Nd) in zip(assignments, shapes):
        per_core_Nd = Nd // num_cores
        shard_spec = ttnn.ShardSpec(core_grid, [Kd, per_core_Nd], ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
        shard_mapping = compute_shard_page_mapping([Kd, Nd], mem_config, TILE_W)
        flat = assignment.ravel()
        proj_max = 0
        for _core, page_indices in shard_mapping:
            shard_bytes = int(tile_byte_lut[flat[list(page_indices)]].sum())
            proj_max = max(proj_max, shard_bytes)
        conservative += proj_max

    assert computed <= conservative, f"Tightened {computed} must be <= conservative {conservative}"

    # Determinism
    computed2 = compute_expert_l1_bytes(assignments, shapes, core_grids)
    assert computed == computed2, "compute_expert_l1_bytes must be deterministic"

    logger.info(f"compute_expert_l1_bytes: {computed} bytes (conservative: {conservative})")


def test_compute_expert_l1_bytes_uniform_assignment():
    """All-bfp8 assignment yields uniform shard sizes across cores."""
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    core_grids = SramExpertCoreGrids.uniform(core_grid)
    tiles_h = K // TILE_W
    tiles_w = N // TILE_W
    uniform_bfp8 = np.zeros((tiles_h, tiles_w), dtype=np.int32)

    assignments = [uniform_bfp8, uniform_bfp8, uniform_bfp8]
    shapes = [(K, N), (K, N), (K, N)]

    computed = compute_expert_l1_bytes(assignments, shapes, core_grids)

    from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import bfp_tile_packed_size

    expected_per_tile = bfp_tile_packed_size(7, TILE_W)
    tiles_per_shard = tiles_h * (tiles_w // 8)
    expected = 3 * tiles_per_shard * expected_per_tile

    assert computed == expected, f"Uniform bfp8: expected {expected}, got {computed}"
    logger.info(f"Uniform bfp8: {computed} bytes")


def test_build_sram_hot_expert_config_ranks_by_frequency():
    """build_sram_hot_expert_config sorts candidates by descending frequency
    and drops zero-frequency experts.
    """
    layer_idx = 3
    freqs = [0] * 256
    # Intentionally out-of-order: expert 5 most frequent, then 2, then 0.
    freqs[0] = 10
    freqs[2] = 30
    freqs[5] = 50
    # Zero-frequency experts must be excluded regardless of index.
    freqs[7] = 0

    config = build_sram_hot_expert_config([layer_idx], {layer_idx: freqs})

    assert config[layer_idx] == [5, 2, 0]


def test_build_sram_hot_expert_config_skips_layer_without_frequencies():
    """Layers missing from routing_frequencies are dropped from the config."""
    config = build_sram_hot_expert_config([3, 5], {3: [0] * 256})
    assert 3 not in config  # all-zero freqs -> no candidates
    assert 5 not in config  # no frequencies for layer 5


def test_build_sram_hot_expert_config_multi_layer():
    """Multi-layer config produces independent rankings per layer."""
    layers = [3, 5]
    experts_per_layer = {3: [0, 1, 2], 5: [4, 5]}
    routing_frequencies = {}
    for li in layers:
        freqs = [0] * 256
        for i, e in enumerate(experts_per_layer[li]):
            freqs[e] = 100 - i * 10
        routing_frequencies[li] = freqs

    config = build_sram_hot_expert_config(layers, routing_frequencies)

    assert 3 in config and 5 in config
    assert config[3] == experts_per_layer[3]
    assert config[5] == experts_per_layer[5]
    logger.info(f"Multi-layer config: {config}")


# DeepSeek gate/up and down projection shapes (K, N) — HF state dict stores
# (out_features, in_features); .T gives compute-layout (K, N).
_GATE_UP_SHAPE = (7168, 2048)
_DOWN_SHAPE = (2048, 7168)
_PROJ_SHAPES = [_GATE_UP_SHAPE, _GATE_UP_SHAPE, _DOWN_SHAPE]  # indexed by proj_idx


def _get_bspm_path() -> Path:
    """Return path to layer-4 BSPM file, or pytest.skip if unavailable."""
    bspm_dir = os.environ.get("BSPM_RESULTS_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_RESULTS_DIR not set")
    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")
    return bspm_path


def test_compute_expert_l1_bytes_bspm():
    """compute_expert_l1_bytes with real BSPM assignments at production shapes.

    Loads all three projection assignments from the layer-4 3.5 b/e BSPM file
    and verifies the L1 footprint is positive and strictly smaller than all-bfp8
    (since bfp4 tiles are 576 bytes vs bfp8's 1088 bytes).
    Does not require a device — pure host-side computation.
    Requires BSPM_RESULTS_DIR.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert
    from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import bfp_tile_packed_size

    bspm_path = _get_bspm_path()

    assignments = []
    shapes = []
    for proj_idx, (K, N) in enumerate(_PROJ_SHAPES):
        asgn = load_bspm_for_expert(
            str(bspm_path), expert_idx=0, proj_idx=proj_idx, tile_rows=K // TILE_W, tile_cols=N // TILE_W
        )
        assignments.append(asgn)
        shapes.append((K, N))

    # Use simple rectangular grids — compute_expert_l1_bytes is device-agnostic.
    # gate/up: 16 cores along a row; down: 28 cores = 13-core row + 15-core row.
    gate_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(15, 0))])
    down_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(14, 1)),
        ]
    )
    core_grids = SramExpertCoreGrids(gate=gate_grid, up=gate_grid, down=down_grid)

    computed = compute_expert_l1_bytes(assignments, shapes, core_grids)
    assert computed > 0, "L1 bytes must be positive for a non-empty BSPM assignment"

    # All-bfp8 upper bound: every tile at bfp8 cost.
    bfp8_bytes = bfp_tile_packed_size(7, TILE_W)
    bfp4_bytes = bfp_tile_packed_size(3, TILE_W)
    all_bfp8_cost = sum((K // TILE_W) * (N // TILE_W) * bfp8_bytes for K, N in shapes)
    assert (
        computed < all_bfp8_cost
    ), f"BSPM (bfp4-dominant) L1 cost {computed} should be < all-bfp8 cost {all_bfp8_cost}"

    # Sanity: bfp4 lower bound — all tiles at bfp4.
    all_bfp4_cost = sum((K // TILE_W) * (N // TILE_W) * bfp4_bytes for K, N in shapes)
    assert computed <= all_bfp4_cost, f"BSPM cost {computed} must be <= all-bfp4 {all_bfp4_cost} (no bfp8 tiles)"

    logger.info(
        f"BSPM expert L1 cost: {computed:,} bytes " f"(all-bfp8: {all_bfp8_cost:,}, all-bfp4: {all_bfp4_cost:,})"
    )

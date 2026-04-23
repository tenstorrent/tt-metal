# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for SramCompressedExpertSlots — SRAM hot expert slots using CompressedTensor.

Validates:
  - Dataclass structure and is_dram_flags routing helper
  - Single-device slot preparation with per-core L1 allocation
  - ExpertKernel fmt metadata tensor creation from slots
  - Expert index encoding and selection meta for SRAM/DRAM routing
  - BSPM real-assignment path: 0 bfp8 tiles, real {bfp4, bfp2, zero} assignment
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp
from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import create_expert_fmt_tensors, encode_expert_indices
from models.demos.deepseek_v3_b1.weights.prepare import (
    SramCompressedExpertSlots,
    SramExpertCoreGrids,
    SramHotExpertConfig,
    _build_l1_compressed_tensor,
    build_sram_hot_expert_config,
    compute_expert_l1_bytes,
    prepare_compressed_sram_slots,
)

TILE_W = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_dict(layer_idx: int, expert_indices: list[int], K: int = 256, N: int = 256):
    """Synthetic state dict with expert weights in HuggingFace layout (out_features, in_features)."""
    sd = {}
    for e in expert_indices:
        torch.manual_seed(e * 1000 + layer_idx)
        sd[f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(N, K)
        sd[f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(N, K)
        sd[f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(K, N)
    return sd


def _build_sram_core_grid(device, num_cores: int) -> ttnn.CoreRangeSet:
    """Pick *num_cores* compute cores from the device, skipping known DRAM workers."""
    grid = device.compute_with_storage_grid_size()
    dram_workers = {(0, 0), (0, 3), (0, 7), (0, 9), (7, 1), (7, 4), (7, 6), (7, 9)}
    cores = []
    for y in range(grid.y):
        for x in range(grid.x):
            if (x, y) in dram_workers:
                continue
            cores.append(ttnn.CoreCoord(x, y))
            if len(cores) == num_cores:
                break
        if len(cores) == num_cores:
            break
    assert len(cores) >= num_cores, f"Need {num_cores} non-DRAM cores, got {len(cores)}"
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _make_assigner(formats=None):
    if formats is None:
        formats = ["bfp8", "bfp4"]
    return CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=formats)


# ---------------------------------------------------------------------------
# Dataclass / host-only tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# On-device slot preparation
# ---------------------------------------------------------------------------

K = 256
N = 256
NUM_CORES = N // TILE_W  # 8 — each core holds (K, 32) shard


def test_build_l1_compressed_tensor(device):
    """_build_l1_compressed_tensor creates a per-core L1 CT with valid addresses."""
    core_grid = _build_sram_core_grid(device, NUM_CORES)
    assigner = _make_assigner()

    torch.manual_seed(42)
    weight = torch.randn(K, N).float()

    ct = _build_l1_compressed_tensor(weight, core_grid, assigner=assigner, device=device)
    assert isinstance(ct, CompressedTensor)

    cores = ttnn.corerange_to_cores(core_grid)
    seen_addrs = set()
    for c in cores:
        addr = ct.get_data_l1_address_per_core(c)
        assert addr > 0, f"core ({c.x},{c.y}) has invalid L1 address"
        seen_addrs.add(addr)
    logger.info(f"L1 CT allocated on {len(cores)} cores, {len(seen_addrs)} unique addresses")


def test_prepare_compressed_sram_slots(device):
    """Full prepare_compressed_sram_slots pipeline on a single device."""
    layer_idx = 3
    expert_indices = [0, 5, 10]

    sd = _make_state_dict(layer_idx, expert_indices, K=K, N=N)
    core_grid = _build_sram_core_grid(device, NUM_CORES)
    assigner = _make_assigner()

    slots = prepare_compressed_sram_slots(
        device=device,
        state_dict=sd,
        layer_idx=layer_idx,
        initial_expert_indices=expert_indices,
        core_grids=SramExpertCoreGrids.uniform(core_grid),
        assigner=assigner,
        move_to_device=True,
    )

    assert slots.num_slots == 3
    assert slots.slot_experts == [0, 5, 10]
    assert len(slots.gate_proj) == 3
    assert len(slots.up_proj) == 3
    assert len(slots.down_proj) == 3

    cores = ttnn.corerange_to_cores(core_grid)
    for proj_name, cts in [("gate", slots.gate_proj), ("up", slots.up_proj), ("down", slots.down_proj)]:
        for slot_idx, ct in enumerate(cts):
            assert isinstance(ct, CompressedTensor), f"{proj_name}[{slot_idx}] is not CompressedTensor"
            for c in cores:
                addr = ct.get_data_l1_address_per_core(c)
                assert addr > 0, f"{proj_name}[{slot_idx}] core ({c.x},{c.y}) invalid L1 addr"
        logger.info(f"  {proj_name}: {len(cts)} slots, all cores have valid L1 addresses")


# ---------------------------------------------------------------------------
# ExpertKernel metadata integration
# ---------------------------------------------------------------------------


def test_sram_slot_fmt_tensors(device):
    """Slots produce valid fmt metadata tensors for ExpertKernel."""
    layer_idx = 3
    expert_indices = [0, 5]

    sd = _make_state_dict(layer_idx, expert_indices, K=K, N=N)
    core_grid = _build_sram_core_grid(device, NUM_CORES)
    assigner = _make_assigner()

    slots = prepare_compressed_sram_slots(
        device=device,
        state_dict=sd,
        layer_idx=layer_idx,
        initial_expert_indices=expert_indices,
        core_grids=SramExpertCoreGrids.uniform(core_grid),
        assigner=assigner,
        move_to_device=True,
    )

    tiles_per_core = (K // TILE_W) * (N // NUM_CORES // TILE_W)
    fmt_tensors = create_expert_fmt_tensors(slots.gate_proj, device, core_grid, tiles_per_core)

    assert len(fmt_tensors) > 0
    for coord, core_tensors in fmt_tensors.items():
        assert len(core_tensors) == NUM_CORES, f"Expected {NUM_CORES} core entries, got {len(core_tensors)}"
        for idx, t in core_tensors.items():
            assert ttnn.is_tensor_storage_on_device(t), f"fmt tensor core {idx} not on device"
    logger.info(f"fmt_tensors created for {len(fmt_tensors)} device coords, {NUM_CORES} cores each")


def test_sram_slot_selection_meta(device):
    """Slots produce valid expert selection metadata for ExpertKernel routing."""
    expert_indices = [2, 7]
    num_total = 16

    slots = SramCompressedExpertSlots(
        num_slots=2,
        slot_experts=expert_indices,
        gate_proj=[None, None],  # type: ignore
        up_proj=[None, None],  # type: ignore
        down_proj=[None, None],  # type: ignore
    )
    flags = slots.is_dram_flags(num_total)
    assert flags[2] == 0
    assert flags[7] == 0
    assert all(flags[i] == 1 for i in range(num_total) if i not in {2, 7})

    encoded = encode_expert_indices([2, 7, 0, 15], flags)
    assert encoded[0] == 0x8000 | 2  # SRAM expert 2
    assert encoded[1] == 0x8000 | 7  # SRAM expert 7
    assert encoded[2] == 0  # DRAM expert 0
    assert encoded[3] == 15  # DRAM expert 15


# ---------------------------------------------------------------------------
# Per-layer SramHotExpertConfig
# ---------------------------------------------------------------------------


def test_sram_hot_expert_config_per_layer(device):
    """SramHotExpertConfig routes different experts to different layers."""
    config: SramHotExpertConfig = {
        3: [0, 5],
        7: [10, 20, 30],
    }

    core_grid = _build_sram_core_grid(device, NUM_CORES)
    assigner = _make_assigner()

    uniform_grids = SramExpertCoreGrids.uniform(core_grid)

    # Layer 3: 2 slots
    sd3 = _make_state_dict(3, config[3], K=K, N=N)
    slots3 = prepare_compressed_sram_slots(
        device=device,
        state_dict=sd3,
        layer_idx=3,
        initial_expert_indices=config[3],
        core_grids=uniform_grids,
        assigner=assigner,
        move_to_device=True,
    )
    assert slots3.num_slots == 2
    assert slots3.slot_experts == [0, 5]

    # Layer 7: 3 slots
    sd7 = _make_state_dict(7, config[7], K=K, N=N)
    slots7 = prepare_compressed_sram_slots(
        device=device,
        state_dict=sd7,
        layer_idx=7,
        initial_expert_indices=config[7],
        core_grids=uniform_grids,
        assigner=assigner,
        move_to_device=True,
    )
    assert slots7.num_slots == 3
    assert slots7.slot_experts == [10, 20, 30]

    # Layer 5: not in config → no slots
    assert config.get(5) is None

    # Routing flags are independent per layer
    flags3 = slots3.is_dram_flags(256)
    flags7 = slots7.is_dram_flags(256)
    assert flags3[0] == 0 and flags3[5] == 0 and flags3[10] == 1
    assert flags7[10] == 0 and flags7[20] == 0 and flags7[30] == 0 and flags7[0] == 1
    logger.info("Per-layer config: layer 3 has 2 slots, layer 7 has 3 slots, layer 5 has none")


# ---------------------------------------------------------------------------
# compute_expert_l1_bytes tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# build_sram_hot_expert_config tests
# ---------------------------------------------------------------------------


def test_build_sram_hot_expert_config_budget():
    """build_sram_hot_expert_config respects L1 budget and prioritizes by frequency."""
    layer_idx = 3
    num_experts = 8
    expert_indices = list(range(num_experts))
    sd = _make_state_dict(layer_idx, expert_indices, K=K, N=N)
    assigner = _make_assigner()
    core_grids = SramExpertCoreGrids.uniform(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    )

    freqs = [0] * 256
    for i, e in enumerate(expert_indices):
        freqs[e] = 100 - i * 10

    routing_frequencies = {layer_idx: freqs}

    large_budget = 10 * 1024 * 1024
    config = build_sram_hot_expert_config(sd, [layer_idx], assigner, core_grids, large_budget, routing_frequencies)
    assert layer_idx in config
    assert len(config[layer_idx]) == num_experts

    tiny_budget = 1
    config_tiny = build_sram_hot_expert_config(sd, [layer_idx], assigner, core_grids, tiny_budget, routing_frequencies)
    assert config_tiny.get(layer_idx) is None or len(config_tiny.get(layer_idx, [])) == 0


def test_build_sram_hot_expert_config_zero_budget():
    """Zero budget yields no selected experts."""
    layer_idx = 5
    sd = _make_state_dict(layer_idx, [0, 1], K=K, N=N)
    assigner = _make_assigner()
    core_grids = SramExpertCoreGrids.uniform(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    )
    freqs = [0] * 256
    freqs[0] = 50
    freqs[1] = 30
    routing_frequencies = {layer_idx: freqs}

    config = build_sram_hot_expert_config(sd, [layer_idx], assigner, core_grids, 0, routing_frequencies)
    assert config.get(layer_idx) is None


def test_build_sram_hot_expert_config_multi_layer():
    """Multi-layer config produces independent allocations per layer."""
    layers = [3, 5]
    experts_per_layer = {3: [0, 1, 2], 5: [4, 5]}
    sd = {}
    for li in layers:
        sd.update(_make_state_dict(li, experts_per_layer[li], K=K, N=N))

    assigner = _make_assigner()
    core_grids = SramExpertCoreGrids.uniform(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    )

    routing_frequencies = {}
    for li in layers:
        freqs = [0] * 256
        for i, e in enumerate(experts_per_layer[li]):
            freqs[e] = 100 - i * 10
        routing_frequencies[li] = freqs

    large_budget = 10 * 1024 * 1024
    config = build_sram_hot_expert_config(sd, layers, assigner, core_grids, large_budget, routing_frequencies)

    assert 3 in config and 5 in config
    assert len(config[3]) == 3
    assert len(config[5]) == 2
    logger.info(f"Multi-layer config: {config}")


# ---------------------------------------------------------------------------
# BSPM real-assignment tests
#
# All tests in this section require BSPM_RESULTS_DIR to point to a BitSculpt
# results directory containing:
#   deepseek-r1-0528/layer_4/precision_eval/precision_map_B_3.5.bspm
#
# The layer-4 Variant-B 3.5 b/e assignment has 0 bfp8 tiles — the exact
# condition that exercises the BSPM path through _build_l1_compressed_tensor
# and prepare_compressed_sram_slots.
# ---------------------------------------------------------------------------

# DeepSeek gate/up and down projection shapes (K, N) in compute layout.
# HF state dict stores weights as (out_features, in_features); .T gives (K, N).
_BSPM_LAYER_IDX = 4
_GATE_UP_SHAPE = (7168, 2048)  # K=7168, N=2048
_DOWN_SHAPE = (2048, 7168)  # K=2048, N=7168
_PROJ_SHAPES = [_GATE_UP_SHAPE, _GATE_UP_SHAPE, _DOWN_SHAPE]  # indexed by proj_idx

# Projection index constants matching BSPM file convention.
_PROJ_GATE = 0
_PROJ_UP = 1
_PROJ_DOWN = 2

# Core counts: N must be divisible by num_cores and N/num_cores must be
# tile-aligned (multiple of 32).
# gate/up: N=2048, 16 cores → 128 cols/core → 4 tile-cols (valid)
# down:    N=7168, 28 cores → 256 cols/core → 8 tile-cols (valid)
_GATE_UP_NUM_CORES = 16
_DOWN_NUM_CORES = 28


def _get_bspm_path() -> Path:
    """Return path to layer-4 BSPM file, or pytest.skip if unavailable."""
    bspm_dir = os.environ.get("BSPM_RESULTS_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_RESULTS_DIR not set")
    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")
    return bspm_path


def _make_bspm_assignment_provider(bspm_path: Path):
    """Return a callable (expert_idx, proj_idx) -> np.ndarray for load_bspm_for_expert."""
    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert

    def provider(expert_idx: int, proj_idx: int) -> np.ndarray:
        K, N = _PROJ_SHAPES[proj_idx]
        return load_bspm_for_expert(
            str(bspm_path),
            expert_idx=expert_idx,
            proj_idx=proj_idx,
            tile_rows=K // TILE_W,
            tile_cols=N // TILE_W,
        )

    return provider


def _make_bspm_state_dict(layer_idx: int, expert_indices: list[int]) -> dict:
    """Random weights at actual DeepSeek projection shapes (HF layout: out × in)."""
    sd = {}
    K_gate, N_gate = _GATE_UP_SHAPE
    K_down, N_down = _DOWN_SHAPE
    for e in expert_indices:
        torch.manual_seed(e * 1000 + layer_idx)
        sd[f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(N_gate, K_gate)
        sd[f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(N_gate, K_gate)
        sd[f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(N_down, K_down)
    return sd


def _make_bspm_core_grids(device) -> SramExpertCoreGrids:
    """Production-matching core grids for the BSPM shapes."""
    gate_grid = _build_sram_core_grid(device, _GATE_UP_NUM_CORES)
    down_grid = _build_sram_core_grid(device, _DOWN_NUM_CORES)
    return SramExpertCoreGrids(gate=gate_grid, up=gate_grid, down=down_grid)


def test_build_l1_compressed_tensor_bspm(device):
    """_build_l1_compressed_tensor with a real BSPM gate_proj assignment (0 bfp8 tiles).

    Exercises the assignment= path in _build_l1_compressed_tensor using the layer-4
    Variant-B 3.5 b/e BSPM assignment for expert 0.  Verifies the CT is allocated in
    L1 with 0 bfp8 tiles and valid per-core addresses.
    Requires BSPM_RESULTS_DIR.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert

    bspm_path = _get_bspm_path()
    K, N = _GATE_UP_SHAPE
    core_grid = _build_sram_core_grid(device, _GATE_UP_NUM_CORES)

    assignment = load_bspm_for_expert(
        str(bspm_path), expert_idx=0, proj_idx=_PROJ_GATE, tile_rows=K // TILE_W, tile_cols=N // TILE_W
    )

    torch.manual_seed(0)
    weight = torch.randn(K, N).float()

    ct = _build_l1_compressed_tensor(weight, core_grid, assignment=assignment, device=device)

    assert isinstance(ct, CompressedTensor)
    assert (
        ct.tile_counts.get("bfp8", 0) == 0
    ), f"Real BSPM 3.5 b/e should have 0 bfp8 tiles, got tile_counts={ct.tile_counts}"
    assert ct.tile_counts.get("bfp4", 0) > 0, f"Expected bfp4 tiles, got {ct.tile_counts}"

    cores = ttnn.corerange_to_cores(core_grid)
    for c in cores:
        addr = ct.get_data_l1_address_per_core(c)
        assert addr > 0, f"core ({c.x},{c.y}) has invalid L1 address"

    logger.info(f"BSPM gate_proj L1 CT tile_counts: {ct.tile_counts} on {len(cores)} cores")


def test_build_l1_compressed_tensor_bspm_down_proj(device):
    """_build_l1_compressed_tensor with real BSPM down_proj assignment (K=2048, N=7168).

    28 cores: N=7168/28=256 cols/core → 8 tile-cols (tile-aligned).
    Requires BSPM_RESULTS_DIR.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert

    bspm_path = _get_bspm_path()
    K, N = _DOWN_SHAPE
    core_grid = _build_sram_core_grid(device, _DOWN_NUM_CORES)

    assignment = load_bspm_for_expert(
        str(bspm_path), expert_idx=0, proj_idx=_PROJ_DOWN, tile_rows=K // TILE_W, tile_cols=N // TILE_W
    )

    torch.manual_seed(1)
    weight = torch.randn(K, N).float()

    ct = _build_l1_compressed_tensor(weight, core_grid, assignment=assignment, device=device)

    assert isinstance(ct, CompressedTensor)
    assert ct.tile_counts.get("bfp8", 0) == 0, f"Real BSPM 3.5 b/e should have 0 bfp8 tiles, got {ct.tile_counts}"
    assert ct.tile_counts.get("bfp4", 0) > 0, f"Expected bfp4 tiles, got {ct.tile_counts}"

    cores = ttnn.corerange_to_cores(core_grid)
    for c in cores:
        addr = ct.get_data_l1_address_per_core(c)
        assert addr > 0, f"core ({c.x},{c.y}) has invalid L1 address"

    logger.info(f"BSPM down_proj L1 CT tile_counts: {ct.tile_counts} on {len(cores)} cores")


def test_prepare_compressed_sram_slots_bspm(device):
    """prepare_compressed_sram_slots with real BSPM assignments for all three projections.

    Uses the assignment_provider path: real {bfp4, bfp2, zero} assignments from the
    layer-4 Variant-B 3.5 b/e BSPM file.  Uses expert_idx=5 (non-zero) to verify
    that per-expert BSPM loading is not accidentally hardcoded to expert 0.
    Validates that all slots allocate to L1 with 0 bfp8 tiles across gate, up, and
    down projections.
    Requires BSPM_RESULTS_DIR.
    """
    bspm_path = _get_bspm_path()
    expert_idx = 5

    sd = _make_bspm_state_dict(_BSPM_LAYER_IDX, [expert_idx])
    core_grids = _make_bspm_core_grids(device)
    assignment_provider = _make_bspm_assignment_provider(bspm_path)

    slots = prepare_compressed_sram_slots(
        device=device,
        state_dict=sd,
        layer_idx=_BSPM_LAYER_IDX,
        initial_expert_indices=[expert_idx],
        core_grids=core_grids,
        assignment_provider=assignment_provider,
        move_to_device=True,
    )

    assert slots.num_slots == 1
    assert slots.slot_experts == [expert_idx]

    proj_grids = {
        "gate": (slots.gate_proj, core_grids.gate),
        "up": (slots.up_proj, core_grids.up),
        "down": (slots.down_proj, core_grids.down),
    }
    for proj_name, (cts, grid) in proj_grids.items():
        ct = cts[0]
        assert isinstance(ct, CompressedTensor), f"{proj_name}: expected CompressedTensor"
        assert ct.tile_counts.get("bfp8", 0) == 0, f"{proj_name}: expected 0 bfp8 tiles, got {ct.tile_counts}"
        assert ct.tile_counts.get("bfp4", 0) > 0, f"{proj_name}: expected bfp4 tiles"
        cores = ttnn.corerange_to_cores(grid)
        for c in cores:
            addr = ct.get_data_l1_address_per_core(c)
            assert addr > 0, f"{proj_name} core ({c.x},{c.y}): invalid L1 address"
        logger.info(f"  BSPM {proj_name}: {ct.tile_counts}")


def test_sram_slot_fmt_tensors_bspm(device):
    """create_expert_fmt_tensors produces valid metadata tensors from BSPM-sourced slots.

    Mirrors test_sram_slot_fmt_tensors but uses the assignment_provider path with
    production shapes (gate_proj 7168×2048, 16 cores) and 1 expert.  One expert
    fills ~91% of the L1 bank (1327 KB / 1462 KB); a second would cause OOM.
    Verifies that fmt tensors are on-device and have the correct per-core structure
    when the underlying CTs come from real {bfp4, bfp2, zero} BSPM assignments.
    Requires BSPM_RESULTS_DIR.
    """
    bspm_path = _get_bspm_path()
    expert_indices = [0]

    sd = _make_bspm_state_dict(_BSPM_LAYER_IDX, expert_indices)
    core_grids = _make_bspm_core_grids(device)
    assignment_provider = _make_bspm_assignment_provider(bspm_path)

    slots = prepare_compressed_sram_slots(
        device=device,
        state_dict=sd,
        layer_idx=_BSPM_LAYER_IDX,
        initial_expert_indices=expert_indices,
        core_grids=core_grids,
        assignment_provider=assignment_provider,
        move_to_device=True,
    )

    K_gate, N_gate = _GATE_UP_SHAPE
    tiles_per_core = (K_gate // TILE_W) * (N_gate // _GATE_UP_NUM_CORES // TILE_W)
    fmt_tensors = create_expert_fmt_tensors(slots.gate_proj, device, core_grids.gate, tiles_per_core)

    assert len(fmt_tensors) > 0
    for coord, core_tensors in fmt_tensors.items():
        assert (
            len(core_tensors) == _GATE_UP_NUM_CORES
        ), f"Expected {_GATE_UP_NUM_CORES} core entries, got {len(core_tensors)}"
        for idx, t in core_tensors.items():
            assert ttnn.is_tensor_storage_on_device(t), f"fmt tensor core {idx} not on device"
    logger.info(
        f"BSPM fmt_tensors: {len(fmt_tensors)} device coords, "
        f"{_GATE_UP_NUM_CORES} cores each, tiles_per_core={tiles_per_core}"
    )


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

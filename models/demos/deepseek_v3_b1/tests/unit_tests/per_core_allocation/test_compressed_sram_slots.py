# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for SramCompressedExpertSlots — SRAM hot expert slots using CompressedTensor.

Validates:
  - Dataclass structure and is_dram_flags routing helper
  - Single-device slot preparation with per-core L1 allocation
  - ExpertKernel fmt metadata tensor creation from slots
  - Expert index encoding and selection meta for SRAM/DRAM routing
"""

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import create_expert_fmt_tensors, encode_expert_indices
from models.demos.deepseek_v3_b1.weights.prepare import (
    SramCompressedExpertSlots,
    SramHotExpertConfig,
    _build_l1_compressed_tensor,
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
        core_grid=core_grid,
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
        core_grid=core_grid,
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

    # Layer 3: 2 slots
    sd3 = _make_state_dict(3, config[3], K=K, N=N)
    slots3 = prepare_compressed_sram_slots(
        device=device,
        state_dict=sd3,
        layer_idx=3,
        initial_expert_indices=config[3],
        core_grid=core_grid,
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
        core_grid=core_grid,
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

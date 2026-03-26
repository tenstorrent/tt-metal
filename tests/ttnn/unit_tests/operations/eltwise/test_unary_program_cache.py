# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for eltwise unary program cache behavior.

Tests target potential caching issues.
The unary operation uses 3 ProgramFactory variants:
  - UnaryProgramFactory (interleaved)
  - UnarySubCoreGridProgramFactory (explicit sub_core_grids)
  - UnaryShardedProgramFactory (sharded input)

compute_program_hash() hashes:
  TILE layout:  args, sub_core_grids, factory_index, input_dtype, input_memory_config, volume, layout
  ROW_MAJOR:    args, sub_core_grids, factory_index, input_dtype, input_memory_config, padded_shape, layout

Where args = entire UnaryParams (op_chain, output_dtype, output_memory_config,
fp32_dest_acc_en, preserve_fp32_precision, bfp8_pack_precise, sub_core_grids).

override_runtime_arguments() only updates buffer addresses — shape/work
distribution changes require separate cache entries.

For TILE layout, only volume is hashed. Same volume = same num_pages = same
work distribution, so different shapes with same volume correctly share a
cache entry.
"""

import os
import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from models.common.utility_functions import is_wormhole_b0


def is_simulator():
    return os.environ.get("TT_METAL_SIMULATOR") != None


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_unary_op(device, op, shape, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run a unary op on device and return (torch_result, ttnn_result)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]

    torch_a = torch.rand(shape, dtype=torch_dtype) + 0.1
    torch_ops = {ttnn.relu: torch.relu, ttnn.sqrt: torch.sqrt, ttnn.abs: torch.abs, ttnn.floor: torch.floor}
    torch_result = torch_ops[op](torch_a)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_result = op(tt_a, memory_config=memory_config)
    tt_result = ttnn.to_torch(tt_result)

    return torch_result, tt_result


# =============================================================================
# Cache reuse tests
# =============================================================================


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_reuse_same_config(device, isolate_program_cache):
    """Same op, same shape, same dtype run twice -> 1 cache entry, different outputs."""
    shape = [1, 1, 32, 64]

    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch.manual_seed(42)
    torch_ref2, tt_out2 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 1
    assert not torch.equal(tt_out1, tt_out2)


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_reuse_same_volume_different_shapes(device, isolate_program_cache):
    """TILE layout: same volume, different shapes -> 1 cache entry.
    unary_ng doesn't hash volume or shape; tile counts are runtime args,
    so any shape with the same op/dtype/memory_config shares one entry."""
    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, [1, 1, 32, 64], dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_unary_op(device, ttnn.relu, [1, 1, 64, 32], dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 1


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_reuse_different_volumes(device, isolate_program_cache):
    """TILE layout: different volumes -> still 1 cache entry.
    unary_ng uses runtime tile counts (not compile-time), so different volumes
    share the same compiled program. override_runtime_arguments handles the
    different per-core tile distributions on cache hit."""
    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, [1, 1, 32, 32], dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_unary_op(device, ttnn.relu, [1, 1, 64, 64], dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 1


# =============================================================================
# Cache miss tests (fields correctly included in hash)
# =============================================================================


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_miss_different_op_types(device, isolate_program_cache):
    """Different unary op types -> different cache entries."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_unary_op(device, ttnn.sqrt, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_miss_different_input_dtypes(device, isolate_program_cache):
    """Different input dtypes -> different cache entries."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_miss_different_memory_configs(device, isolate_program_cache):
    """Different memory configs -> different cache entries."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_unary_op(
        device, ttnn.relu, shape, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_unary_op(
        device, ttnn.relu, shape, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_miss_different_sub_core_grids(device, isolate_program_cache):
    """Different sub_core_grids -> different cache entries.
    sub_core_grids is part of UnaryParams (hashed via args) and also hashed explicitly.
    Uses ttnn.floor which supports sub_core_grids parameter."""
    shape = [1, 1, 32, 64]

    torch_a1 = torch.rand(shape, dtype=torch.float32) + 0.5
    torch_ref1 = torch.floor(torch_a1)
    grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out1 = ttnn.floor(tt_a1, sub_core_grids=grid_a)
    assert_with_pcc(torch_ref1, ttnn.to_torch(tt_out1), 0.9999)

    torch_a2 = torch.rand(shape, dtype=torch.float32) + 0.5
    torch_ref2 = torch.floor(torch_a2)
    grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])
    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = ttnn.floor(tt_a2, sub_core_grids=grid_b)
    assert_with_pcc(torch_ref2, ttnn.to_torch(tt_out2), 0.9999)

    assert device.num_program_cache_entries() == 2


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_miss_different_factories(device, isolate_program_cache):
    """Interleaved vs sub_core_grids factory -> different cache entries.
    factory_index is included in the hash.
    Uses ttnn.floor which supports sub_core_grids parameter."""
    shape = [1, 1, 32, 64]

    # Default: UnaryProgramFactory
    torch_ref1, tt_out1 = run_unary_op(device, ttnn.floor, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    # Explicit sub_core_grids: UnarySubCoreGridProgramFactory
    torch_a2 = torch.rand(shape, dtype=torch.float32) + 0.5
    torch_ref2 = torch.floor(torch_a2)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])
    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = ttnn.floor(tt_a2, sub_core_grids=grid)
    assert_with_pcc(torch_ref2, ttnn.to_torch(tt_out2), 0.9999)

    assert device.num_program_cache_entries() == 2


# =============================================================================
# Correctness under cache reuse
# =============================================================================


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_correctness_repeated_runs(device, isolate_program_cache):
    """Run same op 5 times with different data -> all results correct."""
    shape = [1, 1, 32, 64]
    for _ in range(5):
        torch_ref, tt_out = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
        assert_with_pcc(torch_ref, tt_out, 0.9999)

    assert device.num_program_cache_entries() == 1


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_correctness_same_volume_different_shapes(device, isolate_program_cache):
    """Same volume, different shapes all produce correct results via cache reuse."""
    # All have volume 2048: 32*64, 64*32
    for shape in [[1, 1, 32, 64], [1, 1, 64, 32]]:
        torch_ref, tt_out = run_unary_op(device, ttnn.sqrt, shape, dtype=ttnn.float32)
        assert_with_pcc(torch_ref, tt_out, 0.9999)

    assert device.num_program_cache_entries() == 1


# =============================================================================
# ROW_MAJOR cache tests
# =============================================================================


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_cache_rm_different_widths_need_separate_entries(device, isolate_program_cache):
    """ROW_MAJOR interleaved tensors with different widths have different page sizes,
    so compute_program_hash must produce distinct keys for each shape."""
    torch.manual_seed(0)
    torch_a = torch.empty([1, 1, 1024, 512], dtype=torch.bfloat16).uniform_(1, 100)
    torch_b = torch.empty([1, 1, 512, 1024], dtype=torch.bfloat16).uniform_(1, 100)
    torch_result1 = torch.abs(torch_a)
    torch_result2 = torch.abs(torch_b)
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_result1 = ttnn.abs(tt_a)
    tt_result2 = ttnn.abs(tt_b)
    result1 = ttnn.to_torch(tt_result1)
    result2 = ttnn.to_torch(tt_result2)
    assert torch.equal(result1, torch_result1)
    assert torch.equal(result2, torch_result2)
    assert device.num_program_cache_entries() == 2


# =============================================================================
# Sharded cache tests (GitHub issue #33910)
# =============================================================================


@pytest.mark.skipif(is_simulator() and is_wormhole_b0(), reason="Issue #38203")
def test_unary_sharded_cache_correctness_different_grids(device, isolate_program_cache):
    """Sharded ttnn.abs with different grid configs must produce correct results.
    Reproduces GitHub issue #33910: ttnn.abs ProgramCache data corruption.
    The (64,64) on 2x2 grid case failed when preceded by other shard configs."""
    torch.manual_seed(0)
    test_params = [
        ((32, 128), (3, 0)),
        ((64, 32), (0, 1)),
        ((64, 64), (1, 1)),
    ]
    for shape, grid_size in test_params:
        core_grid = ttnn.CoreGrid(x=grid_size[0] + 1, y=grid_size[1] + 1)
        memory_config = ttnn.create_sharded_memory_config(
            shape=shape,
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
        output = ttnn.abs(input_tensor)
        tt_out = ttnn.to_torch(output)
        torch_ref = torch.abs(torch_tensor)
        assert_with_pcc(torch_ref, tt_out, 0.9999)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for binary_ng program cache behavior.

Tests target potential caching issues.
The binary_ng operation uses a single ProgramFactory with caching based on:

to_hash(): binary_op_type, lhs/rhs/post_activations, memory_config, get_dtype(),
           compute_kernel_config, sub_core_grids, subtile_broadcast_type,
           is_sfpu, is_quant_op, is_where_op

compute_program_hash(): attributes (via to_hash()), input tensor dtypes,
                        input tensor memory_configs, shard_volumes

Fields correctly excluded from hash (handled by override_runtime_arguments):
- logical_shape: not in compute_program_hash() by design - different logical
  shapes share a cache entry and runtime arguments are updated accordingly
- scalar.has_value(): not in to_hash(), but compute_program_hash branches
  on input_tensor_b presence
- input_dtype: not in to_hash(), but compute_program_hash includes input
  tensor dtypes directly
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_binary_ng_op(device, op, shape_a, shape_b, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run a binary op via binary_ng path and return (torch_result, ttnn_result)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]

    torch_a = torch.rand(shape_a, dtype=torch_dtype)
    torch_b = torch.rand(shape_b, dtype=torch_dtype)

    torch_ops = {ttnn.add: torch.add, ttnn.mul: torch.mul, ttnn.sub: torch.sub}
    torch_result = torch_ops[op](torch_a, torch_b)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_result = op(tt_a, tt_b, memory_config=memory_config)
    tt_result = ttnn.to_torch(tt_result)

    return torch_result, tt_result


def run_scalar_ng_op(device, op, shape, scalar, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run a binary-scalar op via binary_ng path and return (torch_result, ttnn_result)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]

    torch_a = torch.rand(shape, dtype=torch_dtype)

    torch_ops = {ttnn.add: lambda a, s: a + s, ttnn.mul: lambda a, s: a * s, ttnn.sub: lambda a, s: a - s}
    torch_result = torch_ops[op](torch_a, scalar)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_result = op(tt_a, scalar, memory_config=memory_config)
    tt_result = ttnn.to_torch(tt_result)

    return torch_result, tt_result


# =============================================================================
# Cache reuse tests (fields correctly excluded from hash)
# =============================================================================


def test_ng_cache_reuse_same_config(device, isolate_program_cache):
    """Same op, same shapes, same dtypes run twice -> 1 cache entry, different outputs."""
    shape = [1, 1, 32, 64]

    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch.manual_seed(42)
    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 1
    assert not torch.equal(tt_out1, tt_out2)


def test_ng_cache_reuse_scalar_different_values(device, isolate_program_cache):
    """Different scalar values but same op -> 1 cache entry, different outputs."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_scalar_ng_op(device, ttnn.add, shape, 0.5, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_scalar_ng_op(device, ttnn.add, shape, 1.5, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.num_program_cache_entries() == 1
    assert not torch.equal(tt_out1, tt_out2)


# =============================================================================
# Cache miss tests (fields correctly included in hash)
# =============================================================================


def test_ng_cache_miss_different_op_types(device, isolate_program_cache):
    """Different binary op types -> different cache entries."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.mul, shape, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 2


def test_ng_cache_miss_different_input_dtypes(device, isolate_program_cache):
    """Different input dtypes -> different cache entries.
    Differentiated via input tensor dtype in compute_program_hash()."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 2


def test_ng_cache_miss_different_memory_configs(device, isolate_program_cache):
    """Different memory configs -> different cache entries."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_binary_ng_op(
        device, ttnn.add, shape, shape, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_binary_ng_op(
        device, ttnn.add, shape, shape, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 2


def test_ng_cache_miss_different_subtile_broadcast(device, isolate_program_cache):
    """Different subtile broadcast types -> different cache entries.
    subtile_broadcast_type is in to_hash() and depends on last-2-dim shapes."""
    # NONE: equal shapes
    torch_ref1, tt_out1 = run_binary_ng_op(device, ttnn.add, [1, 1, 32, 64], [1, 1, 32, 64], dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    # ROW_B: b has single tile row (height_b=1), a is full
    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.add, [1, 1, 32, 64], [1, 1, 1, 64], dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 2


def test_ng_cache_miss_different_output_dtypes(device, isolate_program_cache):
    """Different output dtypes -> different cache entries."""
    shape = [1, 1, 32, 64]

    # bfloat16 input -> bfloat16 output
    torch_a1 = torch.rand(shape, dtype=torch.bfloat16)
    torch_b1 = torch.rand(shape, dtype=torch.bfloat16)
    torch_ref1 = torch.add(torch_a1, torch_b1)

    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b1 = ttnn.from_torch(torch_b1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out1 = ttnn.add(tt_a1, tt_b1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, ttnn.to_torch(tt_out1), 0.9999)

    # bfloat16 input -> float32 output
    torch_a2 = torch.rand(shape, dtype=torch.bfloat16)
    torch_b2 = torch.rand(shape, dtype=torch.bfloat16)
    torch_ref2 = torch.add(torch_a2, torch_b2)

    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b2 = ttnn.from_torch(torch_b2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = ttnn.add(tt_a2, tt_b2, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, ttnn.to_torch(tt_out2), 0.9999)

    assert device.num_program_cache_entries() == 2


def test_ng_scalar_vs_tensor_cache_differentiation(device, isolate_program_cache):
    """Scalar op vs tensor op -> different cache entries.
    scalar.has_value() is not in to_hash(), but compute_program_hash()
    naturally differentiates because the scalar path excludes tensor_b
    from hash arguments while the tensor path includes it."""
    shape = [1, 1, 32, 64]

    # Scalar path
    torch_ref1, tt_out1 = run_scalar_ng_op(device, ttnn.add, shape, 0.5, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    # Tensor path
    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 2


def test_ng_cache_miss_different_sub_core_grids(device, isolate_program_cache):
    """Different sub_core_grids -> different cache entries.
    sub_core_grids is in to_hash() and directly determines worker_grid."""
    shape = [1, 1, 32, 64]

    torch_a1 = torch.rand(shape, dtype=torch.float32)
    torch_b1 = torch.rand(shape, dtype=torch.float32)
    torch_ref1 = torch.add(torch_a1, torch_b1)

    grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b1 = ttnn.from_torch(torch_b1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out1 = ttnn.add(tt_a1, tt_b1, sub_core_grids=grid_a)
    assert_with_pcc(torch_ref1, ttnn.to_torch(tt_out1), 0.9999)

    torch_a2 = torch.rand(shape, dtype=torch.float32)
    torch_b2 = torch.rand(shape, dtype=torch.float32)
    torch_ref2 = torch.add(torch_a2, torch_b2)

    grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])
    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b2 = ttnn.from_torch(torch_b2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = ttnn.add(tt_a2, tt_b2, sub_core_grids=grid_b)
    assert_with_pcc(torch_ref2, ttnn.to_torch(tt_out2), 0.9999)

    assert device.num_program_cache_entries() == 2


def test_ng_different_input_dtypes_same_output_dtype(device, isolate_program_cache):
    """Different input dtypes with same output dtype -> different cache entries.
    input_dtype is not in to_hash(), but compute_program_hash() includes
    input tensor dtypes directly, which compensates."""
    shape = [1, 1, 32, 64]

    # bfloat16 input -> float32 output
    torch_a1 = torch.rand(shape, dtype=torch.bfloat16)
    torch_b1 = torch.rand(shape, dtype=torch.bfloat16)
    torch_ref1 = torch.add(torch_a1, torch_b1)

    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b1 = ttnn.from_torch(torch_b1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out1 = ttnn.add(tt_a1, tt_b1, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, ttnn.to_torch(tt_out1), 0.9999)

    # float32 input -> float32 output (same output dtype, different input dtype)
    torch_a2 = torch.rand(shape, dtype=torch.float32)
    torch_b2 = torch.rand(shape, dtype=torch.float32)
    torch_ref2 = torch.add(torch_a2, torch_b2)

    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b2 = ttnn.from_torch(torch_b2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = ttnn.add(tt_a2, tt_b2, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, ttnn.to_torch(tt_out2), 0.9999)

    assert device.num_program_cache_entries() == 2


# =============================================================================
# Cache reuse tests: logical_shape correctly excluded from hash
#
# Different logical shapes share a cache entry by design. Hashing logical
# shapes would be overkill since override_runtime_arguments handles shape
# differences at runtime.
# =============================================================================


def test_ng_cache_reuse_different_logical_shapes(device, isolate_program_cache):
    """Different logical shapes share 1 cache entry, different outputs (by design).
    logical_shape is correctly excluded from compute_program_hash();
    override_runtime_arguments handles shape differences at runtime."""
    torch_ref1, tt_out1 = run_binary_ng_op(device, ttnn.add, [1, 1, 32, 32], [1, 1, 32, 32], dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.add, [1, 1, 64, 64], [1, 1, 64, 64], dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.num_program_cache_entries() == 1
    assert tt_out1.shape != tt_out2.shape


def test_ng_cache_reuse_different_logical_shapes_correctness(device, isolate_program_cache):
    """Correctness across multiple logical shapes sharing a single cache entry.
    override_runtime_arguments correctly updates runtime args for each shape."""
    for shape_dim in [32, 64, 128]:
        shape = [1, 1, shape_dim, shape_dim]
        torch_a = torch.rand(shape, dtype=torch.float32)
        torch_b = torch.rand(shape, dtype=torch.float32)
        torch_ref = torch.add(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
        tt_out = ttnn.add(tt_a, tt_b)
        assert_with_pcc(torch_ref, ttnn.to_torch(tt_out), 0.9999)

    assert device.num_program_cache_entries() == 1


# =============================================================================
# Correctness under cache reuse
# =============================================================================


def test_ng_cache_correctness_repeated_runs(device, isolate_program_cache):
    """Run same op 5 times with different data -> all results correct."""
    shape = [1, 1, 32, 64]
    for _ in range(5):
        torch_ref, tt_out = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.float32)
        assert_with_pcc(torch_ref, tt_out, 0.9999)


def test_ng_cache_correctness_scalar_repeated(device, isolate_program_cache):
    """Scalar ops with varying values -> all numerically correct."""
    shape = [1, 1, 32, 64]

    for scalar in [0.25, 0.5, 0.75, 1.0, 1.5]:
        torch_ref, tt_out = run_scalar_ng_op(device, ttnn.add, shape, scalar, dtype=ttnn.float32)
        assert_with_pcc(torch_ref, tt_out, 0.999)


def test_ng_cache_correctness_broadcast_repeated(device, isolate_program_cache):
    """Broadcast operations with cache reuse -> all results correct."""
    shape_a = [1, 1, 64, 64]
    shape_b = [1, 1, 1, 64]

    for _ in range(3):
        torch_ref, tt_out = run_binary_ng_op(device, ttnn.add, shape_a, shape_b, dtype=ttnn.float32)
        assert_with_pcc(torch_ref, tt_out, 0.9999)

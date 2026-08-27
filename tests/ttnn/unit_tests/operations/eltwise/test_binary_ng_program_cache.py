# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for binary_ng program cache behavior.

Tests target potential caching issues.
The binary_ng operation uses a single ProgramFactory with caching based on:

operation_attributes_t::attribute_values(): binary_op_type, lhs/rhs/post_activations,
           memory_config, get_dtype(), compute_kernel_config, sub_core_grids,
           worker_grid, subtile_broadcast_type, is_sfpu, is_quant_op, is_where_op,
           input_layout_a/b, output_layout, equal_nan, the shard volumes, and
           c_tensor_shape_in_pages (the sharded output's tensor shape in pages on the accessor path)

tensor_args_t::to_hash(): input tensor dtypes and memory_configs, plus each
           sharded input's tensor shape in pages (BufferDistributionSpec::tensor_shape_in_pages)

The default compute_program_hash() combines both of the above.

Fields correctly excluded from hash (handled by override_runtime_arguments):
- logical_shape: not in compute_program_hash() by design - differently-shaped
  INTERLEAVED calls share a cache entry and runtime arguments are updated
  accordingly. This does not extend to sharded calls, for two reasons that no
  runtime re-application can repair: the native-vs-accessor regime is a
  shape-dependent COMPILE-TIME decision (is_uneven -> is_native_L1_sharding ->
  kernel defines), and on the accessor path the TensorAccessor's page geometry is
  baked into the program. So sharded operands and sharded outputs each contribute
  their tensor shape in pages to the key (issue #54138)
- scalar.has_value(): not in attribute_values(), but compute_program_hash branches
  on input_tensor_b presence
- input_dtype: not in attribute_values(), but compute_program_hash includes input
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
    with device.cache_entries_counter.measure():
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
    with device.cache_entries_counter.measure():
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

    assert device.cache_entries_counter.total == 1
    assert not torch.equal(tt_out1, tt_out2)


@pytest.mark.parametrize(
    "shape_first, shape_second",
    [
        ([1, 1, 32, 64], [1, 1, 128, 256]),  # grow volume
        ([1, 1, 128, 256], [1, 1, 32, 64]),  # shrink volume
    ],
)
def test_ng_inplace_cache_reuse_different_shapes(device, isolate_program_cache, shape_first, shape_second):
    """binary_ng re-applies all per-dispatch state on a cache hit via override_runtime_arguments
    (#48928). An in-place add (output_tensor aliases input) with different logical shapes shares one
    cache entry (volume is excluded from the hash), so the second call is a cache HIT that reuses the
    first program WITHOUT rebuild. binary_ng's override_runtime_arguments must re-derive every per-core
    arg for the current shape or the reused program corrupts the result. Regression guard for the
    in-place cache-hit path (the SDXL silu / moreh class of bug)."""

    def inplace_add(shape, seed):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16)
        b = torch.rand(shape, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        with device.cache_entries_counter.measure():
            tt_c = ttnn.add(tt_a, tt_b, output_tensor=tt_a)  # in-place
        # Prove the op is ACTUALLY in-place: if a future change ignored output_tensor or allocated a
        # fresh output, PCC + single-cache-entry would still pass while no longer testing the alias.
        assert tt_c.buffer_address() == tt_a.buffer_address()
        return a + b, ttnn.to_torch(tt_c)

    ref1, out1 = inplace_add(shape_first, 0)
    assert_with_pcc(ref1, out1, 0.999)

    ref2, out2 = inplace_add(shape_second, 1)  # cache HIT on the differently-shaped program
    assert_with_pcc(ref2, out2, 0.999)

    assert device.cache_entries_counter.total == 1  # proves it was a hit, not a rebuild masking the bug


def test_ng_inplace_cache_hit_sharded_readdresses(device, isolate_program_cache):
    """binary_ng in-place add on SHARDED tensors — the sharding mode SDXL exercised (silu) — driven
    through binary_ng's override_runtime_arguments cache-hit path. Repeated at the SAME shard config
    (a sharded operand's tensor shape in pages is in the key, so a different shape would MISS, not hit) but
    with freshly-allocated operands kept alive, so each cache HIT sees a DIFFERENT buffer address.
    binary_ng's tensor-backed CB / rt-arg addresses must be re-applied on the hit (no rebuild) or the
    result is stale."""
    shape = [1, 1, 256, 256]
    mem = ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=8, x=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    keep_alive = []  # hold refs so each iteration's tensors get fresh (different) addresses
    for i in range(4):
        torch.manual_seed(i)
        a = torch.rand(shape, dtype=torch.bfloat16)
        b = torch.rand(shape, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        with device.cache_entries_counter.measure():
            tt_c = ttnn.add(tt_a, tt_b, output_tensor=tt_a)  # in-place, sharded (output preallocated)
        # prove it's actually in-place. NOTE: buffer_address() requires a single-address buffer, which
        # standard create_sharded_memory_config gives; a per-core-allocation config would TT_FATAL here
        # (use experimental_per_core_buffer_address if this test is ever parametrized onto one).
        assert tt_c.buffer_address() == tt_a.buffer_address()
        keep_alive += [tt_a, tt_b, tt_c]
        assert_with_pcc(a + b, ttnn.to_torch(tt_c), 0.999)

    # One shared program reused across all four differently-addressed in-place hits.
    assert device.cache_entries_counter.total == 1


@pytest.mark.parametrize("first_inplace", [True, False], ids=["inplace_first", "outofplace_first"])
def test_ng_cache_mixed_inplace_outofplace_interleaved(device, isolate_program_cache, first_inplace):
    """REGRESSION (aliased address re-derivation): one cached INTERLEAVED program reused across a MIX of
    in-place (output_tensor aliases an input) and out-of-place calls sharing a single cache entry
    (logical shape is excluded from the hash). The legacy resolve_bindings maps an aliased buffer to its
    FIRST occurrence, so a program built under one aliasing pattern and reused under another would patch
    the writer's output address from the wrong tensor slot. binary_ng's override_runtime_arguments
    re-derives every rt-arg address for the actual current tensors, so it MUST survive both orders —
    this proves the rt-arg axis is re-applied, not just that a cache hit occurred."""

    def do(seed, inplace):
        torch.manual_seed(seed)
        a = torch.rand([1, 1, 32, 64], dtype=torch.bfloat16)
        b = torch.rand([1, 1, 32, 64], dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = tt_a if inplace else None
        with device.cache_entries_counter.measure():
            tt_c = ttnn.add(tt_a, tt_b, output_tensor=out)
        if inplace:
            assert tt_c.buffer_address() == tt_a.buffer_address()
        return a + b, ttnn.to_torch(tt_c)

    # Alternate aliasing across the SAME cache entry, in both orders.
    for i, inplace in enumerate([first_inplace, not first_inplace, first_inplace, not first_inplace]):
        ref, out = do(i, inplace)
        assert_with_pcc(ref, out, 0.999)
    assert device.cache_entries_counter.total == 1


@pytest.mark.parametrize("first_inplace", [True, False], ids=["inplace_first", "outofplace_first"])
def test_ng_cache_mixed_inplace_outofplace_sharded(device, isolate_program_cache, first_inplace):
    """REGRESSION (SDXL-class, the axis that actually broke): one cached SHARDED program reused across
    a MIX of in-place and out-of-place calls sharing a single cache entry. For sharded ops the input/
    output addresses ride on tensor-backed CB base addresses. Under the legacy path these were patched
    by resolved_bindings.cbs via first-occurrence resolution (and get_dynamic could not touch CBs at
    all), so a program built under one aliasing pattern and reused under another mis-resolved the output
    CB to the wrong tensor slot with NOTHING to correct it — the exact PCC~0 behind SDXL in-place silu.
    binary_ng's override_runtime_arguments re-applies the CB addresses by CBIndex from the current
    tensors, so both orders must stay correct."""
    shape = [1, 1, 256, 256]
    mem = ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=8, x=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    keep_alive = []  # hold refs so successive calls see fresh (different) buffer addresses

    def do(seed, inplace):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16)
        b = torch.rand(shape, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        if inplace:
            out = tt_a
        else:
            out = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
            )
        with device.cache_entries_counter.measure():
            tt_c = ttnn.add(tt_a, tt_b, output_tensor=out)
        assert tt_c.buffer_address() == out.buffer_address()
        keep_alive.extend([tt_a, tt_b, tt_c, out])
        return a + b, ttnn.to_torch(tt_c)

    for i, inplace in enumerate([first_inplace, not first_inplace, first_inplace, not first_inplace]):
        ref, out = do(i, inplace)
        assert_with_pcc(ref, out, 0.999)
    # Same shard config AND same shape across all calls → identical tensor shape in pages → one shared cache entry.
    assert device.cache_entries_counter.total == 1


def test_ng_cache_reuse_scalar_different_values(device, isolate_program_cache):
    """Different scalar values but same op -> 1 cache entry, different outputs."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_scalar_ng_op(device, ttnn.add, shape, 0.5, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    torch_ref2, tt_out2 = run_scalar_ng_op(device, ttnn.add, shape, 1.5, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.999)

    assert device.cache_entries_counter.total == 1
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

    assert device.cache_entries_counter.total == 2


def test_ng_cache_miss_different_input_dtypes(device, isolate_program_cache):
    """Different input dtypes -> different cache entries.
    Differentiated via input tensor dtype in compute_program_hash()."""
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.cache_entries_counter.total == 2


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

    assert device.cache_entries_counter.total == 2


def test_ng_cache_miss_different_subtile_broadcast(device, isolate_program_cache):
    """Different subtile broadcast types -> different cache entries.
    subtile_broadcast_type is in attribute_values() and depends on last-2-dim shapes."""
    # NONE: equal shapes
    torch_ref1, tt_out1 = run_binary_ng_op(device, ttnn.add, [1, 1, 32, 64], [1, 1, 32, 64], dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.9999)

    # ROW_B: b has single tile row (height_b=1), a is full
    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.add, [1, 1, 32, 64], [1, 1, 1, 64], dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.cache_entries_counter.total == 2


def test_ng_cache_miss_different_output_dtypes(device, isolate_program_cache):
    """Different output dtypes -> different cache entries."""
    shape = [1, 1, 32, 64]

    # bfloat16 input -> bfloat16 output
    torch_a1 = torch.rand(shape, dtype=torch.bfloat16)
    torch_b1 = torch.rand(shape, dtype=torch.bfloat16)
    torch_ref1 = torch.add(torch_a1, torch_b1)

    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b1 = ttnn.from_torch(torch_b1, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out1 = ttnn.add(tt_a1, tt_b1, dtype=ttnn.bfloat16)
    assert_with_pcc(torch_ref1, ttnn.to_torch(tt_out1), 0.9999)

    # bfloat16 input -> float32 output
    torch_a2 = torch.rand(shape, dtype=torch.bfloat16)
    torch_b2 = torch.rand(shape, dtype=torch.bfloat16)
    torch_ref2 = torch.add(torch_a2, torch_b2)

    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b2 = ttnn.from_torch(torch_b2, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out2 = ttnn.add(tt_a2, tt_b2, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, ttnn.to_torch(tt_out2), 0.9999)

    assert device.cache_entries_counter.total == 2


def test_ng_scalar_vs_tensor_cache_differentiation(device, isolate_program_cache):
    """Scalar op vs tensor op -> different cache entries.
    scalar.has_value() is not in attribute_values(), but compute_program_hash()
    naturally differentiates because the scalar path excludes tensor_b
    from hash arguments while the tensor path includes it."""
    shape = [1, 1, 32, 64]

    # Scalar path
    torch_ref1, tt_out1 = run_scalar_ng_op(device, ttnn.add, shape, 0.5, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, tt_out1, 0.999)

    # Tensor path
    torch_ref2, tt_out2 = run_binary_ng_op(device, ttnn.add, shape, shape, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, tt_out2, 0.9999)

    assert device.cache_entries_counter.total == 2


def test_ng_cache_miss_different_sub_core_grids(device, isolate_program_cache):
    """Different sub_core_grids -> different cache entries.
    sub_core_grids is in attribute_values() and directly determines worker_grid."""
    shape = [1, 1, 32, 64]

    torch_a1 = torch.rand(shape, dtype=torch.float32)
    torch_b1 = torch.rand(shape, dtype=torch.float32)
    torch_ref1 = torch.add(torch_a1, torch_b1)

    grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b1 = ttnn.from_torch(torch_b1, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out1 = ttnn.add(tt_a1, tt_b1, sub_core_grids=grid_a)
    assert_with_pcc(torch_ref1, ttnn.to_torch(tt_out1), 0.9999)

    torch_a2 = torch.rand(shape, dtype=torch.float32)
    torch_b2 = torch.rand(shape, dtype=torch.float32)
    torch_ref2 = torch.add(torch_a2, torch_b2)

    grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])
    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b2 = ttnn.from_torch(torch_b2, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out2 = ttnn.add(tt_a2, tt_b2, sub_core_grids=grid_b)
    assert_with_pcc(torch_ref2, ttnn.to_torch(tt_out2), 0.9999)

    assert device.cache_entries_counter.total == 2


def test_ng_different_input_dtypes_same_output_dtype(device, isolate_program_cache):
    """Different input dtypes with same output dtype -> different cache entries.
    input_dtype is not in attribute_values(), but compute_program_hash() includes
    input tensor dtypes directly, which compensates."""
    shape = [1, 1, 32, 64]

    # bfloat16 input -> float32 output
    torch_a1 = torch.rand(shape, dtype=torch.bfloat16)
    torch_b1 = torch.rand(shape, dtype=torch.bfloat16)
    torch_ref1 = torch.add(torch_a1, torch_b1)

    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b1 = ttnn.from_torch(torch_b1, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out1 = ttnn.add(tt_a1, tt_b1, dtype=ttnn.float32)
    assert_with_pcc(torch_ref1, ttnn.to_torch(tt_out1), 0.9999)

    # float32 input -> float32 output (same output dtype, different input dtype)
    torch_a2 = torch.rand(shape, dtype=torch.float32)
    torch_b2 = torch.rand(shape, dtype=torch.float32)
    torch_ref2 = torch.add(torch_a2, torch_b2)

    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b2 = ttnn.from_torch(torch_b2, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out2 = ttnn.add(tt_a2, tt_b2, dtype=ttnn.float32)
    assert_with_pcc(torch_ref2, ttnn.to_torch(tt_out2), 0.9999)

    assert device.cache_entries_counter.total == 2


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

    assert device.cache_entries_counter.total == 1
    assert tt_out1.shape != tt_out2.shape


def test_ng_cache_reuse_different_logical_shapes_correctness(device, isolate_program_cache):
    """Correctness across multiple logical shapes sharing a single cache entry.
    override_runtime_arguments correctly updates runtime args for each shape."""
    for shape_dim in [32, 64, 128]:
        shape = [1, 1, shape_dim, shape_dim]
        torch_a = torch.rand(shape, dtype=torch.float32)
        torch_b = torch.rand(shape, dtype=torch.float32)

        with device.cache_entries_counter.measure():
            torch_ref = torch.add(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
        with device.cache_entries_counter.measure():
            tt_out = ttnn.add(tt_a, tt_b)
        assert_with_pcc(torch_ref, ttnn.to_torch(tt_out), 0.9999)

    assert device.cache_entries_counter.total == 1


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


# =============================================================================
# Cache miss tests: shape state the key must carry on the SCALAR path
# =============================================================================


def test_ng_scalar_sharded_cache_miss_when_evenness_flips(device, isolate_program_cache):
    """REGRESSION: the scalar overload must key on shape, because the native-vs-accessor regime is a
    compile-time decision that flips with is_uneven and no runtime re-application can repair it.

    ONE explicit ShardSpec held constant, so every other key field is identical across both calls. Only
    the height changes, which flips evenness:

        512 rows / shard height 64 -> 8 exact shards  -> even   -> native regime compiled
        480 rows / shard height 64 -> 7 full + 1 of 32 -> uneven -> needs the accessor regime

    Pre-fix this HUNG the device -- the reused native program busy-waits on an uneven shape, ignores
    SIGINT and needs a card reset. Expect that, not a wrong answer, if it regresses.

    Scope caveat: this does not isolate the shard-volume fix. `a` is sharded, so the shape in pages in
    to_hash() separates these shapes too, and evenness and page count are correlated anyway (the shard
    height is a whole number of tiles). It would pass with the shard-volume change reverted; it is an
    end-to-end guard on the regime flip, not proof of what separates the entries."""
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))])
    shard_spec = ttnn.ShardSpec(shard_grid, [64, 128], ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    def scalar_add(shape, seed):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        with device.cache_entries_counter.measure():
            tt_c = ttnn.add(tt_a, 1.5)
        return a + 1.5, ttnn.to_torch(tt_c)

    ref_even, out_even = scalar_add([1, 1, 512, 128], 0)
    assert_with_pcc(ref_even, out_even, 0.999)

    # Same ShardSpec, same dtype, same memory config, but this shape needs the accessor regime rather
    # than the native one the first call compiled.
    ref_uneven, out_uneven = scalar_add([1, 1, 480, 128], 1)
    assert_with_pcc(ref_uneven, out_uneven, 0.999)

    # Proves the key distinguished the two regimes rather than silently reusing the first program.
    assert device.cache_entries_counter.total == 2


def test_ng_sharded_output_cache_miss_across_page_counts(device, isolate_program_cache):
    """REGRESSION (issue #54138): the OUTPUT side of the accessor-path shape collision.

    Both inputs are interleaved and only the output is sharded, via an explicit memory_config. The
    identical-shape branch of is_native_L1_sharding rejects DRAM inputs, so the native path declines and
    the writer reaches the output through a TensorAccessor -- built over the output buffer with the same
    ArgConfig::RuntimeTensorShape common arg that override_runtime_arguments never refreshes.

    Nothing else in the key varies with the output's extent: the inputs are interleaved so their page
    shapes in pages are absent, and attributes.memory_config carries the output's shard spec but not its shape.
    Measured on Wormhole before the fix, small-then-big: the second call was a cache hit with 14693/16384
    elements wrong.

    This is more reachable than the input-side case, which needs an explicit ShardSpec on an input --
    here the inputs are ordinary interleaved DRAM tensors and only memory_config is unusual."""
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_spec = ttnn.ShardSpec(core_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    def add_into_sharded(shape, seed):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16)
        b = torch.rand(shape, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        with device.cache_entries_counter.measure():
            tt_c = ttnn.add(tt_a, tt_b, memory_config=out_mem)
        return a + b, ttnn.to_torch(tt_c)

    ref_small, out_small = add_into_sharded([1, 1, 64, 128], 0)
    assert_with_pcc(ref_small, out_small, 0.999)

    ref_big, out_big = add_into_sharded([1, 1, 128, 128], 1)
    assert_with_pcc(ref_big, out_big, 0.999)

    assert device.cache_entries_counter.total == 2


def test_ng_scalar_interleaved_memcfg_with_sharded_output_cache_miss(device, isolate_program_cache):
    """REGRESSION: an explicit INTERLEAVED memory_config alongside a SHARDED preallocated output.

    mem_config_actual falls back to the output tensor's config only when no explicit memory_config was
    given, so this combination leaves it interleaved -- while compute_output_specs returns a supplied
    output's spec verbatim, making the real output sharded. A guard that consulted only the input and
    mem_config_actual therefore skipped recording the output's shape in pages, and the accessor-path
    collision reopened on this path even though the general case was fixed.

    Nothing requires the two to agree: the binary_ng invoke template forwards memory_config and
    output_tensor to the prim independently, and validate_on_program_cache_miss does not cross-check
    them. So the guard tests the output tensor separately."""
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_spec = ttnn.ShardSpec(core_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    def scalar_add_into(shape, seed):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_out = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded
        )
        with device.cache_entries_counter.measure():
            # Interleaved memory_config, sharded output -- deliberately disagreeing.
            tt_c = ttnn.add(tt_a, 1.5, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=tt_out)
        return a + 1.5, ttnn.to_torch(tt_c)

    ref_small, out_small = scalar_add_into([1, 1, 64, 128], 0)
    assert_with_pcc(ref_small, out_small, 0.999)

    ref_big, out_big = scalar_add_into([1, 1, 128, 128], 1)
    assert_with_pcc(ref_big, out_big, 0.999)

    assert device.cache_entries_counter.total == 2


def test_ng_scalar_dram_sharded_cache_miss_across_page_counts(device, isolate_program_cache):
    """REGRESSION: on the accessor path the cache key must carry each sharded operand's shape in pages.

    A DRAM-sharded input takes that path, so with one ShardSpec held constant these two shapes share
    every other key field while needing different accessor geometry:

        [1,1, 64,128] -> 2 shards,  8 pages
        [1,1,128,128] -> 4 shards, 16 pages

    Pre-fix the second call was a cache hit returning PCC 0.179 with exactly half the output wrong.

    ORDER MATTERS -- do not reverse these calls. Big-then-small passes vacuously, because every page id
    of the smaller tensor already falls below the larger baked radix. Only small-then-big exposes it.

    Requires the DRAM routing fix to reach the accessor at all; without it the config throws instead."""
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_spec = ttnn.ShardSpec(core_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def scalar_add(shape, seed):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        with device.cache_entries_counter.measure():
            tt_c = ttnn.add(tt_a, 1.5)
        return a + 1.5, ttnn.to_torch(tt_c)

    ref_small, out_small = scalar_add([1, 1, 64, 128], 0)
    assert_with_pcc(ref_small, out_small, 0.999)

    # Cache HIT today: same ShardSpec and dtype, and the accessor path leaves the key's only
    # shape-derived members unset, so nothing distinguishes 8 pages from 16.
    ref_big, out_big = scalar_add([1, 1, 128, 128], 1)
    assert_with_pcc(ref_big, out_big, 0.999)

    assert device.cache_entries_counter.total == 2


@pytest.mark.parametrize(
    "out_dtype",
    [
        ttnn.bfloat16,
        pytest.param(
            ttnn.float32,
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/tenstorrent/tt-metal/issues/54138 -- ttnn.where with a preallocated "
                "float32 output silently ignores the "
                "predicate and returns t_true. Reproduces on a SINGLE call against a cold program cache, "
                "so this is a plain correctness bug rather than the cache-key collision finding #3 "
                "describes. Unfixed; see the docstring. strict so that fixing it fails here and forces "
                "this marker to be removed rather than lingering as a silent XPASS.",
            ),
        ),
    ],
    ids=["out_bf16", "out_f32"],
)
def test_ng_where_scalar_preallocated_output_dtype(device, isolate_program_cache, out_dtype):
    """Issue #54138 finding #3 predicted a CACHE-HIT defect: a caller-supplied output tensor reaches the
    key only by proxy through attributes.dtype, which where_operation_with_scalar leaves as std::nullopt,
    so get_dtype() collapses to the INPUT dtype and the real output dtype never enters the key. The
    finding was tiered "Blocking (minor)" on the premise that "a single call with the odd value works
    correctly" and only the cache hit goes wrong.

    Measured on Wormhole, that premise does not hold. With a preallocated FLOAT32 output and bfloat16
    inputs, a single call against a freshly-cleared program cache already returns the wrong answer --
    2022 of 2048 elements mismatched, and the output is approximately t_true, i.e. the predicate is
    dropped entirely. The bfloat16-output case is exact (0/2048) under the same conditions.

    So this domain is broken with or without a cache, which by the issue's own cache-dependence test
    makes it a report rather than a port blocker: adding the output dtype to the key would only give
    each dtype its own separately-wrong program. Adding it also costs real cache reuse, because hashing
    the output tensor's presence stops in-place and out-of-place calls from sharing one entry (it breaks
    test_ng_cache_mixed_inplace_outofplace_interleaved). The underlying correctness bug must be fixed
    first; only then is a key entry meaningful.

    This test is parametrized so the passing bfloat16 case pins the behavior that DOES work, and the
    xfail marks the float32 case that does not."""
    shape = [1, 1, 32, 64]
    torch.manual_seed(0)

    pred = (torch.rand(shape) > 0.5).to(torch.bfloat16)
    t_true = torch.rand(shape, dtype=torch.bfloat16)
    scalar_false = 0.5

    tt_pred = ttnn.from_torch(pred, layout=ttnn.TILE_LAYOUT, device=device)
    tt_true = ttnn.from_torch(t_true, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32), dtype=out_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    res = ttnn.where(tt_pred, tt_true, scalar_false, output_tensor=tt_out)

    ref = torch.where(pred.bool(), t_true.float(), torch.full(shape, scalar_false))
    assert_with_pcc(ref, ttnn.to_torch(res).float(), 0.999)


@pytest.mark.parametrize("op", [ttnn.add, ttnn.subtract, ttnn.multiply, ttnn.div])
def test_scalar_tensor_scalar_value_excluded_from_hash(device, isolate_program_cache, op):
    """The mirrored scalar reaches the kernel as a runtime arg, so its value must not key
    the cache -- only the operand side does."""
    shape = (1, 1, 320, 384)
    torch_a = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)

    op(1.5, tt_a)
    ttnn.synchronize_device(device)
    before = device.num_program_cache_entries()

    for scalar in [2.0, 2.5, 3.0, 4.0, 5.5, 6.25]:
        result = ttnn.to_torch(op(scalar, tt_a))
        assert_with_pcc(_torch_scalar_op(op)(scalar, torch_a), result, 0.999)

    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == before


@pytest.mark.parametrize("op", [ttnn.subtract, ttnn.div])
def test_scalar_side_is_in_hash(device, isolate_program_cache, op):
    """The two operand orders compile different kernels, so they must not share an entry --
    sharing one would hand back the un-mirrored result on the second call."""
    shape = (1, 1, 320, 384)
    torch_a = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    scalar = 2.0

    tensor_first = ttnn.to_torch(op(tt_a, scalar))
    ttnn.synchronize_device(device)
    after_first = device.num_program_cache_entries()

    scalar_first = ttnn.to_torch(op(scalar, tt_a))
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == after_first + 1
    assert_with_pcc(_torch_scalar_op(op)(scalar, torch_a), scalar_first, 0.999)
    assert_with_pcc(_torch_tensor_op(op)(torch_a, scalar), tensor_first, 0.999)


def _torch_scalar_op(op):
    return {
        ttnn.add: lambda s, t: s + t,
        ttnn.subtract: lambda s, t: s - t,
        ttnn.multiply: lambda s, t: s * t,
        ttnn.div: lambda s, t: s / t,
    }[op]


def _torch_tensor_op(op):
    return {
        ttnn.add: lambda t, s: t + s,
        ttnn.subtract: lambda t, s: t - s,
        ttnn.multiply: lambda t, s: t * s,
        ttnn.div: lambda t, s: t / s,
    }[op]

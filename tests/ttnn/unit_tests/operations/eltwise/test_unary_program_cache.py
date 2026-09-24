# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for eltwise unary program cache behavior.

Tests target potential caching issues.
The unary operation uses 3 ProgramFactory variants:
  - UnaryProgramFactory (interleaved)
  - UnarySubCoreGridProgramFactory (explicit sub_core_grids)
  - UnaryShardedProgramFactory (sharded input)

compute_program_hash() hashes:
  TILE layout:  args, input tensor_layout, output tensor_layout,
                input/output resolved sharding geometry, src/dst shard volumes
  ROW_MAJOR:    args, input tensor_layout, output tensor_layout, padded_shape,
                input/output resolved sharding geometry, src/dst shard volumes

Where args = entire operation_attributes_t via to_hash() (op_chain, output_dtype,
memory_config, fp32_dest_acc_en, preserve_fp32_precision, bfp8_pack_precise,
sub_core_grids, worker_grid), and tensor_layout is (dtype, page_config -- and so the
Tile -- memory_config, alignment). The whole tensor_layout is hashed rather than its
parts because a Metal 2.0 TensorParameter relaxation requires it to be exactly equal
and cannot relax any component of it.

For sharded tensors, the cache key uses the squeezed shard shape in pages and the resolved
core list from each TensorSpec's BufferDistributionSpec (compute_buffer_sharding_args). The
tensor bindings bake that geometry into TensorAccessor compile-time args, and the relaxed
TensorParameter match compares it exactly. Since neither of these can be determined from the
shard spec, the tensor and shard shapes are squeezed together and for GRID_2D the bank list is
trimmed from the unsqueezed shape. This gives different geometries for the same shard spec but
with different shapes.

On a program-cache HIT the program is NOT rebuilt; override_runtime_arguments()
returns ALL per-dispatch state for the current tensors, built by the same run-args
builder create_program_artifacts() uses — every per-core work-split arg (tile counts,
start ids) and packed scalars, plus both tensor arguments. The tensor arguments carry
the reader/writer TensorAccessor base addresses and shape, and the base addresses of
the sharded path's borrowed input/output dataflow buffers.

For TILE layout, volume is NOT hashed, so differently-shaped calls share one
cache entry and override_runtime_arguments re-applies the new shape's work split
on the hit. The one exception is an interleaved TILE tensor overpadded past its
tile boundary: its Alignment resolves to the padded H/W rather than the tile dims,
so those tensors do key on padded shape and fragment per shape. Because addresses are re-derived from the actual current tensors
(never inferred from Buffer* identity), in-place (out=x) and mixed in-place/
out-of-place reuse of one cache entry stay correct by construction.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_ulp


def run_unary_op(device, op, shape, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run a unary op on device and return (torch_result, ttnn_result).
    The device op is wrapped with cache_entries_counter.measure() so that
    cache_entries_counter.total accumulates only the new entries created."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]

    torch_a = torch.rand(shape, dtype=torch_dtype) + 0.1
    torch_ops = {ttnn.relu: torch.relu, ttnn.sqrt: torch.sqrt, ttnn.abs: torch.abs, ttnn.floor: torch.floor}
    torch_result = torch_ops[op](torch_a)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    with device.cache_entries_counter.measure():
        tt_result = op(tt_a, memory_config=memory_config)
    tt_result = ttnn.to_torch(tt_result)

    return torch_result, tt_result


# =============================================================================
# Cache reuse tests
# =============================================================================


def test_unary_cache_reuse_same_config(device):
    """Same op, same shape, same dtype run twice -> 1 cache entry, different outputs."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 32, 64]

    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
    assert_equal(torch_ref1, tt_out1)

    torch.manual_seed(42)
    torch_ref2, tt_out2 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
    assert_equal(torch_ref2, tt_out2)

    assert device.cache_entries_counter.total == 1
    assert not torch.equal(tt_out1, tt_out2)


def test_unary_cache_reuse_same_volume_different_shapes(device):
    """TILE layout: same volume, different shapes -> 1 cache entry.
    unary doesn't hash volume or shape; tile counts are runtime args,
    so any shape with the same op/dtype/memory_config shares one entry."""
    device.cache_entries_counter.reset()

    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, [1, 1, 32, 64], dtype=ttnn.float32)
    assert_equal(torch_ref1, tt_out1)

    torch_ref2, tt_out2 = run_unary_op(device, ttnn.relu, [1, 1, 64, 32], dtype=ttnn.float32)
    assert_equal(torch_ref2, tt_out2)

    assert device.cache_entries_counter.total == 1


def test_unary_cache_reuse_different_volumes(device):
    """TILE layout: different volumes -> still 1 cache entry.
    unary uses runtime tile counts (not compile-time), so different volumes
    share the same compiled program. override_runtime_arguments handles the
    different per-core tile distributions on cache hit."""
    device.cache_entries_counter.reset()

    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, [1, 1, 32, 32], dtype=ttnn.float32)
    assert_equal(torch_ref1, tt_out1)

    torch_ref2, tt_out2 = run_unary_op(device, ttnn.relu, [1, 1, 64, 64], dtype=ttnn.float32)
    assert_equal(torch_ref2, tt_out2)

    assert device.cache_entries_counter.total == 1


# =============================================================================
# Cache miss tests (fields correctly included in hash)
# =============================================================================


def test_unary_cache_miss_different_op_types(device):
    """Different unary op types -> different cache entries."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
    assert_equal(torch_ref1, tt_out1)

    torch_ref2, tt_out2 = run_unary_op(device, ttnn.sqrt, shape, dtype=ttnn.float32)
    assert_with_ulp(expected_result=torch_ref2, actual_result=tt_out2, ulp_threshold=1)

    assert device.cache_entries_counter.total == 2


def test_unary_cache_miss_different_input_dtypes(device):
    """Different input dtypes -> different cache entries."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.bfloat16)
    assert_equal(torch_ref1, tt_out1)

    torch_ref2, tt_out2 = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
    assert_equal(torch_ref2, tt_out2)

    assert device.cache_entries_counter.total == 2


def test_unary_cache_miss_different_memory_configs(device):
    """Different memory configs -> different cache entries."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_unary_op(
        device, ttnn.relu, shape, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert_equal(torch_ref1, tt_out1)

    torch_ref2, tt_out2 = run_unary_op(
        device, ttnn.relu, shape, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    assert_equal(torch_ref2, tt_out2)

    assert device.cache_entries_counter.total == 2


def test_unary_cache_miss_different_alignments(device):
    """Same dtype, logical shape and memory config. Differing padded shape -> different cache entries."""
    device.cache_entries_counter.reset()
    logical = [1, 1, 32, 32]

    for padded in ([1, 1, 64, 64], [1, 1, 96, 96]):
        torch.manual_seed(0)
        torch_a = torch.rand(logical, dtype=torch.bfloat16) + 0.1
        unpadded = ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_a = unpadded.pad(padded, [0] * len(logical), 0.0).to(ttnn.TILE_LAYOUT).to(device)
        assert tt_a.padded_shape == ttnn.Shape(padded)
        assert tt_a.shape == ttnn.Shape(logical)

        with device.cache_entries_counter.measure():
            tt_out = ttnn.relu(tt_a)
        assert_equal(torch.relu(torch_a), ttnn.to_torch(tt_out))

    assert device.cache_entries_counter.total == 2


def test_unary_cache_miss_different_sub_core_grids(device):
    """Different sub_core_grids -> different cache entries.
    sub_core_grids is part of UnaryParams (hashed via args) and also hashed explicitly.
    Uses ttnn.floor which supports sub_core_grids parameter."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 32, 64]

    torch_a1 = torch.rand(shape, dtype=torch.float32) + 0.5
    torch_ref1 = torch.floor(torch_a1)
    grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    tt_a1 = ttnn.from_torch(torch_a1, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out1 = ttnn.floor(tt_a1, sub_core_grids=grid_a)
    assert_equal(torch_ref1, ttnn.to_torch(tt_out1))

    torch_a2 = torch.rand(shape, dtype=torch.float32) + 0.5
    torch_ref2 = torch.floor(torch_a2)
    grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])
    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out2 = ttnn.floor(tt_a2, sub_core_grids=grid_b)
    assert_equal(torch_ref2, ttnn.to_torch(tt_out2))

    assert device.cache_entries_counter.total == 2


def test_unary_cache_miss_different_factories(device):
    """Interleaved vs sub_core_grids factory -> different cache entries.
    factory_index is included in the hash.
    Uses ttnn.floor which supports sub_core_grids parameter."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 32, 64]

    torch_ref1, tt_out1 = run_unary_op(device, ttnn.floor, shape, dtype=ttnn.float32)
    assert_equal(torch_ref1, tt_out1)

    torch_a2 = torch.rand(shape, dtype=torch.float32) + 0.5
    with device.cache_entries_counter.measure():
        torch_ref2 = torch.floor(torch_a2)

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])
    tt_a2 = ttnn.from_torch(torch_a2, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out2 = ttnn.floor(tt_a2, sub_core_grids=grid)
    assert_equal(torch_ref2, ttnn.to_torch(tt_out2))

    assert device.cache_entries_counter.total == 2


# =============================================================================
# Correctness under cache reuse
# =============================================================================


def test_unary_cache_correctness_repeated_runs(device):
    """Run same op 5 times with different data -> all results correct."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 32, 64]
    for _ in range(5):
        torch_ref, tt_out = run_unary_op(device, ttnn.relu, shape, dtype=ttnn.float32)
        assert_equal(torch_ref, tt_out)

    assert device.cache_entries_counter.total == 1


def test_unary_cache_correctness_same_volume_different_shapes(device):
    """Same volume, different shapes all produce correct results via cache reuse."""
    device.cache_entries_counter.reset()
    for shape in [[1, 1, 32, 64], [1, 1, 64, 32]]:
        torch_ref, tt_out = run_unary_op(device, ttnn.sqrt, shape, dtype=ttnn.float32)
        assert_with_ulp(expected_result=torch_ref, actual_result=tt_out, ulp_threshold=1)

    assert device.cache_entries_counter.total == 1


# =============================================================================
# ROW_MAJOR cache tests
# =============================================================================


def test_unary_cache_rm_different_widths_need_separate_entries(device):
    """ROW_MAJOR interleaved tensors with different widths have different page sizes,
    so compute_program_hash must produce distinct keys for each shape."""
    device.cache_entries_counter.reset()
    torch.manual_seed(0)
    torch_a = torch.empty([1, 1, 64, 32], dtype=torch.bfloat16).uniform_(1, 100)
    torch_b = torch.empty([1, 1, 32, 64], dtype=torch.bfloat16).uniform_(1, 100)
    torch_result1 = torch.abs(torch_a)
    torch_result2 = torch.abs(torch_b)
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_result1 = ttnn.abs(tt_a)
        tt_result2 = ttnn.abs(tt_b)
    result1 = ttnn.to_torch(tt_result1)
    result2 = ttnn.to_torch(tt_result2)
    assert torch.equal(result1, torch_result1)
    assert torch.equal(result2, torch_result2)
    assert device.cache_entries_counter.total == 2


# =============================================================================
# Sharded cache tests (GitHub issue #33910)
# =============================================================================


def test_unary_sharded_cache_correctness_different_grids(device):
    """Sharded ttnn.abs with different grid configs must produce correct results.
    Reproduces GitHub issue #33910: ttnn.abs ProgramCache data corruption.
    The (64,64) on 2x2 grid case failed when preceded by other shard configs."""
    device.cache_entries_counter.reset()
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
        with device.cache_entries_counter.measure():
            output = ttnn.abs(input_tensor)
        tt_out = ttnn.to_torch(output)
        torch_ref = torch.abs(torch_tensor)
        assert_equal(torch_ref, tt_out)


@pytest.mark.parametrize("first_inplace", [True, False], ids=["inplace_first", "outofplace_first"])
def test_unary_sharded_mixed_inplace_outofplace(device, first_inplace):
    """REGRESSION (the exact SDXL failure): sharded unary reused across a MIX of in-place
    (output_tensor aliases the input) and out-of-place calls sharing ONE cache entry (same shape/
    config). Under the legacy descriptor path, sharded input/output addresses rode on tensor-backed
    CB base addresses, re-patched only by resolved_bindings.cbs — get_dynamic returned rt-args and
    could not touch CBs. (They are now borrowed DFBs refreshed from the override's tensor arguments.)
    Combined with first-occurrence alias resolution, a program built under one aliasing pattern and
    reused under another mis-resolves the output CB to the wrong slot with nothing to correct it →
    PCC ~0 (SDXL in-place silu). With the parity check built in, this ALSO trips the framework's
    fast-path-vs-rebuild assertion at the exact stale CB. Both orders must stay correct."""
    device.cache_entries_counter.reset()
    shape = (256, 256)
    mem = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=1, y=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    keep_alive = []

    def do(seed, inplace):
        torch.manual_seed(seed)
        t = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_in = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        if inplace:
            out = tt_in
        else:
            out = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mem,
            )
        with device.cache_entries_counter.measure():
            tt_out = ttnn.relu(tt_in, output_tensor=out)
        assert tt_out.buffer_address() == out.buffer_address()
        keep_alive.extend([tt_in, tt_out, out])
        assert_equal(torch.relu(t), ttnn.to_torch(tt_out))

    for i, inplace in enumerate([first_inplace, not first_inplace, first_inplace, not first_inplace]):
        do(i, inplace)
    assert device.cache_entries_counter.total == 1


# =============================================================================
# override_runtime_arguments migration guards (interleaved) — the get_dynamic ->
# override_runtime_arguments port. Every per-dispatch value the custom hash
# EXCLUDES (work split via volume, buffer addresses) must be re-applied on hit.
# =============================================================================


@pytest.mark.parametrize(
    "shape_first, shape_second",
    [
        ([1, 1, 32, 64], [1, 1, 128, 256]),  # grow volume
        ([1, 1, 128, 256], [1, 1, 32, 64]),  # shrink volume
    ],
)
def test_unary_inplace_cache_reuse_different_shapes(device, shape_first, shape_second):
    """MIGRATION GUARD (override_runtime_arguments): interleaved TILE in-place relu (output_tensor
    aliases the input) reused across DIFFERENT logical shapes sharing ONE cache entry (TILE layout
    excludes volume from compute_program_hash). The second call is a cache HIT that reuses the first
    program WITHOUT rebuild; override_runtime_arguments must re-derive every per-core work-split arg
    (per-core tile counts, start_tile_id, and the work-core set itself) for the new shape, or the
    reused program corrupts the result. Interleaved analog of the SDXL in-place silu class of bug."""
    device.cache_entries_counter.reset()

    def inplace_relu(shape, seed):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        with device.cache_entries_counter.measure():
            tt_c = ttnn.relu(tt_a, output_tensor=tt_a)  # in-place
        # Prove the op is ACTUALLY in-place: a fresh-output regression would still pass PCC +
        # single-entry while no longer exercising the alias.
        assert tt_c.buffer_address() == tt_a.buffer_address()
        return torch.relu(a), ttnn.to_torch(tt_c)

    ref1, out1 = inplace_relu(shape_first, 0)
    assert_equal(ref1, out1)

    ref2, out2 = inplace_relu(shape_second, 1)  # cache HIT on the differently-shaped program
    assert_equal(ref2, out2)

    assert device.cache_entries_counter.total == 1  # proves it was a hit, not a rebuild masking the bug


@pytest.mark.parametrize("first_inplace", [True, False], ids=["inplace_first", "outofplace_first"])
def test_unary_cache_mixed_inplace_outofplace_interleaved(device, first_inplace):
    """MIGRATION GUARD (aliased address re-derivation, interleaved): one cached INTERLEAVED program
    reused across a MIX of in-place (output_tensor aliases the input) and out-of-place calls sharing a
    single cache entry (TILE volume excluded from the hash). The legacy resolve_bindings maps an
    aliased buffer to its FIRST occurrence, so a program built under one aliasing pattern and reused
    under another would patch the writer's output address from the wrong tensor slot.
    override_runtime_arguments re-supplies both tensor arguments (reader tensor::src=input, writer
    tensor::dst=output) for the actual current tensors, so it MUST survive both orders — proving the
    address axis is re-applied, not just that a cache hit occurred."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 32, 64]
    keep_alive = []  # hold refs so successive calls see fresh (different) buffer addresses

    def do(seed, inplace):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = tt_a if inplace else None
        with device.cache_entries_counter.measure():
            tt_c = ttnn.relu(tt_a, output_tensor=out)
        if inplace:
            assert tt_c.buffer_address() == tt_a.buffer_address()
        keep_alive.extend([tt_a, tt_c])
        assert_equal(torch.relu(a), ttnn.to_torch(tt_c))

    # Alternate aliasing across the SAME cache entry, in both orders.
    for i, inplace in enumerate([first_inplace, not first_inplace, first_inplace, not first_inplace]):
        do(i, inplace)
    assert device.cache_entries_counter.total == 1


def test_unary_inplace_cache_hit_interleaved_readdresses(device):
    """MIGRATION GUARD (stale buffer address on hit, interleaved): repeated in-place relu at the SAME
    shape/config but with freshly-allocated operands kept alive, so each cache HIT sees a DIFFERENT
    buffer address. override_runtime_arguments must re-supply the reader/writer tensor arguments on
    every hit (no rebuild) or the result reads/writes a stale address."""
    device.cache_entries_counter.reset()
    shape = [1, 1, 64, 64]
    keep_alive = []  # hold refs so each iteration's tensors get fresh (different) addresses
    for i in range(4):
        torch.manual_seed(i)
        a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        with device.cache_entries_counter.measure():
            tt_c = ttnn.relu(tt_a, output_tensor=tt_a)  # in-place
        assert tt_c.buffer_address() == tt_a.buffer_address()
        keep_alive += [tt_a, tt_c]
        assert_equal(torch.relu(a), ttnn.to_torch(tt_c))

    # One shared program reused across all four differently-addressed in-place hits.
    assert device.cache_entries_counter.total == 1


@pytest.mark.parametrize(
    "squeezes_to_rank1, squeezes_to_rank2",
    [([1, 1, 192, 64], [1, 1, 96, 128]), ([1, 1, 320, 64], [1, 1, 64, 320])],
)
def test_unary_cache_miss_nd_sharded_different_geometry(device, squeezes_to_rank1, squeezes_to_rank2):
    """Two ND_SHARDED with NdShardSpec but shapes squeeze to different distribution
    geometry -> different cache entries.

    squeeze_shape_ranks merges a dim into the one right of it while tensor_size == shard_size holds.
    A width of 64 (2 pages) lets the dims merge to rank 1, while a wider one stops the merge at rank 2."""
    device.cache_entries_counter.reset()
    memory_config = ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([1, 1, 64, 64]),
            grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    for seed, shape in enumerate((squeezes_to_rank1, squeezes_to_rank2)):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_a = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        assert tt_a.memory_config().shard_spec is None
        with device.cache_entries_counter.measure():
            tt_out = ttnn.relu(tt_a)
        assert_equal(torch.relu(a), ttnn.to_torch(tt_out))

    assert device.cache_entries_counter.total == 2


@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_unary_cache_miss_width_sharded_different_geometry(device, buffer_type):
    """Two WIDTH_SHARDED shards with different padded shapes squeeze to different distribution geometry
    -> different cache entries."""
    device.cache_entries_counter.reset()
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    for seed, shape in enumerate(([64, 64], [64, 128])):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_a = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        with device.cache_entries_counter.measure():
            tt_out = ttnn.relu(tt_a)
        assert_equal(torch.relu(a), ttnn.to_torch(tt_out))

    assert device.cache_entries_counter.total == 2


def test_unary_cache_miss_block_sharded_different_cores(device):
    """Same shard geometry, different resolved core list -> different cache entries."""
    device.cache_entries_counter.reset()
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    for seed, shape in enumerate(([64, 128], [64, 192])):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_a = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        with device.cache_entries_counter.measure():
            tt_out = ttnn.relu(tt_a)
        assert_equal(torch.relu(a), ttnn.to_torch(tt_out))

    assert device.cache_entries_counter.total == 2


def test_unary_cache_miss_block_sharded_same_core_count_different_layout(device):
    """Same resolved core COUNT, different core coordinates -> different cache entries.

    The test above separates two shapes by how many banks they span. This one holds the
    count fixed so only the coordinates differ, which is the case a core list hashed by a
    coordinate-collapsing function (e.g. x ^ (y << 1)) would miss. GRID_2D trims the configured
    4x4 grid from the unsqueezed shape, so with one 64x64 shard spec:
        [64, 256]  -> 4 wide, 1 tall -> [(0,0), (1,0), (2,0), (3,0)]
        [128, 128] -> 2 wide, 2 tall -> [(0,0), (1,0), (0,1), (1,1)]
    """
    device.cache_entries_counter.reset()
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    for seed, shape in enumerate(([64, 256], [128, 128])):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_a = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        with device.cache_entries_counter.measure():
            tt_out = ttnn.relu(tt_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert_equal(torch.relu(a), ttnn.to_torch(tt_out))

    assert device.cache_entries_counter.total == 2


@pytest.mark.parametrize(
    "shard_shape, logical_shape1, logical_shape2",
    [
        ((32, 64), [1, 1, 64, 64], [64, 64]),
        ((32, 64), [64, 64], [1, 1, 64, 64]),
        ((64, 64), [1, 1, 128, 64], [1, 1, 96, 64]),
        ((64, 64), [1, 1, 96, 64], [1, 1, 128, 64]),
    ],
)
def test_unary_sharded_input_on_interleaved_path_cache_reuse(device, shard_shape, logical_shape1, logical_shape2):
    """Back-to-back calls whose input is sharded while the op runs its interleaved path must both be
    correct while sharing one cache entry."""
    device.cache_entries_counter.reset()
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    for seed, shape in enumerate((logical_shape1, logical_shape2)):
        torch.manual_seed(seed)
        a = torch.rand(shape, dtype=torch.bfloat16) + 0.1
        tt_a = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        with device.cache_entries_counter.measure():
            tt_out = ttnn.relu(tt_a)
        assert_equal(torch.relu(a), ttnn.to_torch(tt_out))

    assert device.cache_entries_counter.total == 1

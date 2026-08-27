# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Universal I/O coverage for ttnn.repeat_interleave.

Tests the op's transparent handling of:
  * interleaved inputs in L1 and DRAM, and mixed buffer locations (L1 in / DRAM out)
  * ROW_MAJOR and TILE layouts (layout preserved on output)
  * sharded inputs/outputs:
      - native fast-path (ROW_MAJOR 4D, any of N/H/W/C, HEIGHT/WIDTH sharded) -> stays sharded,
        except WIDTH-sharded + dim=C which falls back to interleaved output (still correct)
      - round-trip fallback (TILE, or non-4D tensors)
  * program-cache reuse: same shape -> no rebuild, different shape -> rebuild
  * supported dtype combinations (bf16, bf8_b, uint16/uint8)
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _sharded_cfg(shape, strategy):
    grid = ttnn.CoreGrid(y=1, x=8)
    return ttnn.create_sharded_memory_config(list(shape), grid, strategy, ttnn.ShardOrientation.ROW_MAJOR)


# Interleaved: L1 / DRAM / mixed buffer locations
@pytest.mark.parametrize("in_mc", [L1, DRAM], ids=["in_L1", "in_DRAM"])
@pytest.mark.parametrize("out_mc", [L1, DRAM], ids=["out_L1", "out_DRAM"])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_interleaved_buffer_locations(device, in_mc, out_mc, dim):
    torch_input = torch.rand(1, 1, 32, 32, dtype=torch.bfloat16)
    torch_result = torch.repeat_interleave(torch_input, 2, dim=dim)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mc)
    out = ttnn.repeat_interleave(x, 2, dim=dim, memory_config=out_mc)
    assert out.memory_config().buffer_type == out_mc.buffer_type
    assert_equal(torch_result, ttnn.to_torch(out))


# Layout: ROW_MAJOR and TILE, preserved on output
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_layout_preserved(device, layout, dim):
    torch_input = torch.rand(1, 1, 32, 32, dtype=torch.bfloat16)
    torch_result = torch.repeat_interleave(torch_input, 3, dim=dim)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device)
    out = ttnn.repeat_interleave(x, 3, dim=dim)
    assert out.layout == layout
    assert_equal(torch_result, ttnn.to_torch(out))


# Dtype coverage x layout
@pytest.mark.parametrize(
    "dtype, torch_dtype, pcc",
    [
        (ttnn.bfloat16, torch.bfloat16, None),
        (ttnn.uint16, torch.int16, None),
        (ttnn.bfloat8_b, torch.bfloat16, 0.99),
        (ttnn.bfloat4_b, torch.bfloat16, 0.9),
        (ttnn.uint8, torch.int32, None),
    ],
    ids=["bf16", "uint16", "bf8_b", "bf4_b", "uint8"],
)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_dtype_coverage(device, dtype, torch_dtype, pcc, dim):
    if torch_dtype in (torch.int16, torch.int32):
        torch_input = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch_dtype)
    else:
        torch_input = torch.rand(1, 1, 32, 32, dtype=torch_dtype)
    torch_result = torch.repeat_interleave(torch_input, 2, dim=dim)
    x = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.repeat_interleave(x, 2, dim=dim)
    assert out.dtype == dtype
    if pcc is None:
        assert_equal(torch_result, ttnn.to_torch(out))
    else:
        assert_with_pcc(torch_result.float(), ttnn.to_torch(out).float(), pcc)


# dim / rank: any dim in [-rank, rank) for tensors of arbitrary rank (negative dims, ranks 2/3/5)
@pytest.mark.parametrize("repeats", [2, 3])
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((1, 1, 32, 32), -1),
        ((1, 1, 32, 32), -2),
        ((1, 1, 32, 32), -4),
        ((2, 3, 4, 32, 32), 0),  # rank 5
        ((2, 3, 4, 32, 32), 2),  # rank 5, middle dim
        ((2, 3, 4, 32, 32), 4),  # rank 5, last dim
        ((2, 3, 4, 32, 32), -1),  # rank 5, negative last dim
        ((3, 32, 32), 0),  # rank 3
        ((32, 32), 0),  # rank 2
        ((8,), 0),  # rank 1: exercises the unsqueeze -> recurse -> reshape special case
        ((8,), -1),  # rank 1, negative dim (same as dim=0, only valid dim for rank 1)
    ],
)
def test_dim_and_rank(device, repeats, shape, dim):
    torch_input = torch.rand(*shape, dtype=torch.bfloat16)
    torch_result = torch.repeat_interleave(torch_input, repeats, dim=dim)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.repeat_interleave(x, repeats, dim=dim)
    assert_equal(torch_result, ttnn.to_torch(out))


# Small / non-tile-aligned shapes (e.g. [2,2]) in both layouts
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "shape, repeats, dim",
    [
        ((2, 2), 2, 0),
        ((2, 2), 2, 1),
        ((2, 2), 3, 0),
        ((4, 6), 2, 0),
        ((5, 7), 2, 0),
    ],
)
def test_small_shapes(device, layout, shape, repeats, dim):
    torch_input = torch.rand(*shape, dtype=torch.bfloat16)
    torch_result = torch.repeat_interleave(torch_input, repeats, dim=dim)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device)
    out = ttnn.repeat_interleave(x, repeats, dim=dim)
    assert_equal(torch_result, ttnn.to_torch(out))


# Sharded native fast-path (ROW_MAJOR 4D, all 4 dims via ttnn.upsample) -> stays sharded for H/W/N,
# and for C under HEIGHT sharding; C under WIDTH sharding falls back to interleaved output (still
# correct) because the transpose used to route C through upsample isn't natively sharded for that combo.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("repeats", [2, 3])
def test_sharded_native_hw(device, strategy, dim, repeats):
    shape = (1, 32, 8, 16)  # [N, H, W, C]
    torch_input = torch.rand(*shape, dtype=torch.bfloat16)
    torch_result = torch.repeat_interleave(torch_input, repeats, dim=dim)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    x = ttnn.to_memory_config(x, _sharded_cfg(shape, strategy))
    out = ttnn.repeat_interleave(x, repeats, dim=dim)
    assert out.memory_config().is_sharded()  # native path keeps it sharded
    assert_equal(torch_result, ttnn.to_torch(out))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH])
@pytest.mark.parametrize("repeats", [2, 3])
def test_sharded_native_n(device, strategy, repeats):
    shape = (2, 32, 8, 16)  # [N, H, W, C], N > 1 to exercise dim=0
    torch_input = torch.rand(*shape, dtype=torch.bfloat16)
    torch_result = torch.repeat_interleave(torch_input, repeats, dim=0)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    x = ttnn.to_memory_config(x, _sharded_cfg(shape, strategy))
    out = ttnn.repeat_interleave(x, repeats, dim=0)
    assert out.memory_config().is_sharded()
    assert_equal(torch_result, ttnn.to_torch(out))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH])
@pytest.mark.parametrize("repeats", [2, 3])
def test_sharded_native_c(device, strategy, repeats):
    shape = (1, 32, 8, 16)  # [N, H, W, C]
    torch_input = torch.rand(*shape, dtype=torch.bfloat16)
    torch_result = torch.repeat_interleave(torch_input, repeats, dim=3)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    x = ttnn.to_memory_config(x, _sharded_cfg(shape, strategy))
    out = ttnn.repeat_interleave(x, repeats, dim=3)
    if strategy == ttnn.ShardStrategy.HEIGHT:
        assert out.memory_config().is_sharded()
    # WIDTH: transpose(2,3) isn't natively sharded for this combo, so it correctly falls back to
    # interleaved output; only correctness is asserted below, not sharded-ness.
    assert_equal(torch_result, ttnn.to_torch(out))


# Sharded round-trip fallback (TILE input, or explicit sharded output)
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("sharded_output", [False, True])
def test_sharded_fallback(device, strategy, dim, sharded_output):
    shape = (1, 1, 256, 512) if strategy == ttnn.ShardStrategy.WIDTH else (1, 1, 256, 64)
    torch_input = torch.rand(*shape, dtype=torch.bfloat16)
    torch_result = torch.repeat_interleave(torch_input, 2, dim=dim)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x = ttnn.to_memory_config(x, _sharded_cfg(shape, strategy))
    out_mc = _sharded_cfg(torch_result.shape, strategy) if sharded_output else None
    out = ttnn.repeat_interleave(x, 2, dim=dim, memory_config=out_mc)
    assert out.memory_config().is_sharded() == sharded_output
    assert_equal(torch_result, ttnn.to_torch(out))


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


# Program-cache: same shape reuses (no new entry), different shape rebuilds (new entry); results
# stay correct throughout. Asserts actual cache size, not just numerical correctness, so a
# regression in cache-hit/rebuild behavior fails this test even if results still happen to match.
def test_program_cache_reuse(device, isolate_program_cache):
    def run(shape, repeats, dim):
        torch_input = torch.rand(*shape, dtype=torch.bfloat16)
        torch_result = torch.repeat_interleave(torch_input, repeats, dim=dim)
        x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.repeat_interleave(x, repeats, dim=dim)
        assert_equal(torch_result, ttnn.to_torch(out))

    assert device.num_program_cache_entries() == 0, "Program cache should be empty before the test"

    run((1, 1, 32, 32), 2, 2)
    after_first = device.num_program_cache_entries()
    assert after_first > 0, "Expected at least one program cache entry after the first call"

    run((1, 1, 32, 32), 2, 2)  # same shape/repeats/dim -> cache reuse, no new entry
    assert device.num_program_cache_entries() == after_first, (
        f"Expected cache reuse (still {after_first} entries), " f"got {device.num_program_cache_entries()}"
    )

    run((1, 1, 64, 32), 2, 2)  # different shape -> rebuild, new entry
    after_reshape = device.num_program_cache_entries()
    assert (
        after_reshape > after_first
    ), f"Expected a new cache entry after a shape change (>{after_first}), got {after_reshape}"

    run((1, 1, 32, 32), 3, 1)  # different repeats/dim -> rebuild, new entry
    assert device.num_program_cache_entries() > after_reshape, (
        f"Expected a new cache entry after a repeats/dim change (>{after_reshape}), "
        f"got {device.num_program_cache_entries()}"
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for two program-cache hazards in the Metal 2.0 concat program factories.

1. Input aliasing (silent wrong results).
   The Metal 2.0 spec-factory cache-hit path binds each input to an argument position by first
   MeshTensor-address match and freezes that table into the cache entry, so concat([x, x]) records
   both inputs at position 0. If concat([x, x]) and concat([a, b]) shared one entry, the later
   concat([a, b]) would replay that table, read `a` twice and never touch `b` -- wrong data, no error.
   concat's compute_program_hash folds in an input-aliasing signature (for each input, the index of
   the first input backed by the same MeshTensor), so aliased and distinct call patterns land in
   SEPARATE cache entries. The regression guard below is therefore: a call whose aliasing pattern
   differs from the one that primed the entry must be a cache MISS that builds its own entry and
   binds every input, and re-running that same pattern must be a HIT that still returns correct data.

2. Single partial shard (loud failure).
   A height-sharded tensor whose total row count is smaller than one shard is a valid TensorSpec.
   The sharded two-tensor factories size their borrowed L1 buffers from the shard; the spec-time
   borrowed-DFB size check must validate that against the per-bank allocation (not the tensor's
   packed size), or it wrongly rejects the call. Guarded here so that fix does not regress.
"""

import pytest
import torch

import ttnn


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def _grid(num_cores):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})


def _height_sharded(num_cores, shard_shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_grid(num_cores), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


# One row per concat program factory reachable with two inputs of identical spec.
ALIAS_CASES = {
    "interleaved_tile_dim0": dict(
        shape=(1, 1, 64, 64),
        layout=ttnn.TILE_LAYOUT,
        dim=0,
        in_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.DRAM_MEMORY_CONFIG,
    ),
    "interleaved_rm_dim2": dict(
        shape=(1, 1, 32, 64),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dim=2,
        in_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.DRAM_MEMORY_CONFIG,
    ),
    "height_sharded_rm_dim3": dict(
        shape=(1, 1, 64, 64),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dim=3,
        in_mem=_height_sharded(2, (32, 64)),
        out_mem=_height_sharded(2, (32, 128)),
    ),
    "height_sharded_tile_dim3": dict(
        shape=(1, 1, 64, 64),
        layout=ttnn.TILE_LAYOUT,
        dim=3,
        in_mem=_height_sharded(2, (32, 64)),
        out_mem=_height_sharded(2, (32, 128)),
    ),
}


def _to_device(device, torch_tensor, case):
    return ttnn.from_torch(torch_tensor, layout=case["layout"], device=device, memory_config=case["in_mem"])


def _run_concat(device, torch_inputs, tt_inputs, case, context):
    """Run concat on tt_inputs and assert it matches torch.cat(torch_inputs)."""
    out = ttnn.concat(tt_inputs, dim=case["dim"], memory_config=case["out_mem"])
    got = ttnn.to_torch(out)
    expected = torch.cat(torch_inputs, dim=case["dim"])
    if not torch.equal(got, expected):
        # Diagnose the aliasing signature: every slot filled from the first input.
        first_only = torch.cat([torch_inputs[0]] * len(torch_inputs), dim=case["dim"])
        pytest.fail(
            f"{context}: wrong result; output equals concat of the first input repeated: "
            f"{torch.equal(got, first_only)}"
        )


def _assert_distinct_pattern_miss_then_hit(device, torch_inputs, tt_inputs, case, entries_before, context):
    """
    A call whose aliasing pattern differs from the one that primed the shared spec must not reuse
    that entry: it is a cache miss that builds its own entry (entries += 1) and binds every input
    (correct data). Re-running the identical pattern is a cache hit (no new entry) and still correct.
    """
    _run_concat(device, torch_inputs, tt_inputs, case, context)
    entries_after = device.num_program_cache_entries()
    assert entries_after == entries_before + 1, (
        f"{context}: expected a cache MISS building a distinct entry "
        f"(entries {entries_before} -> {entries_before + 1}), got {entries_after}. "
        "A different aliasing pattern must not reuse the primed entry."
    )
    _run_concat(device, torch_inputs, tt_inputs, case, f"{context} repeated")
    assert (
        device.num_program_cache_entries() == entries_after
    ), f"{context} repeated: expected a cache HIT (no new entry) for the identical aliasing pattern"


@pytest.mark.parametrize("case_name", ALIAS_CASES.keys())
def test_alias_then_distinct_inputs(device, isolate_program_cache, case_name):
    """concat([x, x]) primes the entry; concat([a, b]) is a distinct pattern -> own entry, reads both."""
    case = ALIAS_CASES[case_name]
    torch.manual_seed(0)

    x_t = torch.randn(case["shape"]).bfloat16()
    x = _to_device(device, x_t, case)
    _run_concat(device, [x_t, x_t], [x, x], case, "priming call concat([x, x])")
    entries_after_prime = device.num_program_cache_entries()

    a_t = torch.randn(case["shape"]).bfloat16()
    b_t = torch.randn(case["shape"]).bfloat16()
    a = _to_device(device, a_t, case)
    b = _to_device(device, b_t, case)
    _assert_distinct_pattern_miss_then_hit(
        device, [a_t, b_t], [a, b], case, entries_after_prime, "concat([a, b]) after concat([x, x])"
    )


@pytest.mark.parametrize("case_name", ALIAS_CASES.keys())
def test_distinct_then_alias_inputs(device, isolate_program_cache, case_name):
    """Reverse order: concat([a, b]) primes the entry; concat([x, x]) is a distinct pattern -> own entry."""
    case = ALIAS_CASES[case_name]
    torch.manual_seed(0)

    a_t = torch.randn(case["shape"]).bfloat16()
    b_t = torch.randn(case["shape"]).bfloat16()
    _run_concat(device, [a_t, b_t], [_to_device(device, a_t, case), _to_device(device, b_t, case)], case, "priming")
    entries_after_prime = device.num_program_cache_entries()

    x_t = torch.randn(case["shape"]).bfloat16()
    x = _to_device(device, x_t, case)
    _assert_distinct_pattern_miss_then_hit(
        device, [x_t, x_t], [x, x], case, entries_after_prime, "concat([x, x]) after concat([a, b])"
    )


def test_partial_alias_three_inputs(device, isolate_program_cache):
    """concat([x, y, x]) then concat([a, b, c]): distinct pattern -> own entry, no stale [a, b, a]."""
    case = ALIAS_CASES["interleaved_tile_dim0"]
    torch.manual_seed(0)

    x_t = torch.randn(case["shape"]).bfloat16()
    y_t = torch.randn(case["shape"]).bfloat16()
    x = _to_device(device, x_t, case)
    y = _to_device(device, y_t, case)
    _run_concat(device, [x_t, y_t, x_t], [x, y, x], case, "priming call concat([x, y, x])")
    entries_after_prime = device.num_program_cache_entries()

    a_t, b_t, c_t = (torch.randn(case["shape"]).bfloat16() for _ in range(3))
    a, b, c = (_to_device(device, t, case) for t in (a_t, b_t, c_t))
    _assert_distinct_pattern_miss_then_hit(
        device, [a_t, b_t, c_t], [a, b, c], case, entries_after_prime, "concat([a, b, c]) after concat([x, y, x])"
    )


@pytest.mark.parametrize("num_cores", [1, 2])
@pytest.mark.parametrize(
    "layout, shape, in_shard, out_shard",
    [
        (ttnn.ROW_MAJOR_LAYOUT, (1, 1, 30, 64), (32, 64), (32, 128)),
        (ttnn.TILE_LAYOUT, (1, 1, 32, 64), (64, 64), (64, 128)),
    ],
    ids=["rm_30_rows_shard_32", "tile_32_rows_shard_64"],
)
def test_single_partial_shard(device, layout, shape, in_shard, out_shard, num_cores):
    """A height-sharded tensor smaller than one shard must be accepted by the sharded factories."""
    torch.manual_seed(0)
    a_t = torch.randn(shape).bfloat16()
    b_t = torch.randn(shape).bfloat16()
    in_mem = _height_sharded(num_cores, in_shard)
    a = ttnn.from_torch(a_t, layout=layout, device=device, memory_config=in_mem)
    b = ttnn.from_torch(b_t, layout=layout, device=device, memory_config=in_mem)

    out = ttnn.concat([a, b], dim=3, memory_config=_height_sharded(num_cores, out_shard))
    assert torch.equal(ttnn.to_torch(out), torch.cat([a_t, b_t], dim=3))

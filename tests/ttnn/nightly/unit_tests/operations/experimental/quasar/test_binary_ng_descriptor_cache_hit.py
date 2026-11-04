# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Program-cache-hit coverage for an interleaved quasar binary_ng ADD.

`test_binary_ng_resnet_add.py::test_resnet_add_program_cache_hit` only exercises the fully-sharded
in-place ADD config, whose operands are borrowed. Here an interleaved (DRAM or L1) tensor-tensor ADD
runs several times with the program cache enabled. That config runs the Quasar-native factory on Quasar
and the Metal 2.0 factory on Wormhole and Blackhole. Both bind each operand's buffer to the kernels, so
a cache hit must read every operand at its current address. Each dispatch allocates fresh inputs and
holds every tensor alive, so it lands on DISTINCT buffer addresses. The 2nd+ dispatches are genuine
cache hits (asserted via the entry count), and PCC must still pass on every dispatch: a program that
kept the first dispatch's addresses would read stale data and give a wrong result.

Run:
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_descriptor_cache_hit.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# bf16 accuracy; never weaken below this.
_PCC = 0.9997

# Multi-tile, multi-core interleaved shape (2x4 tiles) so the per-core runtime-arg loop that emits the
# a/b/c Buffer* bindings runs on more than one core.
_SHAPE = (2 * 32, 4 * 32)

# Number of dispatches: the first is a cache miss (builds the program), the rest must be cache hits.
_NUM_DISPATCHES = 3


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram_interleaved", "l1_interleaved"],
)
def test_descriptor_interleaved_add_program_cache_hit(device, mem_config):
    torch.manual_seed(0)

    num_before = device.num_program_cache_entries()

    # Hold every tensor alive across the loop so the allocator hands out a DISTINCT address to each new
    # allocation -- this is what forces the cache-hit dispatches to patch the Buffer* bindings to new
    # addresses rather than silently reusing the miss-time address.
    kept_alive = []
    input_addrs = set()
    output_addrs = set()

    for i in range(_NUM_DISPATCHES):
        a_pt = torch.randn(_SHAPE, dtype=torch.bfloat16) + float(i)
        b_pt = torch.randn(_SHAPE, dtype=torch.bfloat16) - float(i)

        a_tt = ttnn.from_torch(
            a_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
        )
        b_tt = ttnn.from_torch(
            b_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
        )

        # Interleaved tensor-tensor ADD: the native factory on Quasar, the Metal 2.0 factory on WH/BH.
        out_tt = ttnn.experimental.quasar.add(a_tt, b_tt, memory_config=mem_config)

        input_addrs.add(a_tt.buffer_address())
        input_addrs.add(b_tt.buffer_address())
        output_addrs.add(out_tt.buffer_address())

        assert_with_pcc(ttnn.to_torch(out_tt), torch.add(a_pt, b_pt), _PCC)
        kept_alive.extend([a_tt, b_tt, out_tt])

    # Every dispatch shared one program: exactly one new cache entry across all of them.
    num_after = device.num_program_cache_entries()
    assert (
        num_after - num_before == 1
    ), f"expected 1 new cache entry across {_NUM_DISPATCHES} dispatches, got {num_after - num_before}"

    # The hit dispatches genuinely ran on reallocated buffers: distinct input and output addresses.
    assert len(input_addrs) == 2 * _NUM_DISPATCHES, f"inputs reused addresses: {sorted(input_addrs)}"
    assert len(output_addrs) == _NUM_DISPATCHES, f"outputs reused addresses: {sorted(output_addrs)}"


# Two interleaved shapes with the SAME volume but a DIFFERENT H/W tile split. `operation_attributes_t`
# carries no shape and `get_shard_volumes` is nullopt for interleaved tensors, so before the hash fix
# BOTH shapes hashed identically -> the second landed on the first's cached program and re-used its
# frozen shape-dependent reader/writer scalars (wrong tile range / OOB). Equal volume specifically
# defeats a volume-only fix, forcing the hash to distinguish the actual dims.
_SHAPE_S1 = (2 * 32, 4 * 32)  # 8 tiles, 1x4 tile grid
_SHAPE_S2 = (4 * 32, 2 * 32)  # 8 tiles, 4x2 tile grid -- same volume, different Ht/Wt


@pytest.mark.parametrize(
    "mem_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram_interleaved", "l1_interleaved"],
)
def test_descriptor_cross_shape_distinct_cache_entries(device, mem_config):
    torch.manual_seed(0)

    def dispatch(shape):
        a_pt = torch.randn(shape, dtype=torch.bfloat16)
        b_pt = torch.randn(shape, dtype=torch.bfloat16)
        a_tt = ttnn.from_torch(
            a_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
        )
        b_tt = ttnn.from_torch(
            b_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config
        )
        out_tt = ttnn.experimental.quasar.add(a_tt, b_tt, memory_config=mem_config)
        assert_with_pcc(ttnn.to_torch(out_tt), torch.add(a_pt, b_pt), _PCC)

    num_before = device.num_program_cache_entries()

    # Prime the cache with shape S1.
    dispatch(_SHAPE_S1)
    num_after_s1 = device.num_program_cache_entries()
    assert num_after_s1 - num_before == 1, f"S1 should add exactly 1 cache entry, got {num_after_s1 - num_before}"

    # Same op, DIFFERENT logical shape S2. With the shape folded into the program hash this must
    # allocate a SECOND, distinct cache entry (and produce a correct S2 result). Without the fix the
    # equal-volume S2 collides onto S1's entry: no new entry and/or a wrong output.
    dispatch(_SHAPE_S2)
    num_after_s2 = device.num_program_cache_entries()
    assert (
        num_after_s2 - num_after_s1 == 1
    ), f"S2 (different shape) must create a distinct cache entry, got {num_after_s2 - num_after_s1} new"

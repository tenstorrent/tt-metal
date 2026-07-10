# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Program-cache-hit coverage for the DESCRIPTOR path of quasar binary_ng.

`test_binary_ng_resnet_add.py::test_resnet_add_program_cache_hit` only exercises the fully-sharded
in-place ADD config, which `BinaryNgDeviceOperation::select_program_factory` routes to the METAL_V2
factory (`matches_metal_v2_slice` -> `ProgramFactoryMetalV2`, variant index 1). It therefore does NOT
cover the descriptor `ProgramFactory::create_descriptor` path (variant index 0), whose reader/writer
runtime args bind the a/b/c buffers as `Buffer*` (via `KernelDescriptor::emplace_runtime_args`) instead
of baking raw `buffer->address()` values. Those bindings are what the descriptor adapter re-resolves on
a program-cache hit so a reallocated input/output is read at its new address.

`matches_metal_v2_slice` requires HEIGHT/BLOCK-sharded L1 tensors, so ANY interleaved config falls
through to the descriptor path — this is provable from the selector source and needs no arch-specific
config. Here we run an interleaved (DRAM or L1) tensor-tensor ADD several times with the program cache
enabled, allocating fresh inputs AND holding every tensor alive so each dispatch lands on DISTINCT
buffer addresses. The 2nd+ dispatches are genuine cache hits (asserted via the entry count), and PCC
must still pass on every dispatch: if the buffer addresses were baked at create-time (the pre-patch
behaviour) instead of re-resolved from the `Buffer*` bindings, the hit dispatches would read the stale
first-dispatch addresses and the result would be wrong.

Run on Wormhole / Blackhole:
    pytest tests/ttnn/unit_tests/operations/experimental/quasar/test_binary_ng_descriptor_cache_hit.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# The descriptor path reaches the same bf16 accuracy as any other; never weaken below this.
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

        # Interleaved tensor-tensor ADD -> not sharded -> select_program_factory returns the descriptor
        # ProgramFactory (variant index 0), NOT ProgramFactoryMetalV2.
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

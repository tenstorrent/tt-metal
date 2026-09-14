# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Program-cache-hit coverage for all four typecast program factories (Metal 2.0).

Typecast was ported from the legacy `ProgramDescriptor` API to Metal 2.0
(`create_program_artifacts`). That changed how a *cache hit* reaches the current tensors:

  - Legacy: the factories pushed `Buffer*` objects as runtime args
    (`KernelDescriptor::emplace_runtime_args`) and the descriptor adapter patched those
    address slots in place on every dispatch.
  - Metal 2.0: the buffers are `TensorParameter`s bound by name; on a hit the adapter calls
    `UpdateTensorArgs` to re-resolve every `TensorArgument` against the *current* tensors. For
    `TypecastShardedProgramFactory` this also re-attaches the borrowed-memory DFBs
    (`DataflowBufferSpec::borrowed_from`), whose backing L1 address comes straight from the
    tensor argument.

Single-invocation correctness tests cannot see a failure in that path: they only ever run the
miss. If a binding were frozen at create-time, the 2nd+ dispatches would read/write the *first*
dispatch's buffers and silently produce stale results.

Each test below dispatches the same config several times with freshly allocated inputs, holding
every tensor alive so the allocator must hand out DISTINCT addresses (asserted). The cache delta
is measured around the typecast call ONLY -- `ttnn.to_torch` on a sharded tensor can itself
dispatch ops, so a global before/after count would be wrong. Exactly one entry across all
dispatches proves the 2nd+ were genuine hits; the per-dispatch golden comparison (inputs differ
every iteration) proves the rebinding actually happened.

One test per factory, selected exactly the way `TypecastDeviceOperation::select_program_factory`
routes:
  - interleaved tiled          -> TypecastProgramFactory
  - tiled + sub_core_grids     -> TypecastSubgridProgramFactory
  - L1-sharded, equal tile size-> TypecastShardedProgramFactory  (borrowed DFBs + output self-loop)
  - L1-sharded, tile mismatch  -> TypecastProgramFactory         (non-optimized-sharded fallback)
  - ROW_MAJOR interleaved      -> TypecastRowMajorChunkedProgramFactory

Run on Wormhole / Blackhole:
    pytest tests/ttnn/unit_tests/operations/eltwise/test_typecast_program_cache.py
"""

import pytest
import torch

import ttnn

# First dispatch is the cache miss that builds the program; the rest must be hits.
_NUM_DISPATCHES = 3

# Multi-tile, multi-core so the per-core runtime-arg loop and the work split both run wide.
_SHAPE = (1, 1, 128, 128)

# 4 cores in a row: 16 tiles / 4 cores divides evenly, which the subgrid factory requires.
_CORE_RANGE_SET = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})


def _height_sharded_config():
    """HEIGHT-sharded L1 config over _CORE_RANGE_SET: 128 rows / 4 cores = 32 rows per shard."""
    shard_spec = ttnn.ShardSpec(_CORE_RANGE_SET, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _run_cache_hit_dispatches(device, make_input, output_dtype, golden, typecast_kwargs):
    """Dispatch typecast _NUM_DISPATCHES times on fresh buffers; assert 1 cache entry + correctness.

    make_input(i) -> (torch_input, tt_input) for dispatch i (data must differ per dispatch, so a
    stale-buffer read cannot coincidentally match the golden).
    golden(torch_input) -> the expected torch tensor, compared exactly (all conversions used here
    are exact: widening bfloat16->float32, or integral float32->int32).
    """
    device.cache_entries_counter.reset()

    # Holding every tensor alive is what forces the allocator to a NEW address each dispatch --
    # without it the freed buffer is handed back and a frozen binding would still look correct.
    kept_alive = []
    input_addrs = set()
    output_addrs = set()

    for i in range(_NUM_DISPATCHES):
        torch_input, tt_input = make_input(i)

        # Measure the typecast alone: to_torch() below may dispatch its own ops on sharded tensors.
        with device.cache_entries_counter.measure():
            tt_output = ttnn.typecast(tt_input, output_dtype, **typecast_kwargs)

        input_addrs.add(tt_input.buffer_address())
        output_addrs.add(tt_output.buffer_address())

        assert tt_output.dtype == output_dtype
        torch_output = ttnn.to_torch(tt_output)
        expected = golden(torch_input)
        assert torch.equal(torch_output, expected), (
            f"wrong result on dispatch {i} "
            f"({'cache miss' if i == 0 else 'cache hit'}) -- a stale binding would look like this"
        )

        kept_alive.extend([tt_input, tt_output])

    assert device.cache_entries_counter.total == 1, (
        f"expected exactly 1 cache entry across {_NUM_DISPATCHES} dispatches "
        f"(1 miss + {_NUM_DISPATCHES - 1} hits), got {device.cache_entries_counter.total}"
    )

    # Prove the hits really did run against reallocated buffers.
    assert len(input_addrs) == _NUM_DISPATCHES, f"inputs reused addresses: {sorted(input_addrs)}"
    assert len(output_addrs) == _NUM_DISPATCHES, f"outputs reused addresses: {sorted(output_addrs)}"


def _bf16_to_fp32_input(device, memory_config, layout):
    """bfloat16 -> float32 is a widening conversion, so the golden comparison is exact."""

    def make_input(i):
        torch_input = torch.randn(_SHAPE, dtype=torch.bfloat16) + float(i)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
            memory_config=memory_config,
        )
        return torch_input, tt_input

    return make_input


@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram_interleaved", "l1_interleaved"],
)
def test_typecast_interleaved_program_cache_hit(device, memory_config):
    """TypecastProgramFactory: interleaved + TILE. Reader/writer rebind input/output per dispatch."""
    torch.manual_seed(0)

    _run_cache_hit_dispatches(
        device,
        make_input=_bf16_to_fp32_input(device, memory_config, ttnn.TILE_LAYOUT),
        output_dtype=ttnn.float32,
        golden=lambda t: t.to(torch.float32),
        typecast_kwargs={"memory_config": memory_config},
    )


def test_typecast_subgrid_program_cache_hit(device):
    """TypecastSubgridProgramFactory: sub_core_grids routes here, one compute spec over all cores."""
    torch.manual_seed(0)

    _run_cache_hit_dispatches(
        device,
        make_input=_bf16_to_fp32_input(device, ttnn.DRAM_MEMORY_CONFIG, ttnn.TILE_LAYOUT),
        output_dtype=ttnn.float32,
        golden=lambda t: t.to(torch.float32),
        typecast_kwargs={"memory_config": ttnn.DRAM_MEMORY_CONFIG, "sub_core_grids": _CORE_RANGE_SET},
    )


def test_typecast_sharded_optimized_program_cache_hit(device):
    """TypecastShardedProgramFactory: the borrowed-memory case.

    float32 -> int32 keeps tile_size equal (4096 both), which is what
    can_use_sharded_optimized_factory requires; L1-sharded in AND out routes here. This is the
    highest-value path in this file: both DFBs are `borrowed_from` a TensorParameter, so a hit has
    to re-attach their backing L1 addresses -- and the output DFB is additionally self-looped.
    Integral float input makes float32 -> int32 exact.
    """
    torch.manual_seed(0)
    sharded_config = _height_sharded_config()

    def make_input(i):
        torch_input = torch.randint(-1000, 1000, _SHAPE, dtype=torch.int32).to(torch.float32) + float(i)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_config,
        )
        return torch_input, tt_input

    _run_cache_hit_dispatches(
        device,
        make_input=make_input,
        output_dtype=ttnn.int32,
        golden=lambda t: t.to(torch.int32),
        typecast_kwargs={"memory_config": sharded_config},
    )


def test_typecast_sharded_fallback_program_cache_hit(device):
    """TypecastProgramFactory as the non-optimized-sharded fallback.

    bfloat16 -> float32 makes tile_size(in) != tile_size(out) (2048 vs 4096), so
    can_use_sharded_optimized_factory returns false and a sharded input falls back here. Its
    TensorParameters then carry *sharded* TensorSpecs through the binding channel, which is the
    least-exercised new path in the port.
    """
    torch.manual_seed(0)
    sharded_config = _height_sharded_config()

    _run_cache_hit_dispatches(
        device,
        make_input=_bf16_to_fp32_input(device, sharded_config, ttnn.TILE_LAYOUT),
        output_dtype=ttnn.float32,
        golden=lambda t: t.to(torch.float32),
        typecast_kwargs={"memory_config": sharded_config},
    )


def test_typecast_row_major_chunked_program_cache_hit(device):
    """TypecastRowMajorChunkedProgramFactory: ROW_MAJOR interleaved routes here."""
    torch.manual_seed(0)

    _run_cache_hit_dispatches(
        device,
        make_input=_bf16_to_fp32_input(device, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT),
        output_dtype=ttnn.float32,
        golden=lambda t: t.to(torch.float32),
        typecast_kwargs={"memory_config": ttnn.DRAM_MEMORY_CONFIG},
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.indexed_entry_copy: one dim-0 entry copied in place, the entry chosen on device."""

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged
from tt_py_test_utils_common.utility_functions import skip_with_llk_assert, skip_with_watcher

# Blackhole only, like the other KDA ops.
pytestmark = run_for_blackhole()

# (dtype, layout, entry shape past dim 0, ND shard of the multi-entry table or None for interleaved): the forms
# the KDA staging uses, an ND-sharded FP32 tile slab with [1, 1, D, 32] shards and an interleaved BF16 row slab.
FORMS = [
    pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, (3, 64, 64), None, id="fp32-tile"),
    pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, (3, 64, 64), [1, 1, 64, 32], id="fp32-tile-nd"),
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (3, 192), None, id="bf16-row-major"),
]
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def _to_device(tensor, device, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(tensor, device=device, dtype=dtype, layout=layout, memory_config=memory_config)


def _index(device, value):
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _random(shape, dtype, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(TORCH_DTYPE[dtype])


def _table_memory(device, nd_shard):
    return ttnn.DRAM_MEMORY_CONFIG if nd_shard is None else _nd_dram(device, nd_shard)


def _nd_dram(device, shard_shape):
    banks = device.dram_grid_size().x
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=shard_shape,
            grid=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(banks)}
            ),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


@pytest.mark.parametrize("dtype, layout, entry_shape, nd_shard", FORMS)
@pytest.mark.parametrize("scale, offset", [(1, 0), (3, 2)], ids=["plain", "scaled"])
def test_load_and_store_selected_entries(device, dtype, layout, entry_shape, nd_shard, scale, offset):
    entries = 7
    table_host = _random((entries, *entry_shape), dtype, 1)
    table = _to_device(table_host, device, dtype, layout, _table_memory(device, nd_shard))
    single = _to_device(torch.zeros(1, *entry_shape, dtype=TORCH_DTYPE[dtype]), device, dtype, layout)
    for value in sorted({0, (entries - 1 - offset) // scale // 2, (entries - 1 - offset) // scale}):
        entry = value * scale + offset
        index = _index(device, value)

        ttnn.experimental.kda.indexed_entry_copy(table, single, index, index_scale=scale, index_offset=offset)
        assert torch.equal(ttnn.to_torch(single), table_host[entry : entry + 1]), f"load entry {entry}"

        update = _random((1, *entry_shape), dtype, 100 + entry)
        source = _to_device(update, device, dtype, layout)
        ttnn.experimental.kda.indexed_entry_copy(source, table, index, index_scale=scale, index_offset=offset)
        table_host[entry : entry + 1] = update
        assert torch.equal(ttnn.to_torch(table), table_host), f"store entry {entry} or a neighbour"
        ttnn.deallocate(source)
        ttnn.deallocate(index)


@pytest.mark.parametrize("dtype, layout, entry_shape, nd_shard", FORMS)
def test_entry_past_the_end_writes_nothing(device, dtype, layout, entry_shape, nd_shard):
    entries = 4
    table_host = _random((entries, *entry_shape), dtype, 2)
    table = _to_device(table_host, device, dtype, layout, _table_memory(device, nd_shard))
    single_host = _random((1, *entry_shape), dtype, 3)
    single = _to_device(single_host, device, dtype, layout)
    # Past the end before scaling; only after scaling; the largest index; and two whose 32-bit entry
    # index * scale + offset would wrap back into range (2 * 0x80000001 -> 2, 1 + 0xFFFFFFFF -> 0).
    for value, scale, offset in [(entries, 1, 0), (1, 2, 2), (0xFFFFFFFF, 1, 0), (2, 0x80000001, 0), (1, 1, 0xFFFFFFFF)]:
        index = _index(device, value)
        ttnn.experimental.kda.indexed_entry_copy(table, single, index, index_scale=scale, index_offset=offset)
        ttnn.experimental.kda.indexed_entry_copy(single, table, index, index_scale=scale, index_offset=offset)
        ttnn.deallocate(index)
    assert torch.equal(ttnn.to_torch(table), table_host)
    assert torch.equal(ttnn.to_torch(single), single_host)

    # A valid index under a scale that would overflow a naive bound still reaches its entry.
    index = _index(device, 0)
    ttnn.experimental.kda.indexed_entry_copy(table, single, index, index_scale=0xFFFFFFFF, index_offset=1)
    assert torch.equal(ttnn.to_torch(single), table_host[1:2])


@pytest.mark.parametrize("dtype, layout, entry_shape, nd_shard", FORMS)
def test_single_entry_on_both_sides_copies_at_entry_zero(device, dtype, layout, entry_shape, nd_shard):
    source_host = _random((1, *entry_shape), dtype, 4)
    source = _to_device(source_host, device, dtype, layout)
    destination_host = torch.zeros(1, *entry_shape, dtype=TORCH_DTYPE[dtype])
    destination = _to_device(destination_host, device, dtype, layout)

    ttnn.experimental.kda.indexed_entry_copy(source, destination, _index(device, 1), index_scale=1, index_offset=0)
    assert torch.equal(ttnn.to_torch(destination), destination_host)
    ttnn.experimental.kda.indexed_entry_copy(source, destination, _index(device, 0), index_scale=1, index_offset=0)
    assert torch.equal(ttnn.to_torch(destination), source_host)


def test_store_into_nd_sharded_matches_fill_cache_for_user(device):
    """Native interleaved tiles into an ND-sharded [1, 1, D, 32] slab land where fill_cache_for_user_ puts them."""
    entries, heads, head_dim = 6, 3, 128
    memory_config = _nd_dram(device, [1, 1, head_dim, 32])
    slab_host = _random((entries, heads, head_dim, head_dim), ttnn.float32, 5)
    reference = _to_device(slab_host, device, ttnn.float32, ttnn.TILE_LAYOUT, memory_config)
    ours = _to_device(slab_host, device, ttnn.float32, ttnn.TILE_LAYOUT, memory_config)
    state_host = _random((1, heads, head_dim, head_dim), ttnn.float32, 6)
    state = _to_device(state_host, device, ttnn.float32, ttnn.TILE_LAYOUT)
    loaded = _to_device(torch.zeros_like(state_host), device, ttnn.float32, ttnn.TILE_LAYOUT)

    for batch in (0, 4, entries - 1):
        ttnn.kv_cache.fill_cache_for_user_(reference, state, batch)
        index = _index(device, batch)
        ttnn.experimental.kda.indexed_entry_copy(state, ours, index, index_scale=1, index_offset=0)
        assert torch.equal(ttnn.to_torch(ours), ttnn.to_torch(reference)), f"batch {batch}"
        ttnn.experimental.kda.indexed_entry_copy(ours, loaded, index, index_scale=1, index_offset=0)
        assert torch.equal(ttnn.to_torch(loaded), state_host), f"load back batch {batch}"
        ttnn.deallocate(index)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
@pytest.mark.parametrize("dtype, layout, entry_shape, nd_shard", FORMS)
def test_one_trace_follows_the_index(device, dtype, layout, entry_shape, nd_shard):
    """A captured load + store pair addresses the entry written into the index before each replay."""
    entries, scale, offset = 6, 2, 1
    table_host = _random((entries, *entry_shape), dtype, 9)
    table = _to_device(table_host, device, dtype, layout, _table_memory(device, nd_shard))
    working = _to_device(torch.zeros(1, *entry_shape, dtype=TORCH_DTYPE[dtype]), device, dtype, layout)
    update = _to_device(torch.zeros(1, *entry_shape, dtype=TORCH_DTYPE[dtype]), device, dtype, layout)
    index = _index(device, 0)

    def step():
        ttnn.experimental.kda.indexed_entry_copy(table, working, index, index_scale=scale, index_offset=offset)
        ttnn.experimental.kda.indexed_entry_copy(update, table, index, index_scale=scale, index_offset=offset)

    step()  # compile before capture: index 0 stores the zero update into entry `offset`
    table_host[offset : offset + 1] = 0
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    step()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for value in (2, 0, 1, 2):
            entry = value * scale + offset
            expected_load = table_host[entry : entry + 1].clone()
            new = _random((1, *entry_shape), dtype, 200 + value)
            for destination, host in ((index, torch.tensor([value], dtype=torch.int64)), (update, new)):
                staged = ttnn.from_torch(
                    host,
                    device=device,
                    dtype=destination.dtype,
                    layout=destination.layout,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(staged, destination)
                ttnn.deallocate(staged)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            table_host[entry : entry + 1] = new
            assert torch.equal(ttnn.to_torch(working), expected_load), f"replay load index {value}"
            assert torch.equal(ttnn.to_torch(table), table_host), f"replay store index {value}"
    finally:
        ttnn.release_trace(device, trace)


def test_rejects_row_major_nd_sharding(device, expect_error):
    table = _to_device(
        torch.zeros(4, 3, 192), device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, _nd_dram(device, [1, 3, 64])
    )
    single = _to_device(torch.zeros(1, 3, 192), device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(RuntimeError, "ND-sharded DRAM for TILE"):
        ttnn.experimental.kda.indexed_entry_copy(table, single, _index(device, 0), index_scale=1, index_offset=0)


@pytest.mark.parametrize(
    "source_shape, destination_shape, message",
    [
        ((2, 32, 32), (2, 32, 32), "must hold a single entry"),
        ((4, 32, 64), (1, 32, 32), "differ past dim 0"),
        ((4, 30, 32), (1, 32, 32), "differ past dim 0"),  # same padded shape, different logical shape
        ((64, 32), (32, 32), "at least 3 for TILE"),
    ],
    ids=["two-multi-entry", "entry-shape", "logical-shape", "rank-2-tile"],
)
def test_rejects(device, expect_error, source_shape, destination_shape, message):
    source = _to_device(torch.zeros(source_shape), device, ttnn.float32, ttnn.TILE_LAYOUT)
    destination = _to_device(torch.zeros(destination_shape), device, ttnn.float32, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, message):
        ttnn.experimental.kda.indexed_entry_copy(source, destination, _index(device, 0), index_scale=1, index_offset=0)


# Kimi-K3's per-chip carry staging at TP4 (24 local heads of 128, three history rows), with two user slots of the
# rank's 18 KDA layers in the slab and the layer at position 5: every KDA layer does each of these once per chunk.
_K3_LAYERS, _K3_SLOTS, _K3_POSITION = 18, 2, 5
_PRODUCTION_PERF_MARGIN = 0.05


@dataclass(frozen=True)
class _ProductionCase:
    case_id: str
    dtype: ttnn.DataType
    layout: ttnn.Layout
    entry_shape: tuple[int, ...]
    nd_shard: list[int] | None
    load: bool
    expected_duration_ns: int


_PRODUCTION_CASES = (
    _ProductionCase("recurrent-store", ttnn.float32, ttnn.TILE_LAYOUT, (24, 128, 128), [1, 1, 128, 32], False, 8_800),
    _ProductionCase("recurrent-load", ttnn.float32, ttnn.TILE_LAYOUT, (24, 128, 128), [1, 1, 128, 32], True, 8_940),
    _ProductionCase("convolution-store", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (144, 192), None, False, 1_940),
    _ProductionCase("convolution-load", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (144, 192), None, True, 1_950),
)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
def test_kimi_k3_production_performance(device, case: _ProductionCase) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for indexed_entry_copy performance checks")
    slab = _to_device(
        torch.zeros(_K3_SLOTS * _K3_LAYERS, *case.entry_shape, dtype=TORCH_DTYPE[case.dtype]),
        device,
        case.dtype,
        case.layout,
        _table_memory(device, case.nd_shard),
    )
    carry_host = _random((1, *case.entry_shape), case.dtype, 10)
    carry = _to_device(carry_host, device, case.dtype, case.layout)
    index = _index(device, 1)
    source, destination = (slab, carry) if case.load else (carry, slab)

    def run() -> ttnn.Tensor:
        return ttnn.experimental.kda.indexed_entry_copy(
            source, destination, index, index_scale=_K3_LAYERS, index_offset=_K3_POSITION
        )

    run()  # warm the program cache so the profiled call measures the cached program
    _, programs = profile_realtime_program_merged(device, run)
    *_, (runtime_id, perf_record) = [
        (runtime_id, program)
        for runtime_id, program in programs.items()
        if any("indexed_entry_copy.cpp" in source for source in program["kernel_sources"])
    ]
    duration_ns = perf_record["duration_ns"]
    if not case.load:
        entry = _K3_LAYERS + _K3_POSITION
        assert torch.equal(ttnn.to_torch(slab)[entry : entry + 1], carry_host), "stored entry"
    # The op reads and writes one entry: the single-entry side, counted once as input and once as output.
    performance = perf_model.performance(
        fpu=perf_model.FpuOps(),
        sfpu=perf_model.SfpuOps(),
        inputs=(carry,),
        outputs=(carry,),
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.LoFi,
    )
    logger.info(
        f"indexed_entry_copy {case.case_id}: measured_ns={duration_ns:.0f}, runtime_id={runtime_id}, "
        f"dram_bytes={performance.work.dram_bytes}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}"
    )
    lower = case.expected_duration_ns * (1 - _PRODUCTION_PERF_MARGIN)
    upper = case.expected_duration_ns * (1 + _PRODUCTION_PERF_MARGIN)
    assert lower <= duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {case.expected_duration_ns} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.indexed_entry_copy: one dim-0 entry copied in place, the entry chosen on device."""

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher

# Blackhole only, like the other KDA ops.
pytestmark = run_for_blackhole()

# (dtype, layout, entry shape past dim 0, ND shard of the multi-entry table or None for interleaved): the forms
# the KDA staging uses, an ND-sharded FP32 tile slab with [1, 1, D, 32] shards and an interleaved BF16 row slab. The
# K3 shape (384 tiles, 17-18 per worker) runs every worker through three staging batches, so staging wraps around.
FORMS = [
    pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, (3, 64, 64), None, id="fp32-tile"),
    pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, (3, 64, 64), [1, 1, 64, 32], id="fp32-tile-nd"),
    pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, (24, 128, 128), None, id="fp32-tile-k3"),
    pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, (24, 128, 128), [1, 1, 128, 32], id="fp32-tile-k3-nd"),
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


def _overwrite(destination, host):
    staged = ttnn.from_torch(
        host,
        device=destination.device(),
        dtype=destination.dtype,
        layout=destination.layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.copy(staged, destination)
    ttnn.deallocate(staged)


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


# (index[0], index_scale, index_offset) -> the entry of a 7-entry table that is copied, or None for a no-op.
_TABLE_ENTRIES = 7
INDEXING = [
    ((0, 1, 0), 0),
    ((3, 1, 0), 3),
    ((6, 1, 0), 6),  # last entry
    ((0, 3, 2), 2),
    ((1, 3, 2), 5),
    ((7, 1, 0), None),  # past the end before scaling
    ((2, 3, 2), None),  # past the end only after scaling: 8
    ((0xFFFFFFFF, 1, 0), None),  # the largest index
    ((2, 0x80000001, 0), None),  # a naive 32-bit index * scale wraps back to 2
    ((1, 1, 0xFFFFFFFF), None),  # a naive 32-bit index + offset wraps back to 0
    ((0, 0xFFFFFFFF, 1), 1),  # a valid index under a scale that would overflow a naive bound
]


@pytest.mark.parametrize("dtype, layout, entry_shape, nd_shard", FORMS)
def test_index_selects_the_entry(device, dtype, layout, entry_shape, nd_shard):
    """Each INDEXING row loads the selected entry and stores into it, or leaves both tensors untouched."""
    table_host = _random((_TABLE_ENTRIES, *entry_shape), dtype, 1)
    table = _to_device(table_host, device, dtype, layout, _table_memory(device, nd_shard))
    single_host = _random((1, *entry_shape), dtype, 2)
    for n, ((value, scale, offset), entry) in enumerate(INDEXING):
        row = f"index {value:#x} scale {scale:#x} offset {offset:#x}"
        index = _index(device, value)
        single = _to_device(single_host, device, dtype, layout)  # a fresh destination for each cached program

        ttnn.experimental.kda.indexed_entry_copy(table, single, index, index_scale=scale, index_offset=offset)
        if entry is not None:
            single_host = table_host[entry : entry + 1].clone()
        assert torch.equal(ttnn.to_torch(single), single_host), f"load, {row}"

        update = _random((1, *entry_shape), dtype, 100 + n)
        source = _to_device(update, device, dtype, layout)
        ttnn.experimental.kda.indexed_entry_copy(source, table, index, index_scale=scale, index_offset=offset)
        if entry is not None:
            table_host[entry : entry + 1] = update
        assert torch.equal(ttnn.to_torch(table), table_host), f"store, {row}"
        ttnn.deallocate(source)
        ttnn.deallocate(single)
        ttnn.deallocate(index)


# One entry on both sides needs no ND form: the consumer's single-entry tensors are interleaved.
@pytest.mark.parametrize("dtype, layout, entry_shape, nd_shard", [FORMS[0], FORMS[4]])
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
            _overwrite(index, torch.tensor([value], dtype=torch.int64))
            _overwrite(update, new)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            table_host[entry : entry + 1] = new
            assert torch.equal(ttnn.to_torch(working), expected_load), f"replay load index {value}"
            assert torch.equal(ttnn.to_torch(table), table_host), f"replay store index {value}"
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize(
    "dtype, layout, source_shape, destination_shape, source_nd_shard, message",
    [
        (ttnn.float32, ttnn.TILE_LAYOUT, (2, 32, 32), (2, 32, 32), None, "must hold a single entry"),
        (ttnn.float32, ttnn.TILE_LAYOUT, (4, 32, 64), (1, 32, 32), None, "differ past dim 0"),
        # same padded shape, different logical shape
        (ttnn.float32, ttnn.TILE_LAYOUT, (4, 30, 32), (1, 32, 32), None, "differ past dim 0"),
        (ttnn.float32, ttnn.TILE_LAYOUT, (64, 32), (32, 32), None, "at least 3 for TILE"),
        # a row-major ND page is one shard row, which the entry page count does not model
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (4, 3, 192), (1, 3, 192), [1, 3, 64], "ND-sharded DRAM for TILE"),
    ],
    ids=["two-multi-entry", "entry-shape", "logical-shape", "rank-2-tile", "row-major-nd"],
)
def test_rejects(device, expect_error, dtype, layout, source_shape, destination_shape, source_nd_shard, message):
    source = _to_device(torch.zeros(source_shape), device, dtype, layout, _table_memory(device, source_nd_shard))
    destination = _to_device(torch.zeros(destination_shape), device, dtype, layout)
    with expect_error(RuntimeError, message):
        ttnn.experimental.kda.indexed_entry_copy(source, destination, _index(device, 0), index_scale=1, index_offset=0)


# Entries are strided by the single-entry side's padded shape, so a destination whose logical shape matches the
# 7-entry source but whose padding does not would read or write past an entry.
@pytest.mark.parametrize(
    "destination_padded, message",
    [((1, 64, 32), "and destination padded"), ((2, 32, 32), "must not pad dim 0")],
    ids=["padded-entry", "padded-dim-0"],
)
def test_rejects_padding(device, expect_error, destination_padded, message):
    source = _to_device(torch.zeros(7, 32, 32), device, ttnn.float32, ttnn.TILE_LAYOUT)
    destination = ttnn.reshape(
        _to_device(torch.zeros(destination_padded), device, ttnn.float32, ttnn.TILE_LAYOUT),
        (1, 32, 32),
        padded_shape=destination_padded,
    )
    with expect_error(RuntimeError, message):
        ttnn.experimental.kda.indexed_entry_copy(source, destination, _index(device, 6), index_scale=1, index_offset=0)


# Kimi-K3's per-chip carry staging at TP4 (24 local heads of 128, three history rows), with two user slots of the
# rank's 18 KDA layers in the slab and the layer at position 5: every KDA layer does each of these once per chunk.
_K3_LAYERS, _K3_SLOTS, _K3_POSITION = 18, 2, 5
_PRODUCTION_PERF_MARGIN = 0.05
# The 1.5 MiB copies are DRAM-bound with reads and writes sharing the channels, and vary about +/-5% run to run on a
# Blackhole P150 (store 9.07-9.46 us, load 9.21-9.53 us), so their band is wider than the latency-bound tail's.
_RECURRENT_PERF_MARGIN = 0.07


@dataclass(frozen=True)
class _ProductionCase:
    case_id: str
    dtype: ttnn.DataType
    layout: ttnn.Layout
    entry_shape: tuple[int, ...]
    nd_shard: list[int] | None
    load: bool
    expected_duration_ns: int
    margin: float = _PRODUCTION_PERF_MARGIN


_PRODUCTION_CASES = (
    _ProductionCase(
        "recurrent-store",
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        (24, 128, 128),
        [1, 1, 128, 32],
        False,
        8_900,
        _RECURRENT_PERF_MARGIN,
    ),
    _ProductionCase(
        "recurrent-load",
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        (24, 128, 128),
        [1, 1, 128, 32],
        True,
        8_700,
        _RECURRENT_PERF_MARGIN,
    ),
    _ProductionCase("convolution-store", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (144, 192), None, False, 1_860),
    _ProductionCase("convolution-load", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (144, 192), None, True, 1_840),
)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
def test_kimi_k3_production_performance(device, case: _ProductionCase) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for indexed_entry_copy performance checks")
    slab_host = _random((_K3_SLOTS * _K3_LAYERS, *case.entry_shape), case.dtype, 11)
    slab = _to_device(slab_host, device, case.dtype, case.layout, _table_memory(device, case.nd_shard))
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
        if any("indexed_entry_copy_reader.cpp" in source for source in program["kernel_sources"])
    ]
    duration_ns = perf_record["duration_ns"]
    entry = _K3_LAYERS + _K3_POSITION
    if case.load:
        assert torch.equal(ttnn.to_torch(carry), slab_host[entry : entry + 1]), "loaded entry"
    else:
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
    lower = case.expected_duration_ns * (1 - case.margin)
    upper = case.expected_duration_ns * (1 + case.margin)
    assert lower <= duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {case.expected_duration_ns} ns, margin +/- {case.margin * 100:.0f}%)"
    )

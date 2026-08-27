# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.utils_for_testing import assert_equal

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="topk_large_indices is Blackhole-only"),
]


def _make_bf16_exact_input(num_rows: int, n: int) -> torch.Tensor:
    rows = []
    for row in range(num_rows):
        hi16 = (0x3F80 + row * (n + 1) + np.arange(n, dtype=np.uint32)).astype(np.uint32)
        values = torch.from_numpy((hi16 << 16).view(np.float32).copy())
        rows.append(values.to(torch.bfloat16))
    return torch.stack(rows)


def _make_large_index_input(num_rows: int, n: int, k: int) -> torch.Tensor:
    values = torch.zeros((num_rows, n), dtype=torch.bfloat16)
    hi16 = (0x3F80 + np.arange(k, dtype=np.uint32)).astype(np.uint32)
    high_values = torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)
    values[:, -k:] = high_values
    return values


def _to_device(torch_input: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _assert_index_metadata(tt_indices: ttnn.Tensor, expected_shape: list[int]) -> None:
    assert list(tt_indices.shape) == expected_shape
    assert tt_indices.dtype == ttnn.uint32
    assert tt_indices.layout == ttnn.ROW_MAJOR_LAYOUT


def _assert_indices(tt_indices: ttnn.Tensor, expected: torch.Tensor, expected_shape: list[int]) -> None:
    _assert_index_metadata(tt_indices, expected_shape)

    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32)
    assert_equal(indices.to(torch.int64), expected.to(torch.int64))


def _assert_topk_matches_torch(torch_input: torch.Tensor, tt_indices: ttnn.Tensor, k: int) -> None:
    _, ref_indices = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)
    expected_shape = list(torch_input.shape)
    expected_shape[-1] = k
    _assert_indices(tt_indices, ref_indices, expected_shape)


@pytest.mark.parametrize(
    "k,num_rows,n",
    [
        (512, 1, 512),
        (512, 2, 512),
        (512, 1, 1024),
        (512, 2, 1024),
        (16, 1, 16),
        (256, 2, 513),
        (768, 2, 1537),
        (1024, 1, 1024),
        (1024, 2, 1024),
        (1024, 1, 2048),
        (1024, 2, 2048),
        (1536, 2, 3000),
        (2048, 1, 2048),
        (2048, 2, 2048),
        (2032, 2, 4095),
        (2048, 1, 4096),
        (2048, 2, 4096),
    ],
)
def test_topk_large_indices_row_major_bfloat16_uint32_indices(device, k, num_rows, n):
    torch_input = _make_bf16_exact_input(num_rows, n)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


def test_topk_large_indices_random_k2048_multichunk_multirow(device):
    # Minimal repro shape for a class of silicon failures seen during bring-up
    # (then the row-parallel factory; since the auto-rects flip this shape
    # auto-selects the multi-row rectangle engine — llk_k=2048, num_chunks=4,
    # P=4, one rect per row). If this passes while the routed
    # composite fails, the composite's post-op chain (gather at index width
    # 2048 -> gather's untested RM multi-core variant) is the culprit, not
    # this op. Tie-safe value-multiset assertions (random bf16 has duplicates).
    torch.manual_seed(0)
    rows, n, k = 2, 8192, 2048
    torch_input = torch.randn(rows, n, dtype=torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)

    assert indices.min() >= 0
    assert indices.max() < n
    for row_indices in indices:
        assert row_indices.unique().numel() == k

    actual_values = torch.gather(torch_input.float(), dim=-1, index=indices)
    ref_values, _ = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)
    assert_equal(actual_values.sort(dim=-1).values, ref_values.sort(dim=-1).values)


def test_topk_large_indices_random_bfloat16_ties_return_distinct_indices(device):
    torch.manual_seed(0)
    rows = 8
    n = 4096
    k = 2048
    torch_input = torch.randn(1, 1, rows, n, dtype=torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)[0, 0]

    assert indices.min() >= 0
    assert indices.max() < n
    for row_indices in indices:
        assert row_indices.unique().numel() == k

    actual_values = torch.gather(torch_input.float()[0, 0], dim=-1, index=indices)
    ref_values, _ = torch.topk(torch_input.float()[0, 0], k, dim=-1, largest=True, sorted=True)
    assert_equal(actual_values.sort(dim=-1).values, ref_values.sort(dim=-1).values)


def test_topk_large_indices_stable_classic_chunk_skip_keeps_late_winners(device):
    # k=64 snaps to a 512-wide LLK window; 32768 elements force the classic
    # body and exercise chunk-skip after stable rank stamps have reached the
    # resident threshold word. Late winners also require the custom SFPU max
    # fold to observe ordinary values after the preceding top-k phase.
    n, k = 32768, 64
    torch_input = torch.zeros((1, n), dtype=torch.bfloat16)
    first_winner = n - 96
    torch_input[0, first_winner:] = 2.0

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, stable=True)

    expected = torch.arange(first_winner, first_winner + k, dtype=torch.int64).unsqueeze(0)
    _assert_indices(tt_indices, expected, [1, k])


@pytest.mark.parametrize(
    "shape,k",
    [
        ((512,), 512),
        ((2, 3, 512), 512),
        ((2, 2, 513), 256),
    ],
)
def test_topk_large_indices_supported_ranks(device, shape, k):
    n = shape[-1]
    num_rows = int(np.prod(shape[:-1], dtype=np.uint64)) if len(shape) > 1 else 1
    torch_input = _make_bf16_exact_input(num_rows, n).reshape(shape)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


def test_topk_large_indices_requires_explicit_k(device):
    torch_input = _make_bf16_exact_input(num_rows=1, n=512)
    with pytest.raises(TypeError):  # allow-pytest.raises: host binding TypeError (missing kwarg), not a device fault
        ttnn.experimental.topk_large_indices(_to_device(torch_input, device))


@pytest.mark.parametrize(
    "k,n",
    [
        (512, 513),
        (512, 1023),
        (16, 17),
        (256, 511),
        (768, 1025),
        (1024, 1025),
        (1024, 2047),
        (1536, 2049),
        (2048, 2049),
        (2032, 4095),
        (2048, 4095),
    ],
)
def test_topk_large_indices_row_major_non_multiple_n(device, k, n):
    num_rows = 2
    torch_input = _make_bf16_exact_input(num_rows, n)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


@pytest.mark.parametrize("k", [16, 512, 768, 1024, 1536, 2032, 2048])
def test_topk_large_indices_row_major_parallelizes_640_rows(device, k):
    num_rows = 640
    n = k
    torch_input = _make_large_index_input(num_rows=num_rows, n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


def test_topk_large_indices_row_major_640_rows_51200_k1536(device):
    num_rows = 640
    n = 51200
    k = 1536
    torch_input = _make_large_index_input(num_rows=num_rows, n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


TOPK_LARGE_INDICES_PERF_MARGIN = 0.01

# Real-time-profiler baselines measured on a Blackhole dev board. The symmetric band catches both
# regressions and unexpected speedups that should trigger baseline review.
TOPK_LARGE_INDICES_PRODUCTION_PERF_CONFIGS = [
    # (case_id, num_rows, allocated_length, valid_length, k, expected_duration_ns)
    ("prefill", 640, 51200, None, 1536, 1_683_850),
    ("bounded_cache", 2, 102400, 56320, 1536, 316_890),
]


@pytest.mark.skip(
    reason="expected_duration_ns pins predate the multi-core landings and must be re-baselined "
    "on the IOMMU perf-runner class: https://github.com/tenstorrent/tt-metal/issues/53459"
)
@pytest.mark.parametrize(
    "case_id,num_rows,n,valid_length,k,expected_duration_ns",
    TOPK_LARGE_INDICES_PRODUCTION_PERF_CONFIGS,
    ids=[config[0] for config in TOPK_LARGE_INDICES_PRODUCTION_PERF_CONFIGS],
)
@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_topk_large_indices_production_perf_check(
    device,
    case_id,
    num_rows,
    n,
    valid_length,
    k,
    expected_duration_ns,
):
    """Check production-shape duration on the marker-selected host-IOMMU perf runner."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for topk_large_indices perf checks (needs IOMMU)")

    torch_input = _make_large_index_input(num_rows=num_rows, n=n, k=k)
    tt_input = _to_device(torch_input, device)

    def run_topk_large_indices():
        return ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)

    measured_out, perf_record = profile_realtime_program(device, run_topk_large_indices)
    duration_ns = perf_record["duration_ns"]
    lower = expected_duration_ns * (1 - TOPK_LARGE_INDICES_PERF_MARGIN)
    upper = expected_duration_ns * (1 + TOPK_LARGE_INDICES_PERF_MARGIN)

    logger.info(
        f"topk_large_indices perf check {case_id}: duration={duration_ns / 1e6:.3f} ms "
        f"(expected {expected_duration_ns / 1e6:.3f} ms, band [{lower / 1e6:.3f}, {upper / 1e6:.3f}]), "
        f"shape=({num_rows}, {n}), valid_length={valid_length}, k={k}, "
        f"profiler_runtime_id={perf_record['runtime_id']}"
    )

    del measured_out

    assert lower <= duration_ns <= upper, (
        f"{case_id} duration {duration_ns / 1e6:.3f} ms outside band "
        f"[{lower / 1e6:.3f}, {upper / 1e6:.3f}] ms "
        f"(expected {expected_duration_ns / 1e6:.3f} ms, "
        f"margin +/- {TOPK_LARGE_INDICES_PERF_MARGIN * 100:.1f}%)"
    )


def test_topk_large_indices_program_cache_per_engine_regime(device):
    # Engine selection is automatic: the cost model picks the multi-row
    # rectangle engine when 2*ceil(chunks/P) + ceil(log2 P) beats the
    # row-parallel 2*chunks by the multi-row margin, and the program hash
    # carries the derived split-config fields. The caching invariant is
    # therefore PER ENGINE REGIME (P150 13x10 worker grid assumed, as in the
    # merge-tree tests below):
    #   - shapes that stay row-parallel share ONE shape-free cached program
    #     across any row count and width: (2, 3000) declines rects on the
    #     margin (best rect cost 3 vs row cost 4, short of the required
    #     margin), while (128, 51200) and (640, 51200) exceed every
    #     rectangle's row capacity (25 chunks, max capacity 65 rows at P=2).
    #     At k >= 1024 the compute-body mode is width-independent, so no
    #     width term enters the hash within the regime;
    #   - valid_length is runtime-only and the split is modeled on the
    #     PHYSICAL width, so growing prefixes on one shape NEVER recompile;
    #   - an engine-crossing shape compiles exactly ONE extra program:
    #     (2, 131072) is 64 chunks, so the model auto-selects rects
    #     (P=32: 2*2 + 5 = 9, far under the row-parallel 2*64 = 128) and the
    #     changed split-config fields hash to a new entry. Repeats of it hit
    #     the cache; regime growth is bounded because the model quantizes P
    #     to a handful of choices and fixed-shape callers compile once.
    k = 1536
    row_parallel_cases = [(2, 3000), (128, 51200), (640, 51200)]
    tt_inputs = [
        _to_device(_make_large_index_input(num_rows=num_rows, n=n, k=k), device) for num_rows, n in row_parallel_cases
    ]
    tt_crossing = _to_device(_make_large_index_input(num_rows=2, n=131072, k=k), device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        cache_entries = []
        for tt_input, (num_rows, _) in zip(tt_inputs, row_parallel_cases):
            tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k)
            cache_entries.append(device.num_program_cache_entries())
            _assert_index_metadata(tt_indices, [num_rows, k])
        for valid_length in (4097, 26000):  # runtime-only: same (128, 51200) program
            tt_indices = ttnn.experimental.topk_large_indices(tt_inputs[1], k=k, valid_length=valid_length)
            cache_entries.append(device.num_program_cache_entries())
            _assert_index_metadata(tt_indices, [128, k])

        assert cache_entries[0] > 0
        assert max(cache_entries) == min(cache_entries)  # one program serves the whole regime
        row_parallel_entries = cache_entries[-1]

        tt_indices = ttnn.experimental.topk_large_indices(tt_crossing, k=k)
        _assert_index_metadata(tt_indices, [2, k])
        # Crossing into the rect engine compiles exactly one new program ...
        assert device.num_program_cache_entries() == row_parallel_entries + 1
        tt_indices = ttnn.experimental.topk_large_indices(tt_crossing, k=k)
        _assert_index_metadata(tt_indices, [2, k])
        # ... and repeat calls of the same shape stay cached.
        assert device.num_program_cache_entries() == row_parallel_entries + 1
    finally:
        device.disable_and_clear_program_cache()


def test_topk_large_indices_hybrid_uses_single_rectangle_for_one_row_remainder(device):
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    rows, n, k = num_cores + 1, 32768, 512
    torch_input = _make_large_index_input(num_rows=rows, n=n, k=k)
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        entries_before = device.num_program_cache_entries()
        tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k)
        ttnn.synchronize_device(device)

        # The hybrid composite compiles a row-parallel program for the full
        # wave and a one-rectangle column-parallel program for the last row.
        # The pre-fix single-launch fallback adds only one program.
        assert device.num_program_cache_entries() >= entries_before + 2
        _assert_topk_matches_torch(torch_input, tt_indices, k)
    finally:
        device.disable_and_clear_program_cache()


@pytest.mark.parametrize(
    "k,num_chunks",
    [
        (512, 129),
        (512, 256),
        (512, 257),
        (1024, 65),
        (1024, 128),
        (1024, 129),
        (2048, 33),
        (2048, 64),
        (2048, 129),
    ],
)
def test_topk_large_indices_row_major_uint32_indices_above_uint16(device, k, num_chunks):
    n = num_chunks * k
    torch_input = _make_large_index_input(num_rows=1, n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _, ref_indices = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)

    assert int(ref_indices.min()) >= 65536
    _assert_indices(tt_indices, ref_indices, [1, k])


@pytest.mark.parametrize("k", [16, 512, 768, 1024, 1536, 2032, 2048])
def test_topk_large_indices_row_major_non_multiple_n_uint32_indices_above_uint16(device, k):
    n = 65536 + k + 1
    torch_input = _make_large_index_input(num_rows=1, n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _, ref_indices = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)

    assert int(ref_indices.min()) >= 65536
    _assert_indices(tt_indices, ref_indices, [1, k])


@pytest.mark.parametrize("k", [16, 512, 768, 1024, 1536, 2048])
def test_topk_large_indices_row_major_negative_infinity_indices_are_sentinel(device, k):
    sentinel = 0xFFFFFFFF
    n = k
    torch_input = torch.full((2, n), -float("inf"), dtype=torch.bfloat16)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_indices(tt_indices, torch.full((2, k), sentinel, dtype=torch.int64), [2, k])


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_topk_large_indices_row_major_non_multiple_n_negative_infinity_indices_are_sentinel(device, k):
    sentinel = 0xFFFFFFFF
    n = k + 1
    torch_input = torch.full((2, n), -float("inf"), dtype=torch.bfloat16)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_indices(tt_indices, torch.full((2, k), sentinel, dtype=torch.int64), [2, k])


@pytest.mark.parametrize("k", [16, 512, 768, 1024, 1536, 2048])
def test_topk_large_indices_row_major_mixed_negative_infinity_indices_are_sentinel(device, k):
    sentinel = 0xFFFFFFFF
    n = k + 17
    finite_count = 16
    torch_input = torch.full((2, n), -float("inf"), dtype=torch.bfloat16)
    finite_values = torch.arange(finite_count, dtype=torch.float32).to(torch.bfloat16)
    torch_input[:, :finite_count] = finite_values
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    expected_prefix = torch.arange(finite_count - 1, -1, -1, dtype=torch.int64)
    expected_suffix = torch.full((k - finite_count,), sentinel, dtype=torch.int64)
    expected_row = torch.cat([expected_prefix, expected_suffix])
    expected = expected_row.unsqueeze(0).repeat(2, 1)

    _assert_indices(tt_indices, expected, [2, k])


# ---------------------------------------------------------------------------
# Segmented fusion: rows wider than the 32-chunk fused ceiling run <=32-chunk
# fused segments (segment-local chunk-id stamp, one split per segment) folded
# by unfused cross-segment merges. Cells pin the risky boundaries: first
# unfused-era width (33 chunks), a single-chunk tail segment, deep segment
# counts, and valid_length collapsing a wide program back to one segment.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k,num_rows,n",
    [
        (2048, 2, 67584),  # 33 chunks: segment 1 holds exactly ONE chunk (rebuild-asc after bare sort)
        (2048, 2, 131072),  # 64 chunks: 2 full segments
        (1536, 2, 131072),  # k=1536 runs the 2048 window; USER_K slice of a segmented row
        (2048, 1, 524288),  # 256 chunks: 8 segments (Pavle's scaled shape class)
        (1536, 2, 524288),
        (2048, 1, 1048576),  # 512 chunks: 16 segments, index magnitudes to 2^20
        (1024, 2, 66560),  # K=1024 window: 65 chunks, seg1 = 33... spans capacity at a different K
        (2048, 2, 100000),  # non-chunk-multiple width: -inf padded tail inside the last segment
        (2048, 66, 131072),  # rows above rect capacity: the ROW-PARALLEL segmented body (>32 chunks)
        #                      stays covered by a bare call now that 2-row shapes auto-select rects
    ],
)
def test_topk_large_indices_segmented_widths(device, k, num_rows, n):
    torch_input = _make_large_index_input(num_rows=num_rows, n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    _assert_topk_matches_torch(torch_input, tt_indices, k)


def test_topk_large_indices_segmented_valid_length_collapses_to_one_segment(device):
    # A wide program whose runtime valid_length shrinks the row to <= 32
    # chunks must keep running inside the SAME cached binary -- and again at
    # a segment-boundary-crossing length. Since the auto-rects flip this
    # (2, 524288) shape (256 chunks) runs the multi-row rectangle engine, so
    # the collapse is exercised through the rect slices' runtime prefix
    # rebalancing rather than the row-parallel segmented body; either way
    # valid_length stays runtime-only and no recompile is allowed.
    k, n = 2048, 524288
    # Distinct top-k block INSIDE the smallest tested valid_length so every
    # tested prefix has a tie-free torch reference (zeros elsewhere).
    torch_input = torch.zeros((2, n), dtype=torch.bfloat16)
    hi16 = (0x3F80 + np.arange(k, dtype=np.uint32)).astype(np.uint32)
    torch_input[:, 40960 - k : 40960] = torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)
    tt_input = _to_device(torch_input, device)
    device.enable_program_cache()
    device.clear_program_cache()
    for valid_length in (n, 40960, 98304):  # 256 chunks, 20 chunks (1 seg), 48 chunks (2 segs)
        tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)
        _assert_topk_matches_torch(torch_input[:, :valid_length], tt_indices, k)
    assert device.num_program_cache_entries() == 1  # one program serves every runtime length
    device.disable_and_clear_program_cache()


# ---------------------------------------------------------------------------
# valid_length: bound the search to the first N columns of each (wider) row.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k,n,valid_length",
    [
        (512, 1024, 512),
        (256, 2048, 512),
        (16, 4096, 64),
        (512, 4096, 1024),
        (1024, 4096, 2048),
        (2048, 4096, 2048),
        (256, 2048, 600),  # valid_length not a multiple of the LLK window
        (512, 4096, 513),  # odd valid_length (partial tail chunk)
    ],
)
def test_topk_large_indices_valid_length_ignores_stale_tail(device, k, n, valid_length):
    # _make_bf16_exact_input rows are strictly increasing, so the LARGEST values of the full row live in the
    # tail [valid_length:n]. A correct valid_length must return only indices < valid_length (matching a top-k
    # over the prefix); if the tail leaked, indices >= valid_length would appear. This is the core guarantee.
    num_rows = 2
    torch_input = _make_bf16_exact_input(num_rows, n)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)

    _, ref_indices = torch.topk(torch_input[:, :valid_length].float(), k, dim=-1, largest=True, sorted=True)
    assert int(ref_indices.max()) < valid_length
    _assert_indices(tt_indices, ref_indices, [num_rows, k])


@pytest.mark.parametrize(
    "k,num_rows,n,valid_length",
    [
        (512, 2, 1024, 512),
        (1536, 2, 3000, 2000),
        (2048, 2, 4096, 2049),
        (16, 1, 8192, 100),
        (1536, 2, 102400, 56320),  # the production case: 100K allocated, 50K+5K written
    ],
)
def test_topk_large_indices_valid_length_matches_sliced_input(device, k, num_rows, n, valid_length):
    # valid_length=L must select the exact same top-k VALUE multiset as physically slicing the row to
    # [0, L) and running the full-width op. Indices may legitimately differ ONLY on exact-bf16 ties: the
    # two calls have different physical widths, and the engine (and with it the unspecified non-stable
    # tie order) is chosen per physical width — e.g. the FUSED_E2E gate flips at 32 chunks, and fused
    # cross-chunk compares break ties by stamped chunk id while unfused merges break them by network
    # position. Every index mismatch is therefore asserted to be a bitwise value tie, and the sorted
    # value sequences must match bit-for-bit.
    torch.manual_seed(0)
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)

    bounded = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)
    sliced = ttnn.experimental.topk_large_indices(_to_device(torch_input[:, :valid_length].contiguous(), device), k=k)

    bounded_t = ttnn.to_torch(bounded, dtype=torch.uint32).to(torch.int64)
    sliced_t = ttnn.to_torch(sliced, dtype=torch.uint32).to(torch.int64)
    _assert_index_metadata(bounded, [num_rows, k])
    source = torch_input.to(torch.float32)
    for r in range(num_rows):
        gathered_bounded = source[r, bounded_t[r]]
        gathered_sliced = source[r, sliced_t[r]]
        diff = (bounded_t[r] != sliced_t[r]).nonzero().flatten()
        assert torch.equal(
            gathered_bounded[diff], gathered_sliced[diff]
        ), f"row {r}: {len(diff)} index diffs include a non-tie (values differ)"
        assert torch.equal(
            torch.sort(gathered_bounded, descending=True).values,
            torch.sort(gathered_sliced, descending=True).values,
        ), f"row {r}: top-k value multisets differ"


def test_topk_large_indices_valid_length_full_width_is_noop(device):
    # valid_length == n must behave exactly like omitting valid_length.
    num_rows, n, k = 2, 4096, 1024
    torch_input = _make_bf16_exact_input(num_rows, n)
    with_arg = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=n)
    without = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    assert_equal(
        ttnn.to_torch(with_arg, dtype=torch.uint32).to(torch.int64),
        ttnn.to_torch(without, dtype=torch.uint32).to(torch.int64),
    )


@pytest.mark.parametrize("k", [16, 512, 1024])
def test_topk_large_indices_valid_length_all_neginf_prefix_is_sentinel(device, k):
    # Prefix [0, k) is all -inf while the ignored tail holds large FINITE values. A correct bound returns
    # k sentinels (the tail must never be selected even though it is finite and larger).
    sentinel = 0xFFFFFFFF
    n = k + 64
    torch_input = torch.full((2, n), -float("inf"), dtype=torch.bfloat16)
    torch_input[:, k:] = 5.0  # large finite values living in the ignored tail
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=k)
    _assert_indices(tt_indices, torch.full((2, k), sentinel, dtype=torch.int64), [2, k])


@pytest.mark.parametrize("k", [16, 512, 1024])
def test_topk_large_indices_valid_length_mixed_prefix_ignores_finite_tail(device, k):
    # 16 finite values at the start of the valid prefix, the rest of the prefix -inf, and a large finite tail.
    # Expect the 16 real indices then sentinels -- never the tail.
    sentinel = 0xFFFFFFFF
    n = k + 64
    valid_length = k
    finite_count = 16
    torch_input = torch.full((2, n), -float("inf"), dtype=torch.bfloat16)
    torch_input[:, :finite_count] = torch.arange(finite_count, dtype=torch.float32).to(torch.bfloat16)
    torch_input[:, valid_length:] = 100.0  # large finite tail that must be ignored
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)

    expected_prefix = torch.arange(finite_count - 1, -1, -1, dtype=torch.int64)
    expected_suffix = torch.full((k - finite_count,), sentinel, dtype=torch.int64)
    expected = torch.cat([expected_prefix, expected_suffix]).unsqueeze(0).repeat(2, 1)
    _assert_indices(tt_indices, expected, [2, k])


def test_topk_large_indices_valid_length_program_cache_reuse_while_growing(device):
    # A serving loop grows valid_length each step; because valid_length is a runtime arg (hash-excluded),
    # every step must reuse a single cached program.
    k = 1536
    n = 102400
    torch_input = _make_large_index_input(num_rows=2, n=n, k=k)
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        entries = []
        for valid_length in (2048, 20480, 56320, n):
            ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)
            entries.append(device.num_program_cache_entries())
        assert entries[0] > 0
        assert max(entries) == min(entries)  # no recompile as valid_length grew
    finally:
        device.disable_and_clear_program_cache()


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_topk_large_indices_short_valid_length_emits_sentinels(device, k):
    # Sparse MLA keeps a fixed maximum top-k shape while its cache is shorter than k.  The valid
    # prefix supplies the real indices and the remaining output slots must be the sparse-SDPA sentinel.
    valid_length = 496
    # Keep one full output-width of finite stale data beyond the requested top-k, so every k exercises
    # that the runtime prefix bound—not just the sentinel padding—excludes the physical tail.
    n = 2 * k
    torch_input = _make_bf16_exact_input(num_rows=1, n=n)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)
    _, valid_indices = torch.topk(
        torch_input[:, :valid_length].float(), valid_length, dim=-1, largest=True, sorted=True
    )
    expected = torch.cat(
        [
            valid_indices,
            torch.full((1, k - valid_length), 0xFFFFFFFF, dtype=torch.int64),
        ],
        dim=-1,
    )
    _assert_indices(tt_indices, expected, [1, k])


@pytest.mark.parametrize(
    "k,n,valid_length",
    [
        (512, 1024, 0),  # valid_length must be positive
        (512, 1024, 2048),  # valid_length > n
    ],
)
def test_topk_large_indices_valid_length_out_of_range_raises(device, expect_error, k, n, valid_length):
    torch_input = _make_bf16_exact_input(num_rows=1, n=n)
    with expect_error(RuntimeError, "valid_length"):
        ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)


# ---------------------------------------------------------------------------
# Column-parallel (intra-row multi-core) path: single row, many chunks per row.
# The op splits the row's chunks over a rectangle of local cores and merges the
# per-slice survivors on one final core. Selected automatically when num_rows
# is 1 and the row is wide enough that the split beats a single core.
# ---------------------------------------------------------------------------


def _make_distinct_block_input(n: int, k: int, block_start: int, background: float = 0.0) -> torch.Tensor:
    """One row of `background` with k strictly-increasing bf16-exact values at [block_start, block_start+k)."""
    values = torch.full((1, n), background, dtype=torch.bfloat16)
    hi16 = (0x3F80 + np.arange(k, dtype=np.uint32)).astype(np.uint32)
    block = torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)
    values[:, block_start : block_start + k] = block
    return values


@pytest.mark.parametrize(
    "k,n",
    [
        (512, 32768),  # 64 chunks
        (512, 65536),  # 128 chunks
        (1024, 32768),  # 32 chunks
        (1024, 65536),  # 64 chunks
        (2048, 65536),  # 32 chunks over ~8 slices -> >=4 chained chunk merges per local core
        (2048, 102400),  # 50 chunks: uneven chunk split across slices
        (1536, 65536),  # k below the LLK window (2048): output narrower than the merge sequence
        (2048, 65537),  # 33 chunks with a 1-element tail chunk on the last slice
    ],
)
def test_topk_large_indices_column_parallel_single_row(device, k, n):
    torch_input = _make_large_index_input(num_rows=1, n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


def test_topk_large_indices_column_parallel_top_values_straddle_slice_boundary(device):
    # W=65536 with k=2048 splits into 8 slices of 8192 columns. Center the
    # winning block on a slice boundary so both neighbors contribute half of
    # the final top-k and the cross-slice merge ordering is exercised.
    n, k = 65536, 2048
    boundary = 4 * 8192
    torch_input = _make_distinct_block_input(n, k, block_start=boundary - k // 2)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


def test_topk_large_indices_column_parallel_random_ties_return_distinct_indices(device):
    # Random bf16 over 64K columns is guaranteed to contain duplicate values,
    # including ties that straddle slice boundaries. Tie order between slices
    # is unspecified, so check the selected value multiset and index validity
    # instead of exact index equality.
    torch.manual_seed(0)
    n, k = 65536, 2048
    torch_input = torch.randn(1, n, dtype=torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)[0]

    assert indices.min() >= 0
    assert indices.max() < n
    assert indices.unique().numel() == k

    actual_values = torch_input.float()[0][indices]
    ref_values, _ = torch.topk(torch_input.float()[0], k, largest=True, sorted=True)
    assert_equal(actual_values.sort().values, ref_values.sort().values)


def test_topk_large_indices_column_parallel_all_equal_row(device):
    # Every element ties. Any k distinct indices are a correct answer; the
    # merge tree must not lose or duplicate lanes across slices.
    n, k = 65536, 2048
    torch_input = torch.ones(1, n, dtype=torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)[0]

    assert indices.min() >= 0
    assert indices.max() < n
    assert indices.unique().numel() == k


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_topk_large_indices_column_parallel_all_neginf_is_sentinel(device, k):
    # All lanes are -inf in every slice, so every gathered sequence is the
    # degenerate all--inf run and the final core must emit only sentinels.
    sentinel = 0xFFFFFFFF
    n = 32 * k
    torch_input = torch.full((1, n), -float("inf"), dtype=torch.bfloat16)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_indices(tt_indices, torch.full((1, k), sentinel, dtype=torch.int64), [1, k])


@pytest.mark.parametrize(
    "valid_length,block_start",
    [
        (2048, 0),  # only slice 0 active (single chunk); slices 1..P-1 empty -> writer -inf fill
        (8192, 8192 - 2048),  # slice 0 fully active, the rest empty
        (20000, 17952),  # partial tail chunk mid-slice; trailing slices empty
        (65536, 65536 - 2048),  # full width: no empty slices
    ],
)
def test_topk_large_indices_column_parallel_valid_length(device, valid_length, block_start):
    # The winning block sits inside the valid prefix; the stale tail holds
    # LARGER finite decoys that must never be selected. Empty slices (fully
    # beyond valid_length) contribute writer-fabricated -inf sequences.
    n, k = 65536, 2048
    torch_input = _make_distinct_block_input(n, k, block_start=block_start)
    if valid_length < n:
        torch_input[:, valid_length:] = 100.0

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)

    _, ref_indices = torch.topk(torch_input[:, :valid_length].float(), k, dim=-1, largest=True, sorted=True)
    assert int(ref_indices.max()) < valid_length
    _assert_indices(tt_indices, ref_indices, [1, k])


def test_topk_large_indices_column_parallel_short_valid_length_emits_sentinels(device):
    # valid_length < k: the real indices come from the short prefix and the
    # remaining output lanes must be sentinels, with every slice past the
    # prefix contributing an empty (writer-filled) sequence.
    n, k = 65536, 2048
    valid_length = 496
    torch_input = _make_bf16_exact_input(num_rows=1, n=2048)
    torch_input = torch.nn.functional.pad(torch_input.float(), (0, n - 2048), value=100.0).to(torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)

    _, valid_indices = torch.topk(
        torch_input[:, :valid_length].float(), valid_length, dim=-1, largest=True, sorted=True
    )
    expected = torch.cat(
        [valid_indices, torch.full((1, k - valid_length), 0xFFFFFFFF, dtype=torch.int64)],
        dim=-1,
    )
    _assert_indices(tt_indices, expected, [1, k])


def test_topk_large_indices_column_parallel_valid_length_cache_reuse(device):
    # The column split is derived from the PHYSICAL width only, so a serving
    # loop growing valid_length must reuse one cached column-parallel program
    # (empty slices shrink to active ones purely through runtime args).
    #
    # Every 8192-wide region carries its own strictly-increasing distinct
    # k-block at its start, with each block's values above the previous
    # block's. There are no ties anywhere (tie order vs torch is unspecified,
    # so an all-zeros prefix cannot be exact-asserted), and each growth step
    # has a DIFFERENT exact answer — the newest fully-visible block — so a
    # stale valid_length-derived runtime arg on a cache hit cannot pass.
    n, k = 65536, 2048
    num_blocks = 8
    region_width = n // num_blocks
    torch_input = torch.zeros(1, n, dtype=torch.bfloat16)
    for i in range(num_blocks):
        hi16 = (0x3F80 + i * k + np.arange(k, dtype=np.uint32)).astype(np.uint32)
        block = torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)
        torch_input[:, i * region_width : i * region_width + k] = block
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        entries = []
        # valid_length -> index of the largest fully-visible block:
        # 2048 sees only block 0; 20480 reaches block 2 ([16384, 18432));
        # 40960 reaches block 4 ([32768, 34816)); full width reaches block 7.
        for valid_length, top_block in ((2048, 0), (20480, 2), (40960, 4), (n, 7)):
            tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)
            entries.append(device.num_program_cache_entries())
            block_start = top_block * region_width
            expected = torch.arange(block_start + k - 1, block_start - 1, -1, dtype=torch.int64).unsqueeze(0)
            _assert_indices(tt_indices, expected, [1, k])
        assert entries[0] > 0
        assert max(entries) == min(entries)  # no recompile as valid_length grew
    finally:
        device.disable_and_clear_program_cache()


def test_topk_large_indices_column_and_row_parallel_programs_coexist_in_cache(device):
    # A single-row wide input takes the column-parallel factory while a
    # multi-row input of the same width keeps the row-parallel one; the two
    # must hash to distinct cache entries and both must stay correct. 128
    # rows exceed every rectangle's row capacity at 32 chunks (max 65 at
    # P=2 on the 13x10 grid), so the auto-rects cost model declines and the
    # multi-row call genuinely stays row-parallel; a low row count (e.g. 2)
    # would itself auto-select the rect engine since the flip.
    n, k = 65536, 2048
    single_row = _make_large_index_input(num_rows=1, n=n, k=k)
    multi_row = _make_large_index_input(num_rows=128, n=n, k=k)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        tt_single = ttnn.experimental.topk_large_indices(_to_device(single_row, device), k=k)
        entries_after_single = device.num_program_cache_entries()
        tt_multi = ttnn.experimental.topk_large_indices(_to_device(multi_row, device), k=k)
        entries_after_multi = device.num_program_cache_entries()

        assert entries_after_single > 0
        assert entries_after_multi == entries_after_single + 1

        _assert_topk_matches_torch(single_row, tt_single, k)
        _assert_topk_matches_torch(multi_row, tt_multi, k)
    finally:
        device.disable_and_clear_program_cache()


def test_topk_large_indices_returns_single_indices_tensor(device):
    # Contract: the result is a single indices tensor (not a tuple/list).
    torch_input = _make_bf16_exact_input(num_rows=2, n=1024)
    result = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=512)
    assert isinstance(result, ttnn.Tensor)
    _assert_topk_matches_torch(torch_input, result, 512)


# ---------------------------------------------------------------------------
# Column-parallel MERGE TREE (in-place log2(P) reduction on the slice
# rectangle; slice 0 is the root). Single-row shapes auto-select the tree via
# the cost model (cost(P) = 2*ceil(chunks/P) + ceil(log2 P)); the widths below
# are chosen so the model picks a KNOWN P, so each test still pins a specific
# tree depth without any public slice-count knob.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_chunks", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("k", [512, 2048])
def test_topk_large_indices_tree_random_ties(device, num_chunks, k):
    # Random bf16 has duplicate values, including ties that straddle slice and
    # tree-level boundaries; tie order is unspecified, so assert the selected
    # value multiset + index validity. n spans chunk counts from 4 to 64 for
    # both LLK windows; the model-picked P grows with the chunk count (P ==
    # chunks up to the grid's rectangle capacity), covering 2- to 6-level trees.
    torch.manual_seed(0)
    llk_k = 512 if k == 512 else 2048
    n = num_chunks * llk_k
    torch_input = torch.randn(1, n, dtype=torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)[0]

    assert indices.min() >= 0
    assert indices.max() < n
    assert indices.unique().numel() == k

    actual_values = torch_input.float()[0][indices]
    ref_values, _ = torch.topk(torch_input.float()[0], k, largest=True, sorted=True)
    assert_equal(actual_values.sort().values, ref_values.sort().values)


def test_topk_large_indices_tree_multilevel_adversarial_placement(device):
    # 3-level tree: 8 chunks of the 2048 window (n=16384) make the cost model
    # pick P=8 (cost 2*1+3=5, strictly under every smaller P), slice width
    # 2048. The winning block is split across slices that meet only AT THE
    # ROOT and traverse different tree depths:
    #   slice 5 (upper half of the top-k): 5 -> 4 (level 0), 4 -> 0 (level 2)
    #   slice 3 (lower half):              3 -> 2 (level 0), 2 -> 0 (level 1)
    # A mis-ordered intermediate survivor (the num_chunks=4 lesson, applied at
    # tree levels) only shows when its consumer merges it — the root sees both
    # halves through 2-deep chains. Values are globally distinct, so the final
    # rank order is asserted EXACTLY against torch.
    n, k = 16384, 2048
    region = 2048  # one chunk per slice
    half = k // 2
    torch_input = torch.zeros(1, n, dtype=torch.bfloat16)

    def block(count, base):
        hi16 = (0x3F80 + base + np.arange(count, dtype=np.uint32)).astype(np.uint32)
        return torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)

    # Lower half of the winners in slice 3, upper half in slice 5 (all values
    # distinct and above the zero background).
    torch_input[0, 3 * region : 3 * region + half] = block(half, 0)
    torch_input[0, 5 * region : 5 * region + half] = block(half, half)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _, ref_indices = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)
    _assert_indices(tt_indices, ref_indices, [1, k])


def test_topk_large_indices_tree_valid_length_empty_winners(device):
    # 16 chunks of the 2048 window (n=32768) make the cost model pick P=16
    # (4-level tree). valid_length=5000 is 3 valid chunks, rebalanced over the
    # slices: slices 0..1 full, slice 2 a 904-element tail, slices 3..15
    # EMPTY. Empty WINNERS (4, 6, 8, ...) must adopt their first incoming
    # survivor instead of merging into an empty DST; empty pure losers ship
    # the writer's -inf scratch. Winning block sits inside the prefix, decoys
    # beyond it.
    n, k = 32768, 2048
    valid_length = 5000
    torch_input = _make_distinct_block_input(n, k, block_start=valid_length - k)
    torch_input[:, valid_length:] = 100.0  # stale decoys, must never be selected

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)

    _, ref_indices = torch.topk(torch_input[:, :valid_length].float(), k, dim=-1, largest=True, sorted=True)
    assert int(ref_indices.max()) < valid_length
    _assert_indices(tt_indices, ref_indices, [1, k])


def _rect_core_grid(start_x, end_x, end_y):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(start_x, 0), ttnn.CoreCoord(end_x, end_y))])


def _run_topk_on_manager(device, tt_input, torch_input, core_grid, k=512):
    manager = device.create_sub_device_manager([ttnn.SubDevice([core_grid])], 0)
    device.load_sub_device_manager(manager)
    try:
        result = ttnn.experimental.topk_large_indices(
            tt_input,
            k=k,
            subdevice_id=ttnn.SubDeviceId(0),
            sub_core_grids=core_grid,
        )
        ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0)])
        _assert_topk_matches_torch(torch_input, result, k)
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


def test_topk_large_indices_explicit_full_grid_backward_compatibility(device):
    grid = device.compute_with_storage_grid_size()
    full_grid = _rect_core_grid(0, grid.x - 1, grid.y - 1)
    torch_input = _make_large_index_input(num_rows=127, n=1024, k=512)
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        implicit = ttnn.experimental.topk_large_indices(tt_input, k=512)
        ttnn.synchronize_device(device)
        entries_after_implicit = device.num_program_cache_entries()
        explicit = ttnn.experimental.topk_large_indices(
            tt_input,
            k=512,
            subdevice_id=ttnn.SubDeviceId(0),
            sub_core_grids=full_grid,
        )
        explicit_grid_only = ttnn.experimental.topk_large_indices(tt_input, k=512, sub_core_grids=full_grid)
        ttnn.synchronize_device(device)

        assert entries_after_implicit > 0
        assert device.num_program_cache_entries() == entries_after_implicit
        _assert_topk_matches_torch(torch_input, implicit, 512)
        _assert_topk_matches_torch(torch_input, explicit, 512)
        _assert_topk_matches_torch(torch_input, explicit_grid_only, 512)
    finally:
        device.disable_and_clear_program_cache()


def test_topk_large_indices_restricted_rectangular_and_discontiguous_grids(device):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 11 or grid.y < 10:
        pytest.skip(f"production top-k core profiles require at least an 11x10 worker grid, got {grid}")

    origin_80 = _rect_core_grid(0, 7, 9)
    non_origin_80 = _rect_core_grid(grid.x - 8, grid.x - 1, 9)
    discontiguous_80 = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(grid.x - 4, 0), ttnn.CoreCoord(grid.x - 1, 9)),
        ]
    )
    assert origin_80.num_cores() == non_origin_80.num_cores() == discontiguous_80.num_cores() == 80

    # 163 rows force unequal row groups and exercise start-row accumulation across both
    # ranges in row-wise corerange_to_cores traversal order.
    base_row = _make_large_index_input(num_rows=1, n=1024, k=512)[0]
    torch_input = torch.stack([torch.roll(base_row, shifts=row) for row in range(163)])
    tt_input = _to_device(torch_input, device)
    for core_grid in (origin_80, non_origin_80, discontiguous_80):
        _run_topk_on_manager(device, tt_input, torch_input, core_grid)


def test_topk_large_indices_rejects_invalid_subdevice_core_selection(device, expect_error):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 9:
        pytest.skip(f"subdevice split test requires at least nine worker columns, got {grid}")

    topk_grid = _rect_core_grid(0, 7, grid.y - 1)
    gather_grid = _rect_core_grid(8, grid.x - 1, grid.y - 1)
    manager = device.create_sub_device_manager(
        [ttnn.SubDevice([topk_grid]), ttnn.SubDevice([gather_grid])],
        0,
    )
    device.load_sub_device_manager(manager)
    torch_input = _make_large_index_input(num_rows=32, n=512, k=512)
    tt_input = _to_device(torch_input, device)
    try:
        escaped_grid = _rect_core_grid(7, 8, grid.y - 1)
        with expect_error(RuntimeError, "must be fully contained"):
            ttnn.experimental.topk_large_indices(
                tt_input,
                k=512,
                subdevice_id=ttnn.SubDeviceId(0),
                sub_core_grids=escaped_grid,
            )
        with expect_error(RuntimeError, "is not part of active subdevice manager"):
            ttnn.experimental.topk_large_indices(
                tt_input,
                k=512,
                subdevice_id=ttnn.SubDeviceId(2),
            )
        with expect_error(RuntimeError, "requires at least one TENSIX worker core"):
            ttnn.experimental.topk_large_indices(
                tt_input,
                k=512,
                subdevice_id=ttnn.SubDeviceId(0),
                sub_core_grids=ttnn.CoreRangeSet([]),
            )
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)

    outside_device_grid = _rect_core_grid(grid.x, grid.x, 0)
    with expect_error(RuntimeError, "must be fully contained"):
        ttnn.experimental.topk_large_indices(tt_input, k=512, sub_core_grids=outside_device_grid)


def test_topk_large_indices_program_cache_separates_resolved_core_grids(device):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 11 or grid.y < 10:
        pytest.skip(f"production cache profiles require at least an 11x10 worker grid, got {grid}")

    grid_80 = _rect_core_grid(0, 7, 9)
    grid_30 = _rect_core_grid(grid.x - 3, grid.x - 1, 9)
    manager_80 = device.create_sub_device_manager([ttnn.SubDevice([grid_80])], 0)
    manager_30 = device.create_sub_device_manager([ttnn.SubDevice([grid_30])], 0)
    torch_input = _make_large_index_input(num_rows=64, n=1024, k=512)
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        default_output = ttnn.experimental.topk_large_indices(tt_input, k=512)
        ttnn.synchronize_device(device)
        entries_after_default = device.num_program_cache_entries()

        device.load_sub_device_manager(manager_80)
        output_80 = ttnn.experimental.topk_large_indices(tt_input, k=512, subdevice_id=ttnn.SubDeviceId(0))
        ttnn.synchronize_device(device)
        entries_after_80 = device.num_program_cache_entries()
        assert entries_after_80 > entries_after_default

        device.load_sub_device_manager(manager_30)
        output_30 = ttnn.experimental.topk_large_indices(tt_input, k=512, subdevice_id=ttnn.SubDeviceId(0))
        ttnn.synchronize_device(device)
        assert device.num_program_cache_entries() > entries_after_80
        _assert_topk_matches_torch(torch_input, default_output, 512)
        _assert_topk_matches_torch(torch_input, output_80, 512)
        _assert_topk_matches_torch(torch_input, output_30, 512)
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager_80)
        device.remove_sub_device_manager(manager_30)
        device.disable_and_clear_program_cache()


def test_topk_large_indices_restricted_grid_cache_hit_rebinds_shape_and_valid_length(device):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 10:
        pytest.skip(f"80-core top-k profile requires at least an 8x10 worker grid, got {grid}")

    grid_80 = _rect_core_grid(0, 7, 9)
    manager = device.create_sub_device_manager([ttnn.SubDevice([grid_80])], 0)
    cases = []
    for num_rows, n, valid_length in ((2, 1024, 512), (163, 4096, 2048), (5, 8192, 4096)):
        torch_input = torch.zeros((num_rows, n), dtype=torch.bfloat16)
        torch_input[:, :valid_length] = _make_large_index_input(
            num_rows=num_rows,
            n=valid_length,
            k=512,
        )
        torch_input[:, valid_length:] = 100.0
        cases.append((torch_input, _to_device(torch_input, device), valid_length))

    device.enable_program_cache()
    device.clear_program_cache()
    device.load_sub_device_manager(manager)
    try:
        # Engine selection is shape-aware (the derived split config is part of
        # the program hash), so distinct (rows, n) cases may compile distinct
        # programs. The cache-hit contract under a restricted grid is
        # per-shape: repeating a case must rebind purely through runtime args.
        for torch_input, tt_input, valid_length in cases:
            entries_per_pass = []
            for _ in range(2):
                output = ttnn.experimental.topk_large_indices(
                    tt_input,
                    k=512,
                    valid_length=valid_length,
                    subdevice_id=ttnn.SubDeviceId(0),
                    sub_core_grids=grid_80,
                )
                ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0)])
                entries_per_pass.append(device.num_program_cache_entries())
                _, expected = torch.topk(
                    torch_input[:, :valid_length].float(),
                    512,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
                _assert_indices(output, expected, [torch_input.shape[0], 512])
            assert entries_per_pass[0] > 0
            assert entries_per_pass[1] == entries_per_pass[0]
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)
        device.disable_and_clear_program_cache()

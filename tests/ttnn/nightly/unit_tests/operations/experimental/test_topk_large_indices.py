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
    # Minimal repro shape for a class of silicon failures seen during bring-up:
    # row-parallel factory, llk_k=2048, num_chunks=4, multiple rows, RANDOM
    # input. If this passes while the routed
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


def test_topk_large_indices_program_cache_ignores_row_count_and_array_size(device):
    # (2, 131072) crosses the old 32-chunk fused boundary: at k >= 1024 the
    # compute-body mode is width-independent (one segmented codepath), so a
    # growing-prefill caller must NOT recompile crossing 65536 positions.
    k = 1536
    cases = [(2, 3000), (640, 51200), (5, 4097), (2, 131072)]
    tt_inputs = []
    for num_rows, n in cases:
        torch_input = _make_large_index_input(num_rows=num_rows, n=n, k=k)
        tt_inputs.append(_to_device(torch_input, device))

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        cache_entries = []
        for tt_input, (num_rows, _) in zip(tt_inputs, cases):
            tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k)
            cache_entries.append(device.num_program_cache_entries())
            _assert_index_metadata(tt_indices, [num_rows, k])

        assert cache_entries[0] > 0
        assert max(cache_entries) == min(cache_entries)
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


@pytest.mark.parametrize("k", [512, 2048])
def test_topk_large_indices_neginf_sentinel_false_returns_real_positions(device, k):
    # neginf_sentinel=False: -inf value lanes keep their REAL source column
    # (the fused stamp carries it) instead of the 0xFFFFFFFF sentinel --
    # stock ttnn.topk / torch parity for the composite routing layer.
    # n is chunk-aligned (a multiple of k) so every lane is a real column:
    # unaligned tails are -inf-padded BEFORE the index stamp, so with the
    # sentinel skipped a padded lane would carry its (out-of-range) padded
    # position -- the same looseness the stock path has around its own
    # -inf padding.
    n = 2 * k
    finite_count = 16
    torch_input = torch.full((2, n), -float("inf"), dtype=torch.bfloat16)
    finite_values = torch.arange(finite_count, dtype=torch.float32).to(torch.bfloat16)
    torch_input[:, :finite_count] = finite_values
    tt_indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device), k=k, neginf_sentinel=False
    )

    torch_indices = ttnn.to_torch(tt_indices).to(torch.int64)
    assert list(torch_indices.shape) == [2, k]
    # Finite lanes are exact (descending finite values live at columns 15..0).
    expected_prefix = torch.arange(finite_count - 1, -1, -1, dtype=torch.int64)
    assert_equal(torch_indices[:, :finite_count], expected_prefix.unsqueeze(0).repeat(2, 1))
    # -inf lanes: real, in-range, unique positions whose source value is -inf.
    inf_indices = torch_indices[:, finite_count:]
    assert inf_indices.min() >= finite_count
    assert inf_indices.max() < n
    for row in torch_indices:
        assert row.unique().numel() == k
    gathered = torch.gather(torch_input, -1, inf_indices)
    assert torch.isneginf(gathered.float()).all()


def test_topk_large_indices_column_parallel_neginf_sentinel_false(device):
    # Same contract on the column-parallel (merge-tree) factory.
    k, n, finite_count = 512, 65536, 16
    torch_input = torch.full((1, n), -float("inf"), dtype=torch.bfloat16)
    torch_input[:, :finite_count] = torch.arange(finite_count, dtype=torch.float32).to(torch.bfloat16)
    tt_indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device), k=k, num_slices=4, neginf_sentinel=False
    )

    torch_indices = ttnn.to_torch(tt_indices).to(torch.int64)
    assert list(torch_indices.shape) == [1, k]
    expected_prefix = torch.arange(finite_count - 1, -1, -1, dtype=torch.int64)
    assert_equal(torch_indices[:, :finite_count], expected_prefix.unsqueeze(0))
    inf_indices = torch_indices[:, finite_count:]
    assert inf_indices.min() >= finite_count
    assert inf_indices.max() < n
    assert torch_indices[0].unique().numel() == k


def test_topk_large_indices_neginf_sentinel_flag_in_program_cache(device):
    # The flag is part of the program hash: flipping it on the same shape
    # must compile a second program, and each variant must keep its own
    # contract when replayed from cache.
    k, n, finite_count = 512, 1024, 16
    torch_input = torch.full((2, n), -float("inf"), dtype=torch.bfloat16)
    torch_input[:, :finite_count] = torch.arange(finite_count, dtype=torch.float32).to(torch.bfloat16)
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    base_entries = device.num_program_cache_entries()
    idx_sentinel = ttnn.to_torch(
        ttnn.experimental.topk_large_indices(tt_input, k=k, neginf_sentinel=True)
    ).to(torch.int64)
    after_first = device.num_program_cache_entries()
    idx_real = ttnn.to_torch(
        ttnn.experimental.topk_large_indices(tt_input, k=k, neginf_sentinel=False)
    ).to(torch.int64)
    after_second = device.num_program_cache_entries()
    assert after_first > base_entries
    assert after_second > after_first  # distinct program, not a cache hit

    assert (idx_sentinel[:, finite_count:] == 0xFFFFFFFF).all()
    assert idx_real[:, finite_count:].max() < n
    # Replay both from cache and re-check the contracts.
    idx_sentinel2 = ttnn.to_torch(
        ttnn.experimental.topk_large_indices(tt_input, k=k, neginf_sentinel=True)
    ).to(torch.int64)
    idx_real2 = ttnn.to_torch(
        ttnn.experimental.topk_large_indices(tt_input, k=k, neginf_sentinel=False)
    ).to(torch.int64)
    assert device.num_program_cache_entries() == after_second
    device.disable_and_clear_program_cache()
    assert_equal(idx_sentinel2, idx_sentinel)
    assert_equal(idx_real2, idx_real)


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
    ],
)
def test_topk_large_indices_segmented_widths(device, k, num_rows, n):
    torch_input = _make_large_index_input(num_rows=num_rows, n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)
    _assert_topk_matches_torch(torch_input, tt_indices, k)


def test_topk_large_indices_segmented_with_values(device):
    k, n = 2048, 524288
    torch_input = _make_large_index_input(num_rows=2, n=n, k=k)
    tt_values, tt_indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device), k=k, return_values=True
    )
    _assert_topk_matches_torch(torch_input, tt_indices, k)
    ref_values, _ = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)
    assert_equal(ttnn.to_torch(tt_values).float(), ref_values.to(torch.bfloat16).float())


def test_topk_large_indices_segmented_valid_length_collapses_to_one_segment(device):
    # A wide (segmented) program whose runtime valid_length shrinks the row to
    # <= 32 chunks must run the single-segment path inside the SAME cached
    # segmented binary -- and again at a segment-boundary-crossing length.
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
    # must hash to distinct cache entries and both must stay correct.
    n, k = 65536, 2048
    single_row = _make_large_index_input(num_rows=1, n=n, k=k)
    multi_row = _make_large_index_input(num_rows=2, n=n, k=k)

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


# ---------------------------------------------------------------------------
# return_values=True: the op also emits the top-k VALUES (ROW_MAJOR BFLOAT16,
# sorted descending to match the indices; exact bf16 -inf on sentinel lanes).
# Default (return_values=False) is unchanged: a single indices tensor from the
# byte-identical indices-only program.
# ---------------------------------------------------------------------------


def _run_return_values_case(device, torch_input, k, valid_length=None):
    values, indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device), k=k, valid_length=valid_length, return_values=True
    )

    expected_shape = [torch_input.shape[0], k]
    assert values.dtype == ttnn.bfloat16
    assert values.layout == ttnn.ROW_MAJOR_LAYOUT
    assert list(values.shape) == expected_shape
    _assert_index_metadata(indices, expected_shape)

    torch_values = ttnn.to_torch(values)
    torch_indices = ttnn.to_torch(indices, dtype=torch.uint32).to(torch.int64)

    search = torch_input if valid_length is None else torch_input[:, :valid_length]
    # A prefix shorter than k is legal: only min(k, prefix) lanes are finite,
    # the rest must be exact -inf (asserted below with the sentinel lanes).
    n_finite = min(k, search.shape[-1])
    ref_values, _ = torch.topk(search.float(), n_finite, dim=-1, largest=True, sorted=True)
    # Values: exact, order included — both sides sorted descending, so this
    # holds even under ties (tie index order is unspecified, value order isn't).
    assert_equal(torch_values.float()[:, :n_finite], ref_values)
    if n_finite < k:
        assert (torch_values.float()[:, n_finite:] == -float("inf")).all()

    # Index/value consistency on non-sentinel lanes: input[index] == value.
    sentinel_mask = torch_indices == 0xFFFFFFFF
    safe_indices = torch.where(sentinel_mask, torch.zeros_like(torch_indices), torch_indices)
    gathered = torch.gather(torch_input, dim=-1, index=safe_indices)
    assert_equal(
        torch.where(sentinel_mask, torch_values, gathered),
        torch.where(sentinel_mask, torch_values, torch_values),
    )
    # Sentinel lanes must carry exact bf16 -inf values.
    assert torch.isneginf(torch_values.float()[sentinel_mask]).all()
    return torch_values, torch_indices


@pytest.mark.parametrize(
    "k,num_rows,n",
    [
        (256, 2, 2048),  # llk 512, k < window (row-parallel, contiguous writer path)
        (512, 2, 4096),  # llk 512 exact (row-parallel)
        (1024, 2, 8192),  # llk 1024 (row-parallel, reordered writer path)
        (2048, 2, 8192),  # llk 2048, 2 value tiles per sequence (row-parallel)
        (1536, 3, 51200),  # k below the llk window at production-like width (row-parallel)
    ],
)
def test_topk_large_indices_return_values_row_parallel(device, k, num_rows, n):
    torch.manual_seed(0)
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)
    _run_return_values_case(device, torch_input, k)


@pytest.mark.parametrize("k", [512, 2048])
def test_topk_large_indices_return_values_column_parallel(device, k):
    # Single wide row engages the column-parallel factory (final-core
    # materialization emits the values).
    torch.manual_seed(0)
    torch_input = torch.randn(1, 65536, dtype=torch.bfloat16)
    _run_return_values_case(device, torch_input, k)


@pytest.mark.parametrize("num_rows,n", [(2, 4096), (1, 65536)])  # row-parallel / column-parallel
def test_topk_large_indices_return_values_neginf_lanes(device, num_rows, n):
    # 16 finite values, the rest -inf: values must be the finite prefix
    # descending then exact bf16 -inf; -inf lanes carry the sentinel index.
    k, finite_count = 512, 16
    torch_input = torch.full((num_rows, n), -float("inf"), dtype=torch.bfloat16)
    torch_input[:, :finite_count] = torch.arange(finite_count, dtype=torch.float32).to(torch.bfloat16)

    torch_values, torch_indices = _run_return_values_case(device, torch_input, k)

    assert torch.isneginf(torch_values.float()[:, finite_count:]).all()
    assert (torch_indices[:, finite_count:] == 0xFFFFFFFF).all()
    expected_prefix = torch.arange(finite_count - 1, -1, -1, dtype=torch.int64).unsqueeze(0).repeat(num_rows, 1)
    assert_equal(torch_indices[:, :finite_count], expected_prefix)


@pytest.mark.parametrize("valid_length", [600, 300])
def test_topk_large_indices_return_values_valid_length(device, valid_length):
    # Bounded search: all winners must come from the prefix even though the
    # stale tail holds larger finite decoys. valid_length < k is supported by
    # design: lanes past the prefix's capacity emit -inf values + sentinel
    # indices (the docstring's "[k, last dimension]" domain is stale — the
    # short-prefix sentinel behavior is covered by the indices-only suite too).
    num_rows, n, k = 2, 8192, 512
    torch_input = torch.zeros(num_rows, n, dtype=torch.bfloat16)
    torch_input[:, :valid_length] = torch.randn(num_rows, valid_length).to(torch.bfloat16)
    torch_input[:, valid_length:] = 100.0  # stale tail decoys, must never appear

    torch_values, torch_indices = _run_return_values_case(device, torch_input, k, valid_length=valid_length)
    finite = torch_indices != 0xFFFFFFFF
    assert (torch_values.float()[finite] < 100.0).all()
    assert torch_indices[finite].max() < valid_length

    n_finite = min(k, valid_length)
    # Finite lanes must be exactly the prefix's top-n_finite values.
    ref_values, _ = torch.topk(torch_input[:, :valid_length].float(), n_finite, dim=-1, largest=True, sorted=True)
    got_finite = torch_values.float()[:, :n_finite]
    assert_equal(ref_values.sort(dim=-1).values, got_finite.sort(dim=-1).values)
    # Lanes past the prefix capacity are sentinel index + -inf value.
    if n_finite < k:
        assert (torch_indices[:, n_finite:] == 0xFFFFFFFF).all()
        assert (torch_values.float()[:, n_finite:] == -float("inf")).all()


def test_topk_large_indices_default_stays_indices_only(device):
    # Backward compatibility: without return_values the result is a single
    # indices tensor (not a tuple/list).
    torch_input = _make_bf16_exact_input(num_rows=2, n=1024)
    result = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=512)
    assert isinstance(result, ttnn.Tensor)
    _assert_topk_matches_torch(torch_input, result, 512)


def test_topk_large_indices_return_values_program_cache(device):
    # return_values is in the program hash: flipping it compiles a second
    # program; repeating either flavor reuses its cache entry.
    torch.manual_seed(0)
    num_rows, n, k = 2, 8192, 1024
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        ttnn.experimental.topk_large_indices(tt_input, k=k)
        entries_indices_only = device.num_program_cache_entries()
        values, indices = ttnn.experimental.topk_large_indices(tt_input, k=k, return_values=True)
        entries_with_values = device.num_program_cache_entries()
        assert entries_indices_only > 0
        assert entries_with_values == entries_indices_only + 1

        # Cache hits for both flavors: no growth, results still correct.
        result2 = ttnn.experimental.topk_large_indices(tt_input, k=k)
        values2, indices2 = ttnn.experimental.topk_large_indices(tt_input, k=k, return_values=True)
        assert device.num_program_cache_entries() == entries_with_values

        ref_values, _ = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)
        assert_equal(ttnn.to_torch(values).float(), ref_values)
        assert_equal(ttnn.to_torch(values2).float(), ref_values)
        assert_equal(
            ttnn.to_torch(indices, dtype=torch.uint32).to(torch.int64),
            ttnn.to_torch(indices2, dtype=torch.uint32).to(torch.int64),
        )
        _assert_index_metadata(result2, [num_rows, k])
    finally:
        device.disable_and_clear_program_cache()


# ---------------------------------------------------------------------------
# num_slices: user override of the column-parallel slice count P. Only valid
# when the column-parallel path is selected; loud errors otherwise; hashed
# (distinct P -> distinct cached program).
# ---------------------------------------------------------------------------


def test_topk_large_indices_num_slices_override_correctness(device):
    # Force P=8 at the canonical column-parallel shape (a 3-level tree; the
    # tree cost model would pick 32 -- the point here is the OVERRIDE plumbing
    # produces a correct program; distinct-P correctness is covered by the
    # cache test below). Random input => tie-safe value-multiset assertions.
    torch.manual_seed(0)
    n, k = 65536, 2048
    torch_input = torch.randn(1, n, dtype=torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, num_slices=8)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)[0]

    assert indices.min() >= 0
    assert indices.max() < n
    assert indices.unique().numel() == k

    actual_values = torch_input.float()[0][indices]
    ref_values, _ = torch.topk(torch_input.float()[0], k, largest=True, sorted=True)
    assert_equal(actual_values.sort().values, ref_values.sort().values)


@pytest.mark.parametrize("num_slices", [4, 16])
def test_topk_large_indices_num_slices_non_model_values(device, num_slices):
    # P values the cost model would NOT pick (the tree model picks the chunk
    # count, 32, here): exercises uneven chunk splits end to end.
    torch.manual_seed(1)
    n, k = 65536, 2048
    torch_input = torch.randn(1, n, dtype=torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, num_slices=num_slices)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)[0]

    assert indices.unique().numel() == k
    actual_values = torch_input.float()[0][indices]
    ref_values, _ = torch.topk(torch_input.float()[0], k, largest=True, sorted=True)
    assert_equal(actual_values.sort().values, ref_values.sort().values)


def test_topk_large_indices_num_slices_multirow_matches_row_parallel(device):
    # Explicit num_slices on a multi-row shape opts into the multi-rectangle
    # tree factory (one P-core tree per row, all concurrent). The exact top-k
    # value multiset must match the row-parallel default engine bit-for-bit
    # (index ORDER may differ only on bf16 ties, which _make_bf16_exact_input
    # rules out by construction).
    torch_input = _make_bf16_exact_input(num_rows=2, n=4096)
    tt_input = _to_device(torch_input, device)
    idx_default = ttnn.to_torch(ttnn.experimental.topk_large_indices(tt_input, k=512))
    idx_rect = ttnn.to_torch(ttnn.experimental.topk_large_indices(tt_input, k=512, num_slices=4))
    assert_equal(idx_default, idx_rect)


@pytest.mark.parametrize(
    "num_slices,match",
    [
        (1, "num_slices must be in"),  # below [2, 128]
        (129, "num_slices must be in"),  # above [2, 128]
        (48, "exceeds the row's chunk count"),  # > num_chunks (65536/2048 = 32 chunks)
    ],
)
def test_topk_large_indices_num_slices_out_of_range_rejected(device, expect_error, num_slices, match):
    torch_input = torch.randn(1, 65536, dtype=torch.bfloat16)
    with expect_error(RuntimeError, match):
        ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=2048, num_slices=num_slices)


def test_topk_large_indices_num_slices_program_cache_distinct_entries(device):
    # num_slices is in the program hash: distinct P values compile distinct
    # programs; repeating a P reuses its entry.
    torch.manual_seed(0)
    n, k = 65536, 2048
    torch_input = torch.randn(1, n, dtype=torch.bfloat16)
    tt_input = _to_device(torch_input, device)
    ref_values, _ = torch.topk(torch_input.float()[0], k, largest=True, sorted=True)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        entries = []
        for num_slices in (4, 8, 16, 8):  # trailing 8 must be a cache hit
            tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k, num_slices=num_slices)
            entries.append(device.num_program_cache_entries())
            indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)[0]
            assert indices.unique().numel() == k
            assert_equal(torch_input.float()[0][indices].sort().values, ref_values.sort().values)
        assert entries[0] > 0
        assert entries[1] == entries[0] + 1
        assert entries[2] == entries[1] + 1
        assert entries[3] == entries[2]  # P=8 rerun: cache hit, no growth
    finally:
        device.disable_and_clear_program_cache()


# ---------------------------------------------------------------------------
# Column-parallel MERGE TREE (in-place log2(P) reduction on the slice
# rectangle; slice 0 is the root). These parametrize P via num_slices so the
# tree shape is explicit; the model default path is exercised by all earlier
# column_parallel tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_slices", [4, 8, 16, 32])
@pytest.mark.parametrize("k", [512, 2048])
def test_topk_large_indices_tree_random_ties(device, num_slices, k):
    # Random bf16 has duplicate values, including ties that straddle slice and
    # tree-level boundaries; tie order is unspecified, so assert the selected
    # value multiset + index validity. n chosen so every P has >= 1 chunk per
    # slice for both LLK windows (k=512 -> 64 chunks, k=2048 -> 32 chunks).
    torch.manual_seed(0)
    n = 65536
    torch_input = torch.randn(1, n, dtype=torch.bfloat16)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, num_slices=num_slices)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)[0]

    assert indices.min() >= 0
    assert indices.max() < n
    assert indices.unique().numel() == k

    actual_values = torch_input.float()[0][indices]
    ref_values, _ = torch.topk(torch_input.float()[0], k, largest=True, sorted=True)
    assert_equal(actual_values.sort().values, ref_values.sort().values)


def test_topk_large_indices_tree_multilevel_adversarial_placement(device):
    # 3-level tree (P=8, slice width 8192). The winning block is split across
    # slices that meet only AT THE ROOT and traverse different tree depths:
    #   slice 5 (upper half of the top-k): 5 -> 4 (level 0), 4 -> 0 (level 2)
    #   slice 3 (lower half):              3 -> 2 (level 0), 2 -> 0 (level 1)
    # A mis-ordered intermediate survivor (the num_chunks=4 lesson, applied at
    # tree levels) only shows when its consumer merges it — the root sees both
    # halves through 2-deep chains. Values are globally distinct, so the final
    # rank order is asserted EXACTLY against torch.
    n, k = 65536, 2048
    region = 8192
    half = k // 2
    torch_input = torch.zeros(1, n, dtype=torch.bfloat16)

    def block(count, base):
        hi16 = (0x3F80 + base + np.arange(count, dtype=np.uint32)).astype(np.uint32)
        return torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)

    # Lower half of the winners in slice 3, upper half in slice 5 (all values
    # distinct and above the zero background).
    torch_input[0, 3 * region : 3 * region + half] = block(half, 0)
    torch_input[0, 5 * region : 5 * region + half] = block(half, half)

    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, num_slices=8)

    _, ref_indices = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)
    _assert_indices(tt_indices, ref_indices, [1, k])


def test_topk_large_indices_tree_valid_length_empty_winners(device):
    # P=16 over 32 chunks (slice width 4096); valid_length=5000 leaves slice 0
    # full, slice 1 partial, slices 2..15 empty. Empty WINNERS (2, 4, 8, ...)
    # must adopt their first incoming survivor instead of merging into an
    # empty DST; empty pure losers ship the writer's -inf scratch. Winning
    # block sits inside the prefix, decoys beyond it.
    n, k = 65536, 2048
    valid_length = 5000
    torch_input = _make_distinct_block_input(n, k, block_start=valid_length - k)
    torch_input[:, valid_length:] = 100.0  # stale decoys, must never be selected

    tt_indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device), k=k, valid_length=valid_length, num_slices=16
    )

    _, ref_indices = torch.topk(torch_input[:, :valid_length].float(), k, dim=-1, largest=True, sorted=True)
    assert int(ref_indices.max()) < valid_length
    _assert_indices(tt_indices, ref_indices, [1, k])


@pytest.mark.parametrize("num_slices", [8, 32])
def test_topk_large_indices_tree_return_values(device, num_slices):
    # Values ride the tree with the indices (every shipped survivor carries
    # both regions); the root's with-values materialization must emit the
    # exact torch value order.
    torch.manual_seed(2)
    n, k = 65536, 2048
    torch_input = torch.randn(1, n, dtype=torch.bfloat16)

    values, indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device), k=k, return_values=True, num_slices=num_slices
    )

    torch_values = ttnn.to_torch(values)
    torch_indices = ttnn.to_torch(indices, dtype=torch.uint32).to(torch.int64)
    ref_values, _ = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)

    assert_equal(torch_values.float(), ref_values)  # both sorted descending: exact even under ties
    gathered = torch.gather(torch_input, dim=-1, index=torch_indices)
    assert_equal(gathered, torch_values)
    assert torch_indices[0].unique().numel() == k


# ---------------------------------------------------------------------------
# FLEX formats (opt-ins): tile_output=True, index_dtype=uint16, and TILE-layout
# input. The compute kernels are byte-shared with the default program — only
# the reader/writer sources change — so the strongest assertion is BIT-IDENTITY
# of the logical outputs against the default ROW_MAJOR/UINT32 call on the same
# input (tie order included: same compute, same input, same result).
# ---------------------------------------------------------------------------


def _run_flex_vs_default(
    device,
    torch_input,
    k,
    *,
    tile_input=False,
    tile_output=False,
    index_dtype=None,
    valid_length=None,
    num_slices=None,
):
    tt_in_default = _to_device(torch_input, device)
    ref_values, ref_indices = ttnn.experimental.topk_large_indices(
        tt_in_default, k=k, valid_length=valid_length, return_values=True, num_slices=num_slices
    )
    ref_values_t = ttnn.to_torch(ref_values).float()
    ref_indices_t = ttnn.to_torch(ref_indices, dtype=torch.uint32).to(torch.int64)

    in_layout = ttnn.TILE_LAYOUT if tile_input else ttnn.ROW_MAJOR_LAYOUT
    tt_in_flex = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=in_layout, device=device)
    values, indices = ttnn.experimental.topk_large_indices(
        tt_in_flex,
        k=k,
        valid_length=valid_length,
        return_values=True,
        num_slices=num_slices,
        tile_output=tile_output,
        index_dtype=index_dtype,
    )

    expected_shape = list(torch_input.shape)
    expected_shape[-1] = k
    expected_layout = ttnn.TILE_LAYOUT if tile_output else ttnn.ROW_MAJOR_LAYOUT
    assert values.layout == expected_layout
    assert indices.layout == expected_layout
    assert values.dtype == ttnn.bfloat16
    assert indices.dtype == (index_dtype or ttnn.uint32)
    assert list(values.shape) == expected_shape
    assert list(indices.shape) == expected_shape

    got_values = ttnn.to_torch(values).float()
    if index_dtype == ttnn.uint16:
        got_indices = ttnn.to_torch(indices, dtype=torch.int32).to(torch.int64) & 0xFFFF
        expected_indices = ref_indices_t & 0xFFFF  # sentinel 0xFFFFFFFF truncates to 0xFFFF
    else:
        got_indices = ttnn.to_torch(indices, dtype=torch.uint32).to(torch.int64)
        expected_indices = ref_indices_t

    assert_equal(got_values, ref_values_t)
    assert_equal(got_indices, expected_indices)
    return got_values, got_indices


@pytest.mark.parametrize(
    "k,num_rows,n",
    [
        (32, 32, 4096),  # llk 512, exact 32-row tile fill (no padding rows)
        (512, 2, 4096),  # llk 512, row-parallel, 30 padding rows
        (1024, 3, 8192),  # llk 1024 (reordered CB slices), 29 padding rows
        (2048, 2, 8192),  # llk 2048, 2 sequence tiles
        (512, 33, 2048),  # rows straddle a tile-row boundary (2 tile rows, 31 pad rows)
        (512, 1, 65536),  # column-parallel tree root emission
        (2048, 1, 65536),  # column-parallel, widest window
    ],
)
def test_topk_large_indices_tile_output_matches_default(device, k, num_rows, n):
    torch.manual_seed(0)
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)
    _run_flex_vs_default(device, torch_input, k, tile_output=True)


def test_topk_large_indices_tile_output_rank3_slab_padding(device):
    # TILE padding is per 2D slab: [2, 3, n] pads each [3, k] slab to [32, k]
    # independently; the slab-aware writer addressing (slice_idx/in_slice_row)
    # and per-slab padding fill must not touch neighboring slabs' data rows.
    torch.manual_seed(1)
    torch_input = torch.randn(2, 3, 4096, dtype=torch.bfloat16)
    _run_flex_vs_default(device, torch_input, 512, tile_output=True)


@pytest.mark.parametrize(
    "k,num_rows,n",
    [
        (512, 1, 65536),  # column-parallel: the routed single-row shape (face row 0)
        (2048, 1, 65536),
        (512, 1, 2048),  # single chunk-count row-parallel single core
        (512, 3, 5000),  # odd face rows (delta=32 read staging) + non-tile-multiple width
        (512, 2, 4096),
        (2048, 33, 8192),  # rows straddle input tile rows, incl. bottom faces
        (32, 32, 37984),  # sampling-like: k=32, 32 users, non-pow2 width
    ],
)
def test_topk_large_indices_tile_input_matches_default(device, k, num_rows, n):
    torch.manual_seed(2)
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)
    _run_flex_vs_default(device, torch_input, k, tile_input=True)


def test_topk_large_indices_tile_input_last_chunk_winners(device):
    # Every winner sits at the row's END: the tail chunk's final (partial)
    # slice reads tile-padding garbage columns that the compute must mask by
    # the logical width. Ascending winners = exact rank-order reference.
    num_rows, n, k = 2, 4095, 512
    torch_input = _make_large_index_input(num_rows, n, k)
    _run_flex_vs_default(device, torch_input, k, tile_input=True)


def test_topk_large_indices_tile_input_all_equal_ties(device):
    # All-equal rows: every lane ties; indices must still be k distinct valid
    # positions and bit-identical to the default path's tie resolution.
    torch_input = torch.ones(3, 8192, dtype=torch.bfloat16)
    _, got_indices = _run_flex_vs_default(device, torch_input, 512, tile_input=True, tile_output=True)
    for row in got_indices:
        assert row.unique().numel() == 512
        assert row.max() < 8192


@pytest.mark.parametrize("num_rows,n", [(2, 4096), (1, 65024)])  # row-parallel / column-parallel (n <= 65535)
def test_topk_large_indices_uint16_matches_default(device, num_rows, n):
    torch.manual_seed(3)
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)
    _run_flex_vs_default(device, torch_input, 512, index_dtype=ttnn.uint16)


@pytest.mark.parametrize("tile_output", [False, True])
def test_topk_large_indices_uint16_neginf_sentinel(device, tile_output):
    # -inf lanes carry the 0xFFFF sentinel under the UINT16 contract (the
    # exact value a UINT32 -> UINT16 typecast of 0xFFFFFFFF produces).
    num_rows, n, k, finite_count = 2, 4096, 512, 16
    torch_input = torch.full((num_rows, n), -float("inf"), dtype=torch.bfloat16)
    torch_input[:, :finite_count] = torch.arange(finite_count, dtype=torch.float32).to(torch.bfloat16)

    got_values, got_indices = _run_flex_vs_default(
        device, torch_input, k, tile_output=tile_output, index_dtype=ttnn.uint16
    )
    assert (got_indices[:, finite_count:] == 0xFFFF).all()
    assert torch.isneginf(got_values.float()[:, finite_count:]).all()


def test_topk_large_indices_flex_route_combo(device):
    # The exact combination the ttnn.topk composite routes through: TILE input
    # (single row, no untilize), TILE output, UINT16 indices.
    torch.manual_seed(4)
    torch_input = torch.randn(1, 65024, dtype=torch.bfloat16)  # <= 65535 => u16-eligible
    _run_flex_vs_default(device, torch_input, 2048, tile_input=True, tile_output=True, index_dtype=ttnn.uint16)


def test_topk_large_indices_flex_multirow_combo(device):
    # Sampling-shaped combo: 32 users, k=32, u16 + TILE both ways.
    torch.manual_seed(5)
    torch_input = torch.randn(32, 37984, dtype=torch.bfloat16)
    _run_flex_vs_default(device, torch_input, 32, tile_input=True, tile_output=True, index_dtype=ttnn.uint16)


def test_topk_large_indices_tile_input_valid_length(device):
    # Bounded search on a TILE input: the stale tail (larger decoys) must
    # never be read or ranked.
    torch.manual_seed(6)
    num_rows, n, k, valid_length = 2, 8192, 512, 600
    torch_input = torch.zeros(num_rows, n, dtype=torch.bfloat16)
    torch_input[:, :valid_length] = torch.randn(num_rows, valid_length).to(torch.bfloat16)
    torch_input[:, valid_length:] = 100.0
    got_values, got_indices = _run_flex_vs_default(device, torch_input, k, tile_input=True, valid_length=valid_length)
    finite = got_indices != 0xFFFFFFFF
    assert (got_values.float()[finite] < 100.0).all()
    assert got_indices[finite].max() < valid_length


def test_topk_large_indices_uint16_width_too_large_rejected(device, expect_error):
    torch_input = torch.randn(1, 66560, dtype=torch.bfloat16)
    with expect_error(RuntimeError, "index_dtype=UINT16 requires"):
        ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=512, index_dtype=ttnn.uint16)


def test_topk_large_indices_tile_output_k_not_multiple_of_32_rejected(device, expect_error):
    torch_input = torch.randn(2, 4096, dtype=torch.bfloat16)
    with expect_error(RuntimeError, "tile_output requires k"):
        ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=48, tile_output=True)


def test_topk_large_indices_flex_program_cache(device):
    # tile_output / index_dtype / input layout are all program-hash inputs:
    # each flip compiles a new program; repeats are cache hits.
    torch.manual_seed(7)
    num_rows, n, k = 2, 8192, 512
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)
    ref_values, _ = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        entries = []
        configs = [
            dict(),
            dict(tile_output=True),
            dict(tile_output=True, index_dtype=ttnn.uint16),
            dict(tile_input=True, tile_output=True, index_dtype=ttnn.uint16),
            dict(tile_output=True),  # repeat: cache hit
        ]
        for cfg in configs:
            got_values, _ = _run_flex_vs_default(device, torch_input, k, **cfg)
            assert_equal(got_values, ref_values)
            entries.append(device.num_program_cache_entries())
        assert entries[1] > entries[0]
        assert entries[2] > entries[1]
        assert entries[3] > entries[2]
        assert entries[4] == entries[3]  # repeated tile_output config: no growth
    finally:
        device.disable_and_clear_program_cache()

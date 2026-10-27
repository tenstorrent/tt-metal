# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import ttnn
from loguru import logger

from tt_py_test_utils_common.utility_functions import skip_with_llk_assert, skip_with_watcher
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


def _make_spread_large_index_input(n: int, k: int) -> torch.Tensor:
    """Place k unique winners across the row so index decoding is checked in every segment."""
    values = torch.zeros((1, n), dtype=torch.bfloat16)
    stride = n // k
    winner_indices = torch.arange(k, dtype=torch.int64) * stride + stride // 2
    hi16 = (0x3F80 + np.arange(k, dtype=np.uint32)).astype(np.uint32)
    winner_values = torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)
    values[0, winner_indices] = winner_values
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
    "k,num_chunks,tail_trim",
    [
        (512, 1, 0),
        (512, 2, 0),
        (512, 31, 0),
        (512, 32, 17),  # chunk id 31 and a partial final chunk
        (512, 33, 0),  # first width that keeps the classic body for small K
        (1024, 1, 0),
        (1024, 31, 0),
        (1024, 32, 0),
        (1024, 33, 0),
        (1024, 64, 0),
        (1024, 65, 0),
        (2048, 1, 0),
        (2048, 25, 0),  # GLM warm-context chunk count
        (2048, 31, 0),
        (2048, 32, 0),
        (2048, 33, 1),
        (2048, 64, 0),
        (2048, 65, 0),
        (2048, 250, 0),  # GLM long-context chunk count
    ],
)
def test_topk_large_indices_fused_segment_boundaries(device, k, num_chunks, tail_trim):
    # The input has num_chunks K-wide chunks; tail_trim shortens the final chunk.
    n = num_chunks * k - tail_trim
    torch_input = _make_large_index_input(num_rows=1, n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


@pytest.mark.parametrize(
    "k,llk_k,num_chunks,tail_trim",
    [
        (512, 512, 32, 7),  # fused end-to-end: winners span all chunk stamps
        (768, 1024, 40, 7),  # snapped K: segmented mode and a multi-chunk partial final segment
        (2048, 2048, 40, 7),  # direct segmented mode with winners in both segments
    ],
)
def test_topk_large_indices_spread_winners_decode_global_indices(device, k, llk_k, num_chunks, tail_trim):
    n = num_chunks * llk_k - tail_trim
    torch_input = _make_spread_large_index_input(n=n, k=k)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)


def test_topk_large_indices_program_cache_separates_compute_body_modes(device):
    k = 512
    fused_input = _make_large_index_input(num_rows=1, n=32 * k, k=k)
    classic_input = _make_large_index_input(num_rows=1, n=33 * k, k=k)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        fused = ttnn.experimental.topk_large_indices(_to_device(fused_input, device), k=k)
        entries_after_fused = device.num_program_cache_entries()
        classic = ttnn.experimental.topk_large_indices(_to_device(classic_input, device), k=k)
        entries_after_classic = device.num_program_cache_entries()

        assert entries_after_fused > 0
        assert entries_after_classic == entries_after_fused + 1
        _assert_topk_matches_torch(fused_input, fused, k)
        _assert_topk_matches_torch(classic_input, classic, k)
    finally:
        device.clear_program_cache()


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
    # (case_id, num_rows, allocated_length, valid_length, k, metadata_length, expected_duration_ns)
    ("glm_indexer", 160, 524288, None, 2048, False, 2_311_000),
    ("prefill", 640, 51200, None, 1536, False, 896_000),
    ("bounded_cache", 2, 102400, 56320, 1536, False, 112_000),
    # Same shape and work as bounded_cache, with the length arriving in a tensor. The split point is then
    # derived on device; deriving it from the physical width instead halves 50 chunks at 25 while the device
    # has 28, leaving the halves at 25 and 3 and the duration far outside this band.
    ("bounded_cache_metadata", 2, 102400, 56320, 1536, True, 113_000),
]


@pytest.mark.parametrize(
    "case_id,num_rows,n,valid_length,k,metadata_length,expected_duration_ns",
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
    metadata_length,
    expected_duration_ns,
):
    """Check production-shape duration on the marker-selected host-IOMMU perf runner."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for topk_large_indices perf checks (needs IOMMU)")

    torch_input = _make_large_index_input(num_rows=num_rows, n=n, k=k)
    tt_input = _to_device(torch_input, device)
    metadata = _make_valid_length_metadata(device, valid_length if valid_length is not None else n)

    def run_topk_large_indices():
        if metadata_length:
            return ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length_tensor=metadata)
        return ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)

    measured_out, perf_record = profile_realtime_program(device, run_topk_large_indices)
    duration_ns = perf_record["duration_ns"]
    lower = expected_duration_ns * (1 - TOPK_LARGE_INDICES_PERF_MARGIN)
    upper = expected_duration_ns * (1 + TOPK_LARGE_INDICES_PERF_MARGIN)

    logger.info(
        f"topk_large_indices perf check {case_id}: duration={duration_ns / 1e6:.3f} ms "
        f"(expected {expected_duration_ns / 1e6:.3f} ms, band [{lower / 1e6:.3f}, {upper / 1e6:.3f}]), "
        f"shape=({num_rows}, {n}), valid_length={valid_length}, metadata={metadata_length}, k={k}, "
        f"profiler_runtime_id={perf_record['runtime_id']}"
    )

    del measured_out

    assert lower <= duration_ns <= upper, (
        f"{case_id} duration {duration_ns / 1e6:.3f} ms outside band "
        f"[{lower / 1e6:.3f}, {upper / 1e6:.3f}] ms "
        f"(expected {expected_duration_ns / 1e6:.3f} ms, "
        f"margin +/- {TOPK_LARGE_INDICES_PERF_MARGIN * 100:.1f}%)"
    )


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_topk_large_indices_metadata_trace_production_perf(device):
    """Compare traced scalar and metadata bounds for the production decode shape."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for topk_large_indices perf checks (needs IOMMU)")

    num_rows = 160
    n = 512 * 1024
    k = 2048
    torch_input = _make_large_index_input(num_rows=num_rows, n=n, k=k)
    tt_input = _to_device(torch_input, device)
    metadata = _make_valid_length_metadata(device, n)

    ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=n)
    ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length_tensor=metadata)
    ttnn.synchronize_device(device)

    def capture_and_measure(run_op):
        trace_id = None
        trace_capture_ended = False
        try:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            output = run_op()
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            trace_capture_ended = True
            ttnn.synchronize_device(device)
            _, record = profile_realtime_program(
                device, lambda: ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            )
            _assert_index_metadata(output, [num_rows, k])
            return record
        finally:
            if trace_id is not None:
                try:
                    if not trace_capture_ended:
                        ttnn.end_trace_capture(device, trace_id, cq_id=0)
                finally:
                    ttnn.release_trace(device, trace_id)

    scalar_record = capture_and_measure(lambda: ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=n))
    metadata_record = capture_and_measure(
        lambda: ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length_tensor=metadata)
    )
    scalar_duration_ns = scalar_record["duration_ns"]
    metadata_duration_ns = metadata_record["duration_ns"]
    ratio = metadata_duration_ns / scalar_duration_ns
    logger.info(
        f"topk_large_indices trace perf: scalar={scalar_duration_ns / 1e6:.3f} ms, "
        f"metadata={metadata_duration_ns / 1e6:.3f} ms, ratio={ratio:.4f}, "
        f"shape=({num_rows}, {n}), valid_length={n}, k={k}"
    )
    assert abs(ratio - 1.0) <= TOPK_LARGE_INDICES_PERF_MARGIN, (
        f"metadata trace duration differs from scalar by {abs(ratio - 1.0) * 100:.2f}% "
        f"(allowed {TOPK_LARGE_INDICES_PERF_MARGIN * 100:.1f}%)"
    )


def test_topk_large_indices_program_cache_shares_programs_by_row_split_class(device):
    # Row counts whose leftover rows can split across core pairs share one program, the others share a second one.
    k = 1536
    num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    cases = [(2, 3000), (5, 4097), (2, 51200), (num_cores, 3000), (2 * num_cores, 51200), (num_cores, 4097)]
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
        assert cache_entries[:3] == [cache_entries[0]] * 3
        assert cache_entries[3:] == [cache_entries[0] + 1] * 3
    finally:
        device.clear_program_cache()


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
    # valid_length=L must be numerically identical to physically slicing the row to [0, L) and running the
    # full-width op. Both compute top-k over the exact same prefix, so the indices are bit-identical.
    torch.manual_seed(0)
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)

    bounded = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)
    sliced = ttnn.experimental.topk_large_indices(_to_device(torch_input[:, :valid_length].contiguous(), device), k=k)

    bounded_t = ttnn.to_torch(bounded, dtype=torch.uint32).to(torch.int64)
    sliced_t = ttnn.to_torch(sliced, dtype=torch.uint32).to(torch.int64)
    _assert_index_metadata(bounded, [num_rows, k])
    assert_equal(bounded_t, sliced_t)


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
    # every step must reuse a single cached program. Exercise the exact fused-segment boundary and the
    # first partial chunk of the next segment while high-valued stale data remains at the physical row's end.
    k = 2048
    n = 65 * k
    torch_input = _make_large_index_input(num_rows=2, n=n, k=k)
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        entries = []
        for valid_length in (31 * k, 32 * k, 32 * k + 17, 33 * k, n, n, 32 * k, 31 * k):
            tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)
            entries.append(device.num_program_cache_entries())

            indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)
            assert indices.min() >= 0
            assert indices.max() < valid_length
            for row_indices in indices:
                assert row_indices.unique().numel() == k

        assert entries[0] > 0
        assert max(entries) == min(entries)  # no recompile as valid_length grew
    finally:
        device.clear_program_cache()


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_topk_large_indices_cached_runtime_shape_changes(device, k):
    # Same rows with a new stride; shrinking/repeated prefixes; then changed row counts.
    cases = [(2, 3072, 2048), (2, 4096, 3072), (2, 4096, 2048), (2, 4096, 2048), (3, 3072, 2048), (2, 3072, None)]
    inputs = []
    for step, (rows, width, valid_length) in enumerate(cases):
        values = _make_bf16_exact_input(rows, width)
        # Fresh buffers and different winners expose stale address or schedule updates.
        values = torch.roll(values, shifts=137 * step, dims=-1)
        inputs.append((values, _to_device(values, device), valid_length))

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        entries = []
        for values, tt_input, valid_length in inputs:
            tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)
            entries.append(device.num_program_cache_entries())
            _assert_topk_matches_torch(values[:, :valid_length], tt_indices, k)
        assert entries[0] > 0
        assert max(entries) == min(entries)
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


def _make_valid_length_metadata(device, value: int, *, on_device: bool = True):
    tensor = torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1)
    kwargs = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    if on_device:
        kwargs.update(device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(tensor, **kwargs)


@pytest.mark.parametrize(
    "k,n,valid_length",
    [(1024, 4096, 1024), (256, 2048, 600), (512, 4096, 513)],
    ids=["k1024_n4096_v1024", "k256_n2048_v600", "k512_n4096_v513"],
)
@pytest.mark.parametrize("offset", [0, 128], ids=["off0", "off128"])
def test_topk_large_indices_metadata_matches_scalar(device, k, n, valid_length, offset):
    num_rows = 2
    torch_input = _make_bf16_exact_input(num_rows, n)

    scalar = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)
    metadata = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device),
        k=k,
        valid_length_tensor=_make_valid_length_metadata(device, valid_length - offset),
        valid_length_offset=offset,
    )
    assert_equal(
        ttnn.to_torch(metadata, dtype=torch.uint32).to(torch.int64),
        ttnn.to_torch(scalar, dtype=torch.uint32).to(torch.int64),
    )

    _, expected = torch.topk(torch_input[:, :valid_length].float(), k, dim=-1, largest=True, sorted=True)
    _assert_indices(metadata, expected, [num_rows, k])


@skip_with_watcher("The reader asserts on malformed metadata under the watcher, which is the point of this case.")
def test_topk_large_indices_metadata_zero_length_does_not_hang(device):
    """A metadata tensor holding 0 with the default offset=0 must not hang the device.

    search_len = valid_length + offset is zero only when both are, and calculate_topk_bounds(0, K) yields
    num_chunks = 0 with a wrapped tail_elements ((0 - 1) * K underflows). The reader would then push zero
    chunks while the Classic / FusedEndToEnd compute bodies unconditionally cb_wait_front on the first one.
    Nothing on the host rejects it: the scalar path has TT_FATAL(valid_length > 0), the tensor path cannot
    see the value, and the reader's structural check is a watcher-gated ASSERT that compiles out.

    The reader therefore clamps search_len to input_width when the metadata is malformed, so this degrades
    to a full-row search instead of hanging. Asserting completion is the point -- a regression here shows up
    as a timeout, not a wrong answer, which is why no existing case covers it (they all use
    valid_length - offset >= 385).
    """
    k, n, num_rows = 256, 2048, 2
    torch_input = _make_bf16_exact_input(num_rows, n)

    out = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device),
        k=k,
        valid_length_tensor=_make_valid_length_metadata(device, 0),
    )
    ttnn.synchronize_device(device)

    # Clamped to the full row, so the result must match a scalar full-width search rather than be garbage.
    scalar = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=n)
    ttnn.synchronize_device(device)
    assert_equal(
        ttnn.to_torch(out, dtype=torch.uint32).to(torch.int64),
        ttnn.to_torch(scalar, dtype=torch.uint32).to(torch.int64),
    )


def test_topk_large_indices_metadata_rejects_scalar_alongside_tensor(device, expect_error):
    torch_input = _make_bf16_exact_input(2, 2048)
    with expect_error(RuntimeError, "mutually exclusive"):
        ttnn.experimental.topk_large_indices(
            _to_device(torch_input, device),
            k=256,
            valid_length=600,
            valid_length_tensor=_make_valid_length_metadata(device, 600),
        )


def test_topk_large_indices_rejects_offset_without_metadata(device, expect_error):
    torch_input = _make_bf16_exact_input(1, 2048)
    with expect_error(RuntimeError, "valid_length_offset requires valid_length_tensor"):
        ttnn.experimental.topk_large_indices(
            _to_device(torch_input, device),
            k=256,
            valid_length=1024,
            valid_length_offset=128,
        )


def test_topk_large_indices_metadata_trace_replay(device):
    # Earlier cache-reuse tests disable caching on this module-scoped device.
    device.enable_program_cache()
    num_rows, n, k = 2, 4096, 1024
    bounds = [1024, 2048, 1536, 4096]
    torch_input = _make_bf16_exact_input(num_rows, n)

    refs = {}
    for valid_length in bounds:
        output = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)
        refs[valid_length] = ttnn.to_torch(output, dtype=torch.uint32).to(torch.int64)

    tt_input = _to_device(torch_input, device)
    metadata = _make_valid_length_metadata(device, bounds[0])
    host_metadata = {
        valid_length: _make_valid_length_metadata(device, valid_length, on_device=False) for valid_length in bounds
    }

    ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length_tensor=metadata)
    ttnn.synchronize_device(device)

    trace_id = None
    trace_capture_ended = False
    try:
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        traced_output = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length_tensor=metadata)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        trace_capture_ended = True
        ttnn.synchronize_device(device)

        order = bounds + bounds[::-1] + [bounds[-1], bounds[0]]
        for replay, valid_length in enumerate(order):
            ttnn.copy_host_to_device_tensor(host_metadata[valid_length], metadata)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            actual = ttnn.to_torch(traced_output, dtype=torch.uint32).to(torch.int64)
            if not torch.equal(actual, refs[valid_length]):
                matches = [bound for bound in bounds if torch.equal(actual, refs[bound])]
                raise AssertionError(
                    f"replay {replay} (valid_length={valid_length}) does not match its eager reference; "
                    f"it matches bound(s) {matches or 'none'} instead -- "
                    f"{'stale metadata read' if matches else 'unrelated divergence'}"
                )
    finally:
        if trace_id is not None:
            try:
                if not trace_capture_ended:
                    ttnn.end_trace_capture(device, trace_id, cq_id=0)
            finally:
                ttnn.release_trace(device, trace_id)


def test_topk_large_indices_metadata_rejects_l1(device, expect_error):
    torch_input = _make_bf16_exact_input(1, 2048)
    tt_input = _to_device(torch_input, device)
    metadata = ttnn.to_memory_config(_make_valid_length_metadata(device, 1024), ttnn.L1_MEMORY_CONFIG)

    with expect_error(RuntimeError, "must be in DRAM"):
        ttnn.experimental.topk_large_indices(tt_input, k=512, valid_length_tensor=metadata)


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


@pytest.mark.parametrize("dram_read_noc", [None, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1], ids=["default", "noc0", "noc1"])
@pytest.mark.parametrize(
    "k, num_rows, n",
    [
        (512, 32, 1024),
        (2048, 4, 16384),
        # More rows than one whole round of cores: the leftover rows are split across core pairs, so the
        # writer's survivor hand-off runs on whichever NoC the reader leaves free.
        (2048, 160, 8192),
    ],
    ids=["short_rows", "multi_chunk_k2048", "row_split"],
)
def test_topk_large_indices_dram_read_noc_matches_torch(device, dram_read_noc, k, num_rows, n):
    base_row = _make_large_index_input(num_rows=1, n=n, k=k)[0]
    torch_input = torch.stack([torch.roll(base_row, shifts=row * 7) for row in range(num_rows)])
    tt_output = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, dram_read_noc=dram_read_noc)
    _assert_topk_matches_torch(torch_input, tt_output, k)


def test_topk_large_indices_program_cache_separates_dram_read_noc(device):
    torch_input = _make_large_index_input(num_rows=64, n=1024, k=512)
    tt_input = _to_device(torch_input, device)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        default_output = ttnn.experimental.topk_large_indices(tt_input, k=512)
        ttnn.synchronize_device(device)
        entries_after_default = device.num_program_cache_entries()

        # NOC_0 is Blackhole's preferred DRAM-read NoC, so naming it explicitly resolves to the default program.
        noc0_output = ttnn.experimental.topk_large_indices(tt_input, k=512, dram_read_noc=ttnn.NOC.NOC_0)
        ttnn.synchronize_device(device)
        assert device.num_program_cache_entries() == entries_after_default

        noc1_output = ttnn.experimental.topk_large_indices(tt_input, k=512, dram_read_noc=ttnn.NOC.NOC_1)
        ttnn.synchronize_device(device)
        assert device.num_program_cache_entries() > entries_after_default

        for output in (default_output, noc0_output, noc1_output):
            _assert_topk_matches_torch(torch_input, output, 512)
    finally:
        device.disable_and_clear_program_cache()


def test_topk_large_indices_restricted_grid_cache_hit_rebinds_shape_and_valid_length(device):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 10:
        pytest.skip(f"80-core top-k profile requires at least an 8x10 worker grid, got {grid}")

    grid_80 = _rect_core_grid(0, 7, 9)
    manager = device.create_sub_device_manager([ttnn.SubDevice([grid_80])], 0)
    cases = []
    # Row counts below 80 and at or above 80 select different programs, so the cache-hit path must cover both.
    for num_rows, n, valid_length in ((2, 1024, 512), (5, 8192, 4096), (163, 4096, 2048), (90, 2048, 1024)):
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
        cache_entries = []
        for torch_input, tt_input, valid_length in cases:
            output = ttnn.experimental.topk_large_indices(
                tt_input,
                k=512,
                valid_length=valid_length,
                subdevice_id=ttnn.SubDeviceId(0),
                sub_core_grids=grid_80,
            )
            ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0)])
            cache_entries.append(device.num_program_cache_entries())
            _, expected = torch.topk(
                torch_input[:, :valid_length].float(),
                512,
                dim=-1,
                largest=True,
                sorted=True,
            )
            _assert_indices(output, expected, [torch_input.shape[0], 512])

        assert cache_entries[0] > 0
        assert cache_entries == [cache_entries[0]] * 2 + [cache_entries[0] + 1] * 2
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)
        device.disable_and_clear_program_cache()


def _ceil32(x: int) -> int:
    return ((x + 31) // 32) * 32


@pytest.mark.parametrize(
    # k travels with the case: it must fit inside ceil32(valid_end), which the small ends do not leave
    # room for at a single shared k.
    "n,valid_end,k",
    [
        (2048, 1024, 256),  # already 32-aligned: ceil32 is a no-op
        (2048, 1000, 256),  # NOT 32-aligned -> rounds up to 1024
        (2048, 1025, 256),  # one past an alignment boundary -> rounds to the NEXT one, 1056
        (2048, 1023, 256),  # one short of it -> still 1024
        (4096, 100, 64),  # small and non-aligned -> 128
        (2048, 2048, 256),  # equals the row: the cap must not bite
    ],
    ids=["aligned1024", "unaligned1000", "unaligned1025", "unaligned1023", "small100", "full2048"],
)
def test_topk_large_indices_valid_end_matches_scalar_bound(device, n, valid_end, k):
    """valid_end must reproduce the SCALAR bound exactly: min(valid_length, ceil32(valid_end)).

    Parity is the whole point -- the score op and this op derive their extents from the same metadata
    word, and if the two disagree a looser score with a tighter top-k drops real keys while the reverse
    ranks a stale tail.
    """
    num_rows = 2
    torch_input = _make_bf16_exact_input(num_rows, n)
    expected_bound = min(n, _ceil32(valid_end))

    scalar = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=expected_bound)
    bounded = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device),
        k=k,
        valid_length_tensor=_make_valid_length_metadata(device, n),
        valid_end_tensor=_make_valid_length_metadata(device, valid_end),
    )
    assert_equal(
        ttnn.to_torch(bounded, dtype=torch.uint32).to(torch.int64),
        ttnn.to_torch(scalar, dtype=torch.uint32).to(torch.int64),
    )
    _, expected = torch.topk(torch_input[:, :expected_bound].float(), k, dim=-1, largest=True, sorted=True)
    _assert_indices(bounded, expected, [num_rows, k])


@pytest.mark.parametrize("valid_end", [1000, 1024], ids=["unaligned1000", "aligned1024"])
def test_topk_large_indices_valid_end_excludes_stale_tail(device, valid_end):
    """Entries past ceil32(valid_end) must never be selected, even when they are the largest in the row.

    This is the failure the bound exists to prevent, and it is invisible to a [0, real_len) golden
    comparison: the stale tail is exactly the region such a check does not look at.
    """
    num_rows, n, k = 2, 2048, 64
    bound = _ceil32(valid_end)
    torch_input = _make_bf16_exact_input(num_rows, n)
    # Plant values strictly larger than anything in [0, bound) in the region the bound must exclude.
    torch_input[:, bound:] = torch.finfo(torch.bfloat16).max

    tt_indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device),
        k=k,
        valid_length_tensor=_make_valid_length_metadata(device, n),
        valid_end_tensor=_make_valid_length_metadata(device, valid_end),
    )
    got = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)
    assert int(got.max()) < bound, (
        f"top-k selected index {int(got.max())} at/beyond the bound {bound} "
        f"(valid_end={valid_end}) -- the stale tail was not excluded"
    )
    _, expected = torch.topk(torch_input[:, :bound].float(), k, dim=-1, largest=True, sorted=True)
    _assert_indices(tt_indices, expected, [num_rows, k])


def test_topk_large_indices_valid_end_omitted_is_unchanged(device):
    """Omitting valid_end_tensor must leave the metadata path byte-identical to what it was before.

    The bound is opt-in: absent, its common-arg slot is 0 and the reader skips the cap entirely, so the
    result must match a plain valid_length_tensor run.
    """
    num_rows, n, k = 2, 2048, 256
    torch_input = _make_bf16_exact_input(num_rows, n)

    without = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device), k=k, valid_length_tensor=_make_valid_length_metadata(device, n)
    )
    # Supplying a bound at/after the row end must be the same as not supplying one at all.
    with_noop_bound = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device),
        k=k,
        valid_length_tensor=_make_valid_length_metadata(device, n),
        valid_end_tensor=_make_valid_length_metadata(device, n),
    )
    assert_equal(
        ttnn.to_torch(with_noop_bound, dtype=torch.uint32).to(torch.int64),
        ttnn.to_torch(without, dtype=torch.uint32).to(torch.int64),
    )


def test_topk_large_indices_valid_end_varies_across_trace_replays(device):
    """One captured program must serve DIFFERENT real ends: the value is read on-device per dispatch.

    This is the property that makes the bound trace-safe. If the value were baked in at capture (a host
    runtime arg, or a compile-time flag), every replay would reuse the end that was live during capture
    and a multi-chunk traced prefill would score the wrong window with nothing failing.
    """
    device.enable_program_cache()
    num_rows, n, k = 2, 2048, 64
    torch_input = _make_bf16_exact_input(num_rows, n)
    tt_input = _to_device(torch_input, device)
    length_meta = _make_valid_length_metadata(device, n)
    vend_meta = _make_valid_length_metadata(device, 1024)

    def run():
        return ttnn.experimental.topk_large_indices(
            tt_input, k=k, valid_length_tensor=length_meta, valid_end_tensor=vend_meta
        )

    run()  # warm the program cache before capture
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for valid_end in (1024, 512, 1000, 2048):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([valid_end], dtype=torch.int64).reshape(1, 1, 1, 1),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                vend_meta,
            )
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            bound = _ceil32(valid_end)
            _, expected = torch.topk(torch_input[:, :bound].float(), k, dim=-1, largest=True, sorted=True)
            got = ttnn.to_torch(output, dtype=torch.uint32).to(torch.int64)
            assert_equal(got, expected.to(torch.int64))
            assert int(got.max()) < bound, f"replay with valid_end={valid_end} ranked past its bound {bound}"
    finally:
        ttnn.release_trace(device, trace_id)


# ---------------------------------------------------------------------------
# Row split: with fewer rows than cores, the leftover rows are halved across core pairs and the second half's
# survivor is merged over the NoC by the first. The Classic body has no split form, so every case here picks a
# (k, n) whose snapped K' keeps it on a fused body.
# ---------------------------------------------------------------------------


def _assert_topk_values_match_torch(torch_input: torch.Tensor, tt_indices: ttnn.Tensor, k: int) -> None:
    """Random data has ties, so compare the values gathered by the returned indices, not the indices."""
    n = torch_input.shape[-1]
    _assert_index_metadata(tt_indices, list(torch_input.shape[:-1]) + [k])
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64).reshape(-1, k)
    assert indices.min() >= 0
    assert indices.max() < n
    for row_indices in indices:
        assert row_indices.unique().numel() == k

    flat_input = torch_input.float().reshape(-1, n)
    actual_values = torch.gather(flat_input, dim=-1, index=indices)
    ref_values, _ = torch.topk(flat_input, k, dim=-1, largest=True, sorted=True)
    assert_equal(actual_values, ref_values)


def _make_random_winner_rows(num_rows: int, n: int, k: int, seed: int) -> torch.Tensor:
    """k distinct winners at random positions of each row, in random order, zeros elsewhere: the top k indices have
    one answer at any width (bfloat16 has too few distinct values for a wide row of them)."""
    generator = torch.Generator().manual_seed(seed)
    hi16 = (0x3F80 + np.arange(k, dtype=np.uint32)).astype(np.uint32)
    winner_values = torch.from_numpy((hi16 << 16).view(np.float32).copy()).to(torch.bfloat16)
    values = torch.zeros((num_rows, n), dtype=torch.bfloat16)
    for row in range(num_rows):
        positions = torch.randperm(n, generator=generator)[:k]
        values[row, positions] = winner_values[torch.randperm(k, generator=generator)]
    return values


def _make_tie_saturated_rows(num_rows: int, n: int, distinct: int, seed: int) -> torch.Tensor:
    """Rows drawn from `distinct` values only, so almost every compare-exchange in the sort is a tie.

    bfloat16 already collides heavily at these widths; drawing from a tiny pool forces the tie path
    rather than leaving it to chance."""
    generator = torch.Generator().manual_seed(seed)
    pool = torch.arange(distinct, dtype=torch.float32).to(torch.bfloat16)
    return pool[torch.randint(0, distinct, (num_rows, n), generator=generator)]


@pytest.mark.parametrize("distinct", [2, 8, 64], ids=lambda v: f"distinct={v}")
@pytest.mark.parametrize("k,n", [(2048, 131072)], ids=lambda v: str(v))
def test_topk_large_indices_tie_saturated_is_deterministic(device, distinct, k, n):
    """Tie-break order is part of the op's observable behaviour, and the sort networks decide it.

    The other value tests gather through the returned indices precisely so ties cannot fail them, so
    nothing else here pins what the op does when operands compare equal. Draw from a tiny value pool
    so nearly every compare-exchange is a tie, then require that the returned indices are stable
    across runs and still select a correct top-k. A scheduling change that altered which of two equal
    operands survives, or that made the choice depend on timing, fails here and nowhere else."""
    torch_input = _make_tie_saturated_rows(num_rows=4, n=n, distinct=distinct, seed=11)
    tt_input = _to_device(torch_input, device)

    first_indices = ttnn.experimental.topk_large_indices(tt_input, k=k)
    first = ttnn.to_torch(first_indices, dtype=torch.uint32)
    second = ttnn.to_torch(ttnn.experimental.topk_large_indices(tt_input, k=k), dtype=torch.uint32)

    assert torch.equal(first, second), "tie-break order differed between two runs on identical input"
    _assert_topk_values_match_torch(torch_input, first_indices, k)


# (k, n) pairs that stay on a fused body: K' 512 only up to 32 chunks, K' 1024 and 2048 at any width.
ROW_SPLIT_SHAPES = [(512, 16384), (1024, 65536), (1024, 131072), (2048, 131072)]


@pytest.mark.parametrize("extra_rows", [1, 3])
@pytest.mark.parametrize(
    "k,n",
    [
        (1024, 32768 + 100),
        (2048, 65536),
        # K' 2048 rows past one 32-chunk segment, with leftover rows to halve: the production indexer's program
        # class (long_rows + row_split + split_whole_rows). Nothing else in this file builds it, and it is the
        # largest kernel the op produces, so it is the case that bounds the kernel config buffer.
        (2048, 34 * 2048),
    ],
)
def test_topk_large_indices_leftover_rows_split(device, k, n, extra_rows):
    grid = device.compute_with_storage_grid_size()
    num_rows = grid.x * grid.y + extra_rows
    torch.manual_seed(3)
    torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)
    # Near the row end, so it is past the cut whatever block boundary half_bounds picks: the top index can
    # only be right if the second half's survivor reached the first half over the NoC.
    winner = n - 3
    torch_input[-extra_rows:, winner] = 250.0
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_values_match_torch(torch_input, tt_indices, k)
    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)
    assert (indices[-extra_rows:, 0] == winner).all()


@pytest.mark.parametrize(
    "k,n,valid_length",
    [
        # One valid chunk: the cut lands at 0, so the FIRST half is empty and folds in as one -inf chunk
        # instead of stalling the body on an empty stream.
        (2048, 131072, 2048),
        (1024, 131072, 1024),
        # Both halves non-empty, for contrast.
        (1024, 131072, 40000),
        (2048, 131072, 3000),
    ],
)
def test_topk_large_indices_row_split_metadata_empty_half(device, k, n, valid_length):
    # The length only reaches the device, so half_bounds cuts the row there. A half left with no valid chunk
    # reduces as one all -inf chunk.
    torch.manual_seed(3)
    torch_input = torch.randn(1, n, dtype=torch.bfloat16)
    torch_input[:, valid_length:] = 300.0
    tt_indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device),
        k=k,
        valid_length_tensor=_make_valid_length_metadata(device, valid_length),
    )

    indices = ttnn.to_torch(tt_indices, dtype=torch.uint32).to(torch.int64)
    assert int(indices.max()) < valid_length
    _assert_topk_values_match_torch(torch_input[:, :valid_length], tt_indices, k)


@pytest.mark.parametrize("k,n", ROW_SPLIT_SHAPES, ids=lambda v: str(v))
def test_topk_large_indices_row_split_matches_single_core(device, k, n):
    # One core cannot halve a row, so its result is the reference for the core-pair merge.
    torch_input = _make_spread_large_index_input(n=n, k=k)
    tt_input = _to_device(torch_input, device)

    split = ttnn.experimental.topk_large_indices(tt_input, k=k)
    single = ttnn.experimental.topk_large_indices(tt_input, k=k, sub_core_grids=_rect_core_grid(0, 0, 0))

    assert_equal(
        ttnn.to_torch(split, dtype=torch.uint32).to(torch.int64),
        ttnn.to_torch(single, dtype=torch.uint32).to(torch.int64),
    )
    _assert_topk_matches_torch(torch_input, split, k)


@pytest.mark.parametrize(
    "k,num_chunks,tail_trim",
    [
        # chunks % 32 == 1: a one-chunk last segment, the one case that cannot run across the threads.
        (1024, 33, 0),
        (2048, 33, 1),
        # A second 32-chunk block before the remainder, so the segment's block base has to decode past block 0.
        (2048, 65, 0),
    ],
)
def test_topk_large_indices_one_core_segmented_rows(device, k, num_chunks, tail_trim):
    # One core keeps the whole row, so its 32-chunk segments run in sequence and the K' 2048 ones round-trip the
    # accumulator through L1 while the next segment is split across the threads.
    n = num_chunks * k - tail_trim
    torch_input = _make_random_winner_rows(2, n, k, seed=num_chunks)
    tt_indices = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device), k=k, sub_core_grids=_rect_core_grid(0, 0, 0)
    )

    _assert_topk_matches_torch(torch_input, tt_indices, k)


@pytest.mark.parametrize(
    "num_rows,n,k",
    [
        (None, 36 * 2048 - 7, 2048),  # a later segment of four chunks round-trips the accumulator through L1
        (3, 102400, 2048),  # halves of a row merged over the NoC
    ],
    ids=["seg_36_acc", "split_3"],
)
def test_topk_large_indices_leaves_the_next_op_intact(device, num_rows, n, k):
    # The raw pack/unpack round trips (the L1 accumulator and the survivor transfer) must not leave unpacker state
    # that a later program's init fails to reset.
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    if num_rows is None:
        num_rows = num_cores
    torch_input = _make_spread_large_index_input(n=n, k=k).repeat(num_rows, 1)
    tt_input = _to_device(torch_input, device)
    a = torch.randint(-8, 8, (1, 1, 32 * num_cores, 64)).to(torch.bfloat16)
    b = torch.randint(-8, 8, (1, 1, 32 * num_cores, 64)).to(torch.bfloat16)
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(3):
        tt_indices = ttnn.experimental.topk_large_indices(tt_input, k=k)
        total = ttnn.to_torch(ttnn.add(tt_a, tt_b))
        assert_equal(total, a + b)
    _assert_topk_matches_torch(torch_input, tt_indices, k)


@pytest.mark.parametrize("k", [512, 2048])
def test_topk_large_indices_program_cache_separates_one_chunk_rows(device, k):
    # Rows of one chunk compile a kernel without the multi-chunk path; wider rows must not reuse it.
    one_chunk_input = _make_large_index_input(num_rows=1, n=k, k=k)
    two_chunk_input = _make_large_index_input(num_rows=1, n=2 * k, k=k)

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        one_chunk = ttnn.experimental.topk_large_indices(_to_device(one_chunk_input, device), k=k)
        entries_after_one_chunk = device.num_program_cache_entries()
        two_chunk = ttnn.experimental.topk_large_indices(_to_device(two_chunk_input, device), k=k)
        entries_after_two_chunk = device.num_program_cache_entries()

        assert entries_after_one_chunk > 0
        assert entries_after_two_chunk == entries_after_one_chunk + 1
        _assert_topk_matches_torch(one_chunk_input, one_chunk, k)
        _assert_topk_matches_torch(two_chunk_input, two_chunk, k)
    finally:
        device.clear_program_cache()


@pytest.mark.parametrize("num_rows", [2, 5])
@pytest.mark.parametrize("num_chunks", [33, 40])
def test_topk_large_indices_classic_body_multi_row_covers_every_row(device, num_rows, num_chunks):
    # The Classic body (K' 512 past 32 chunks) has no halved form, so derive_core_row_assignments must fall back
    # to contiguous whole-row ranges. Producing the halved layout for a program without the transfer code makes
    # every core read start_row 0 and leaves the last rows unwritten -- which one row cannot detect, because the
    # duplicate writes all carry the same correct bytes.
    k = 512
    n = num_chunks * k
    torch_input = _make_random_winner_rows(num_rows, n, k, seed=num_chunks + num_rows)
    tt_indices = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k)

    _assert_topk_matches_torch(torch_input, tt_indices, k)

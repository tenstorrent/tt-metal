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
        device.disable_and_clear_program_cache()


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
    ("prefill", 640, 51200, None, 1536, 1_286_400),
    ("bounded_cache", 2, 102400, 56320, 1536, 242_560),
]


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
    k = 1536
    cases = [(2, 3000), (640, 51200), (5, 4097)]
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
        for valid_length in (31 * k, 32 * k, 32 * k + 17, 33 * k, n):
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
        assert max(cache_entries) == min(cache_entries)
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)
        device.disable_and_clear_program_cache()

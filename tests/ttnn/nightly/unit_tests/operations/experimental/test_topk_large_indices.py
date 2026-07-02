# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import ttnn

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
    with pytest.raises(TypeError):
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

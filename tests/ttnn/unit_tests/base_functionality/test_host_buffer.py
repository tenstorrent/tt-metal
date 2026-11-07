# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only HostBuffer sequence access: valid indices return bytes, out-of-range indices raise."""

import numpy as np
import pytest
import torch
import ttnn


def make_host_buffer():
    values = torch.arange(32, dtype=torch.int32)
    tensor = ttnn.from_torch(values, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    buffer = tensor.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))
    expected = values.numpy().tobytes()
    return buffer, expected


def test_host_buffer_getitem_returns_bytes():
    buffer, expected = make_host_buffer()
    assert len(buffer) == len(expected)
    for index in (0, 1, 3, 4, len(expected) // 2, len(expected) - 1):
        assert buffer[index] == expected[index]


def test_host_buffer_iteration_matches_bytes():
    buffer, expected = make_host_buffer()
    assert bytes(buffer[i] for i in range(len(buffer))) == expected
    assert np.array_equal(np.from_dlpack(buffer), np.frombuffer(expected, dtype=np.uint8))


@pytest.mark.parametrize("offset", [0, 1, 640000, 10**7, 10**9])
def test_host_buffer_getitem_out_of_range_raises(expect_error, offset):
    buffer, _ = make_host_buffer()
    with expect_error(IndexError, "HostBuffer index out of range"):
        buffer[len(buffer) + offset]


@pytest.mark.parametrize("source", [bytes([1, 2, 3, 4]), bytearray([1, 2, 3, 4]), [1, 2, 3, 4], (1, 2, 3, 4)])
def test_host_buffer_constructor_from_python_data(source):
    buffer = ttnn._ttnn.tensor.HostBuffer(source)
    assert len(buffer) == 4
    assert [buffer[i] for i in range(4)] == [1, 2, 3, 4]
    assert list(buffer) == [1, 2, 3, 4]


def test_host_buffer_constructor_empty_and_full_range():
    assert len(ttnn._ttnn.tensor.HostBuffer()) == 0
    assert list(ttnn._ttnn.tensor.HostBuffer()) == []
    data = bytes(range(256))
    assert list(ttnn._ttnn.tensor.HostBuffer(data)) == list(data)


def test_host_buffer_list_matches_raw_bytes():
    buffer, expected = make_host_buffer()
    assert list(buffer) == list(expected)

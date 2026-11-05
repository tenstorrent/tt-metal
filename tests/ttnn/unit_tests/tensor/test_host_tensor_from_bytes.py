# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.host_tensor_from_bytes(buffer, tensor_spec)`` wraps a Python byte buffer as a host tensor without copying it.

numpy cannot export a bfloat16 array through DLPack or the buffer protocol, and the ndarray constructor converts
values rather than reinterpreting bytes, so bytes that already are a tensor's storage (a checkpoint's row-major
shards, say) had no zero-copy way onto the host side. ``host_tensor_from_bytes`` reinterprets the bytes under the given spec
and pins the Python object for as long as the tensor's buffer lives.
"""

import gc

import numpy as np
import torch

import ttnn


def _spec(shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.TensorSpec(ttnn.Shape(list(shape)), dtype, layout, ttnn.DRAM_MEMORY_CONFIG)


def _bf16_bytes(tensor):
    """The bfloat16 bytes of a contiguous torch tensor, as a flat uint8 array sharing its memory."""
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint8).reshape(-1)


def test_host_tensor_from_bytes_reinterprets_the_bytes_without_copying():
    shape = (1, 1, 64, 32)
    expected = torch.randn(shape).to(torch.bfloat16)
    raw = _bf16_bytes(expected)

    tensor = ttnn.host_tensor_from_bytes(raw, _spec(shape))

    assert list(tensor.shape) == list(shape)
    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    torch.testing.assert_close(ttnn.to_torch(tensor), expected, rtol=0, atol=0)
    view = np.from_dlpack(tensor.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0)))
    assert view.ctypes.data == raw.ctypes.data, "the tensor borrows the buffer rather than copying it"


def test_host_tensor_from_bytes_keeps_the_buffer_alive():
    shape = (1, 1, 32, 32)

    def build():
        source = torch.randn(shape).to(torch.bfloat16)
        return ttnn.host_tensor_from_bytes(_bf16_bytes(source), _spec(shape)), source.clone()

    tensor, expected = build()  # the only Python reference to the raw bytes is the one the tensor pins
    gc.collect()

    torch.testing.assert_close(ttnn.to_torch(tensor), expected, rtol=0, atol=0)


def test_host_tensor_from_bytes_checks_dtype_shape_and_size(expect_error):
    shape = (1, 1, 32, 32)
    raw = _bf16_bytes(torch.zeros(shape, dtype=torch.bfloat16))

    with expect_error(RuntimeError, "uint8"):
        ttnn.host_tensor_from_bytes(raw.view(np.uint16), _spec(shape))
    with expect_error(RuntimeError, "uint8"):
        ttnn.host_tensor_from_bytes(raw.reshape(32, 64), _spec(shape))
    with expect_error(RuntimeError, "bytes"):
        ttnn.host_tensor_from_bytes(raw[:-2], _spec(shape))

    # A TILE tensor takes its padded bytes: [1, 1, 1, 32] is one 32x32 tile.
    tile_spec = _spec((1, 1, 1, 32), layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "bytes"):
        ttnn.host_tensor_from_bytes(_bf16_bytes(torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)), tile_spec)
    tile = ttnn.host_tensor_from_bytes(np.zeros(32 * 32 * 2, dtype=np.uint8), tile_spec)
    assert list(tile.shape) == [1, 1, 1, 32]
    assert list(tile.padded_shape) == [1, 1, 32, 32]

    # Block floats are sized by the device runtime, which a host-only constructor must not start.
    bfp8_spec = _spec(shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "unsupported data type"):
        ttnn.host_tensor_from_bytes(np.zeros(32 * 32 * 2, dtype=np.uint8), bfp8_spec)


def test_host_tensor_from_bytes_float32_round_trips_through_a_device(device):
    shape = (1, 1, 32, 32)
    expected = torch.randn(shape, dtype=torch.float32)
    raw = expected.contiguous().numpy().view(np.uint8).reshape(-1)

    tensor = ttnn.host_tensor_from_bytes(raw, _spec(shape, dtype=ttnn.float32))

    torch.testing.assert_close(ttnn.to_torch(tensor), expected, rtol=0, atol=0)
    on_device = tensor.to(device)
    torch.testing.assert_close(ttnn.to_torch(on_device.cpu()), expected, rtol=0, atol=0)

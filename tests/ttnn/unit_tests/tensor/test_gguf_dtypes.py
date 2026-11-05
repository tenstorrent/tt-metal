# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GGUF dtypes: from_gguf / to_gguf round trips on the host, serialization, and GGUF tensors failing cleanly in ops
that don't handle them. The device cases need a Blackhole device."""

import numpy as np
import pytest

import ttnn

gguf = pytest.importorskip("gguf")

# [K, N] per case: two pages along K for every format (lcm of page rows and block size), three N tiles.
K_UNIT = {
    "Q8_0": 32,
    "Q4_0": 32,
    "IQ4_NL": 32,
    "Q4_1": 32,
    "MXFP4": 64,
    "NVFP4": 64,
    "Q5_0": 64,
    "Q5_1": 64,
    "Q2_0": 64,
    "Q1_0": 128,
}
N = 96


def _name(dtype):
    return dtype.name.removeprefix("GGUF_")


def _qtype(dtype):
    qtype = getattr(gguf.GGMLQuantizationType, _name(dtype), None)
    if qtype is None:
        pytest.skip(f"gguf-py has no {_name(dtype)}")
    return qtype


def _blocks(dtype, k, n, seed=0):
    """The GGUF bytes of an [N, K] weight: gguf-py's quantization where it has one, random bytes otherwise (the
    conversion only moves bits, so any bytes round-trip)."""
    qtype = _qtype(dtype)
    rng = np.random.default_rng(seed)
    try:
        return gguf.quants.quantize(rng.standard_normal((n, k), dtype=np.float32), qtype)
    except NotImplementedError:
        block_elems, block_bytes = gguf.GGML_QUANT_SIZES[qtype]
        return rng.integers(0, 256, size=(n, k // block_elems * block_bytes), dtype=np.uint8)


def _k(dtype):
    return 2 * K_UNIT.get(_name(dtype), 256)


@pytest.mark.parametrize("dtype", ttnn.GGUF_DTYPES, ids=_name)
def test_round_trip(dtype):
    k = _k(dtype)
    blocks = _blocks(dtype, k, N)
    tensor = ttnn.from_gguf(blocks, (k, N), dtype)
    assert tensor.dtype == dtype
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert tuple(tensor.shape) == (k, N)
    out = ttnn.to_gguf(tensor)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, blocks.reshape(-1))


@pytest.mark.parametrize("dtype", [ttnn.gguf_q4_k, ttnn.gguf_iq2_xxs, ttnn.gguf_q8_0], ids=_name)
def test_dump_load(dtype, tmp_path):
    k = _k(dtype)
    blocks = _blocks(dtype, k, N)
    path = tmp_path / "w.tensorbin"
    ttnn.dump_tensor(path, ttnn.from_gguf(blocks, (k, N), dtype))
    loaded = ttnn.load_tensor(path)
    assert loaded.dtype == dtype
    np.testing.assert_array_equal(ttnn.to_gguf(loaded), blocks.reshape(-1))


def test_rejects_wrong_sizes(expect_error):
    dtype = ttnn.gguf_q4_k
    blocks = _blocks(dtype, 256, 64)
    with expect_error(RuntimeError, "bytes of blocks"):
        ttnn.from_gguf(blocks[:, :-1], (256, 64), dtype)
    with expect_error(RuntimeError, "N % 32 == 0"):
        ttnn.from_gguf(blocks[:48], (256, 48), dtype)


def test_values_need_gguf_py(expect_error):
    dtype = ttnn.gguf_q4_0
    tensor = ttnn.from_gguf(_blocks(dtype, 64, 32), (64, 32), dtype)
    with expect_error(RuntimeError, "to_gguf"):
        ttnn.to_torch(tensor)
    with expect_error(RuntimeError, "Unsupported"):
        ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)


def _width_sharded_dram(device, k, n_tiles):
    banks = device.dram_grid_size().x
    shard_n = 32 * -(-n_tiles // banks)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [k, shard_n], ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("dtype", [ttnn.gguf_q4_k, ttnn.gguf_q6_k, ttnn.gguf_iq4_xs, ttnn.gguf_q8_0], ids=_name)
def test_device_round_trip(device, dtype):
    k, n_tiles = _k(dtype), 13  # not a multiple of the bank count: the last shard is partly empty
    blocks = _blocks(dtype, k, 32 * n_tiles)
    memory_config = _width_sharded_dram(device, k, n_tiles)
    tensor = ttnn.from_gguf(blocks, (k, 32 * n_tiles), dtype, device=device, memory_config=memory_config)
    assert ttnn.is_tensor_storage_on_device(tensor)
    np.testing.assert_array_equal(ttnn.to_gguf(tensor), blocks.reshape(-1))


def test_device_unaware_op_fails(device, expect_error):
    dtype = ttnn.gguf_q4_0
    tensor = ttnn.from_gguf(_blocks(dtype, 64, 64), (64, 64), dtype, device=device)
    with expect_error(RuntimeError, "not supported for binary operation"):
        ttnn.add(tensor, tensor)

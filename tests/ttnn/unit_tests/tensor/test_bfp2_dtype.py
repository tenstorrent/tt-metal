# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for the BFLOAT2_B (2-bit block float) TTNN data type.

BFLOAT2_B stores a sign and one magnitude bit per element, with an 8-bit exponent shared by each 16-element face row,
so it holds {-1, 0, +1} times a power of two exactly. The host tests need no device. The device tests need a Wormhole
or Blackhole device; Quasar has no BFP formats.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _ternary(shape, scale=1.0, seed=0):
    torch.manual_seed(seed)
    return torch.randint(-1, 2, shape, dtype=torch.float32) * scale


def _round_trip(x):
    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)
    assert tt_tensor.dtype == ttnn.bfloat2_b
    assert tt_tensor.layout == ttnn.TILE_LAYOUT
    return ttnn.to_torch(tt_tensor)


def test_bfp2_alias():
    assert ttnn.bfloat2_b == ttnn.DataType.BFLOAT2_B


@pytest.mark.parametrize("shape", [(32, 32), (64, 128), (2, 3, 64, 96), (14, 47)])
@pytest.mark.parametrize("scale", [1.0, 0.125, 8.0])
def test_bfp2_ternary_round_trip_is_exact(shape, scale):
    # Every face row of a scaled ternary tensor shares one exponent, so packing loses nothing.
    x = _ternary(shape, scale)

    y = _round_trip(x)

    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.equal(y, x)


def test_bfp2_from_bfloat16():
    x = _ternary((64, 64))
    assert torch.equal(_round_trip(x.to(torch.bfloat16)), x)


def test_bfp2_quantizes_general_values():
    torch.manual_seed(0)
    x = torch.randn((64, 64), dtype=torch.float32)

    y = _round_trip(x)

    # Values that are already BFLOAT2_B-quantized must survive another round trip unchanged.
    assert torch.equal(_round_trip(y), y)
    nonzero = y != 0
    assert torch.equal(torch.sign(y[nonzero]), torch.sign(x[nonzero]))


def test_bfp2_requires_tile_layout(expect_error):
    x = _ternary((32, 32))
    with expect_error(RuntimeError, "Layout must be Layout::TILE"):
        ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.ROW_MAJOR_LAYOUT)


def test_bfp2_defaults_to_tile_layout():
    assert ttnn.from_torch(_ternary((32, 32)), dtype=ttnn.bfloat2_b).layout == ttnn.TILE_LAYOUT


@pytest.mark.parametrize("other", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_bfp2_to_dtype(other):
    # Ternary values are exact in every one of these formats, so conversions in both directions are lossless.
    x = _ternary((64, 64))
    bfp2_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)

    converted = ttnn.to_dtype(bfp2_tensor, other)
    assert converted.dtype == other
    assert torch.equal(ttnn.to_torch(converted).float(), x)

    back = ttnn.to_dtype(ttnn.from_torch(x, dtype=other, layout=ttnn.TILE_LAYOUT), ttnn.bfloat2_b)
    assert back.dtype == ttnn.bfloat2_b
    assert torch.equal(ttnn.to_torch(back), x)


def test_bfp2_to_dtype_from_row_major_switches_to_tile():
    x = _ternary((64, 64))
    row_major = ttnn.from_torch(x)
    assert row_major.layout == ttnn.ROW_MAJOR_LAYOUT

    converted = ttnn.to_dtype(row_major, ttnn.bfloat2_b)

    assert converted.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(converted), x)


def test_bfp2_to_layout_tile_is_no_op():
    x = _ternary((64, 64))
    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)

    result = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)

    assert result.dtype == ttnn.bfloat2_b
    assert result.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(result), x)


def test_bfp2_to_layout_row_major_rejected_on_host(expect_error):
    # BFLOAT2_B is TILE-only, so ROW_MAJOR is an error rather than the silent no-op BFLOAT8_B/BFLOAT4_B get (#43763).
    tt_tensor = ttnn.from_torch(_ternary((32, 32)), dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "supports TILE layout only"):
        ttnn.to_layout(tt_tensor, ttnn.ROW_MAJOR_LAYOUT)


def test_bfp2_to_layout_row_major_rejected_on_device(device, expect_error):
    # Untilize keeps the BFLOAT2_B dtype and a ROW_MAJOR BFLOAT2_B tensor is invalid, so the request is rejected before
    # any kernel runs.
    tt_tensor = ttnn.from_torch(_ternary((32, 32)), dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Only TILE layout is supported"):
        ttnn.to_layout(tt_tensor, ttnn.ROW_MAJOR_LAYOUT)


def test_bfp2_serialization(tmp_path):
    x = _ternary((64, 96))
    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)

    path = tmp_path / "bfp2_tensor.tensorbin"
    ttnn.dump_tensor(path, tt_tensor)
    loaded = ttnn.load_tensor(path)

    assert loaded.dtype == ttnn.bfloat2_b
    assert loaded.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(loaded), x)


def test_bfp2_print_item_and_to_list():
    x = torch.full((32, 32), 0.5, dtype=torch.float32)
    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)

    text = str(tt_tensor)
    assert "0.5000" in text
    assert "bfloat4_b" not in text.lower()
    assert ttnn.from_torch(x[:1, :1], dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT).item() == 0.5
    assert tt_tensor.to_list()[3][7] == 0.5


def test_bfp2_full():
    tt_tensor = ttnn.full([64, 64], 0.5, dtype=ttnn.bfloat2_b)

    assert tt_tensor.dtype == ttnn.bfloat2_b
    assert tt_tensor.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(tt_tensor), torch.full((64, 64), 0.5))


@pytest.mark.parametrize("other", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_bfp2_device_typecast_float(device, other):
    # Float targets are converted by the unpacker and packer, so these casts are exact for ternary data.
    x = _ternary((64, 64))
    bfp2_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device)
    other_tensor = ttnn.from_torch(x, dtype=other, layout=ttnn.TILE_LAYOUT, device=device)

    assert torch.equal(ttnn.to_torch(ttnn.typecast(bfp2_tensor, other)).float(), x)
    assert torch.equal(ttnn.to_torch(ttnn.typecast(other_tensor, ttnn.bfloat2_b)), x)


@pytest.mark.parametrize("integer", [ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8, ttnn.int8])
def test_bfp2_device_typecast_integer(device, integer):
    # BFLOAT2_B takes the BFLOAT4_B arms of the typecast kernel, so {0, 1} converts exactly in both directions.
    x = _ternary((64, 64)).abs()
    bfp2_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device)
    integer_tensor = ttnn.from_torch(x.to(torch.int32), dtype=integer, layout=ttnn.TILE_LAYOUT, device=device)

    assert torch.equal(ttnn.to_torch(ttnn.typecast(bfp2_tensor, integer)).float(), x)
    assert torch.equal(ttnn.to_torch(ttnn.typecast(integer_tensor, ttnn.bfloat2_b)), x)


@pytest.mark.parametrize("integer", [ttnn.int32, ttnn.int8])
def test_bfp2_device_typecast_signed_integer(device, integer):
    # -1 is exact in BFLOAT2_B, so signed casts are exact on {-1, 0, 1}.
    x = _ternary((64, 64))
    bfp2_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device)
    integer_tensor = ttnn.from_torch(x.to(torch.int32), dtype=integer, layout=ttnn.TILE_LAYOUT, device=device)

    assert torch.equal(ttnn.to_torch(ttnn.typecast(bfp2_tensor, integer)).float(), x)
    assert torch.equal(ttnn.to_torch(ttnn.typecast(integer_tensor, ttnn.bfloat2_b)), x)


def test_bfp2_device_typecast_sharded_same_dtype(device):
    # Only a BFLOAT2_B -> BFLOAT2_B cast reaches the sharded typecast factory: no other format has the 320-byte tile.
    x = _ternary((128, 128))
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [32, 128],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    tt_tensor = ttnn.from_torch(
        x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )

    result = ttnn.typecast(tt_tensor, ttnn.bfloat2_b)

    assert result.dtype == ttnn.bfloat2_b
    assert torch.equal(ttnn.to_torch(result), x)


def test_bfp2_to_memory_config_dram_l1_round_trip(device):
    # Interleaved DRAM <-> L1 moves fall back to ttnn.copy.
    x = _ternary((64, 128))
    dram_tensor = ttnn.from_torch(
        x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    l1_tensor = ttnn.to_memory_config(dram_tensor, ttnn.L1_MEMORY_CONFIG)
    back = ttnn.to_memory_config(l1_tensor, ttnn.DRAM_MEMORY_CONFIG)

    assert l1_tensor.dtype == ttnn.bfloat2_b
    assert l1_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    assert torch.equal(ttnn.to_torch(l1_tensor), x)
    assert torch.equal(ttnn.to_torch(back), x)


def test_bfp2_add_integer_to_bfp2(device):
    # Binary ops convert their result to the requested dtype through the same typecast kernel.
    x = _ternary((64, 64)).abs().to(torch.int32)
    a = ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.add(a, a, dtype=ttnn.bfloat2_b)

    assert result.dtype == ttnn.bfloat2_b
    assert torch.equal(ttnn.to_torch(result), 2 * x.float())


def test_bfp2_binary_inputs_rejected(device, expect_error):
    # Binary ops can write BFLOAT2_B results but don't read BFLOAT2_B inputs yet.
    a = ttnn.from_torch(_ternary((32, 32)), dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Input tensor A dtype DataType::BFLOAT2_B is not supported"):
        ttnn.add(a, a)


def test_bfp2_device_round_trip(device):
    x = _ternary((64, 128))

    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device)

    assert tt_tensor.dtype == ttnn.bfloat2_b
    assert torch.equal(ttnn.to_torch(tt_tensor), x)


def test_bfp2_matmul_matches_bfp4(device):
    # Ternary weights are exact in both formats, so a matmul with BFLOAT2_B weights must match the same matmul with
    # BFLOAT4_B weights bit for bit.
    torch.manual_seed(0)
    x = torch.randn((32, 256), dtype=torch.bfloat16)
    w = _ternary((256, 128))
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    outputs = {}
    for weight_dtype in (ttnn.bfloat2_b, ttnn.bfloat4_b):
        tt_w = ttnn.from_torch(w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        outputs[weight_dtype] = ttnn.to_torch(ttnn.matmul(tt_x, tt_w))

    assert torch.equal(outputs[ttnn.bfloat2_b], outputs[ttnn.bfloat4_b])
    assert_with_pcc(x.float() @ w, outputs[ttnn.bfloat2_b].float(), 0.999)

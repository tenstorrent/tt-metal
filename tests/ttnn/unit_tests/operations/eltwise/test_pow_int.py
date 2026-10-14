# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
UINT16_MAX = 2**16 - 1

# UINT32 stays below 2^31 so values round-trip through torch.int32 unambiguously.
MAX_TEST_VALUE = {ttnn.int32: INT32_MAX, ttnn.uint32: INT32_MAX, ttnn.uint16: UINT16_MAX}
MIN_TEST_VALUE = {ttnn.int32: INT32_MIN, ttnn.uint32: 0, ttnn.uint16: 0}


def golden_int_pow(base: torch.Tensor, exponent: int, ttnn_dtype=ttnn.int32) -> torch.Tensor:
    """x^exponent modulo 2^bits via Python's exact modular pow, as the int32 bit pattern ttnn.to_torch returns."""
    bits = 16 if ttnn_dtype == ttnn.uint16 else 32
    results = torch.tensor([pow(x, exponent, 2**bits) for x in base.flatten().tolist()], dtype=torch.int64)
    if bits == 32:
        results = torch.where(results > INT32_MAX, results - 2**32, results)
    return results.to(torch.int32).reshape(base.shape)


def max_base_without_overflow(exponent: int, max_value: int = INT32_MAX) -> int:
    """Largest b with b^exponent <= max_value, for exponent >= 1."""
    limit = int(max_value ** (1.0 / exponent))
    while (limit + 1) ** exponent <= max_value:
        limit += 1
    while limit**exponent > max_value:
        limit -= 1
    return limit


def random_input_without_overflow(shape, exponent: int, ttnn_dtype) -> torch.Tensor:
    limit = max_base_without_overflow(max(exponent, 1), MAX_TEST_VALUE[ttnn_dtype])
    low = -limit if ttnn_dtype == ttnn.int32 else 0
    return torch.randint(low, min(limit + 1, INT32_MAX), shape, dtype=torch.int32)


def random_full_range_input(shape, ttnn_dtype) -> torch.Tensor:
    return torch.randint(MIN_TEST_VALUE[ttnn_dtype], MAX_TEST_VALUE[ttnn_dtype], shape, dtype=torch.int32)


def torch_pow(torch_input: torch.Tensor, exponent) -> torch.Tensor:
    """Reference for inputs whose result fits the dtype."""
    return torch.pow(torch_input.to(torch.int64), int(exponent)).to(torch.int32)


def skip_on_quasar(device):
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("integer pow is not supported on Quasar")


def run_pow(
    torch_input: torch.Tensor,
    exponent,
    ttnn_dtype,
    device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=None,
) -> torch.Tensor:
    skip_on_quasar(device)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    tt_output = ttnn.pow(tt_input, exponent, memory_config=output_memory_config)

    expected_config = tt_input.memory_config() if output_memory_config is None else output_memory_config
    assert tt_output.dtype == ttnn_dtype, f"expected output dtype {ttnn_dtype}, got {tt_output.dtype}"
    assert tt_output.memory_config() == expected_config, f"expected {expected_config}, got {tt_output.memory_config()}"
    return ttnn.to_torch(tt_output, dtype=torch.int32)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor):
    assert actual.shape == expected.shape, f"shape mismatch: expected {expected.shape}, got {actual.shape}"
    mismatches = (actual != expected).nonzero()
    assert mismatches.numel() == 0, (
        f"{mismatches.shape[0]} mismatches, first at {mismatches[0].tolist()}: "
        f"expected {expected[tuple(mismatches[0])].item()}, got {actual[tuple(mismatches[0])].item()}"
    )


ALL_DTYPES = [ttnn.int32, ttnn.uint32, ttnn.uint16]
SMALL_EXPONENTS = [0, 1, 2, 3]
# Host and dataflow paths don't depend on the exponent; 3 exercises both the squaring and the multiply step.
HOST_PATH_EXPONENT = 3
# Squaring only (32), mixed bits (3, 12345), and the longest all-ones chain (INT32_MAX).
WRAP_EXPONENTS = [3, 32, 12345, INT32_MAX]
DEFAULT_SHAPE = (1, 1, 64, 64)
SHAPES = [
    (32, 32),  # single tile
    (2, 3, 96, 64),  # multi-batch
    (1, 1, 736, 480),  # 345 tiles: more than any grid has cores and not divisible by them, so both core groups run
    (1, 1, 17, 45),  # not tile aligned
    (3, 1, 1, 33),  # not tile aligned, multiple tiles
]


def shape_id(shape) -> str:
    return "x".join(map(str, shape))


def dtype_id(ttnn_dtype) -> str:
    return str(ttnn_dtype).split(".")[-1].lower()


# UINT32 only wraps-tested at exponents 0 and 1, where results stay below 2^31 and round-trip through torch.int32;
# it shares the INT32 multiply, so larger exponents add nothing there.
FULL_RANGE_CASES = [pytest.param(d, e, id=f"{dtype_id(d)}-exp{e}") for e in (0, 1) for d in ALL_DTYPES] + [
    pytest.param(d, e, id=f"{dtype_id(d)}-exp{e}") for e in WRAP_EXPONENTS for d in (ttnn.int32, ttnn.uint16)
]

# Values straddling the points where x^2 and x^3 leave the dtype's range.
EDGE_VALUES = {
    ttnn.int32: [0, 1, -1, 2, -2, 7, -7, 1290, -1290, 1291, -1291, 46340, -46340, 46341, -46341, INT32_MAX, INT32_MIN],
    ttnn.uint16: [0, 1, 2, 3, 7, 40, 41, 255, 256, 257, 300, 32767, 32768, 65534, UINT16_MAX],
}

# Hand-computed results, independent of golden_int_pow.
KNOWN_VALUE_CASES = [
    pytest.param(ttnn.int32, 0, [0, -5, INT32_MIN], [1, 1, 1], id="int32-zero-exponent"),
    pytest.param(ttnn.int32, 3, [-2, 1290, -1290], [-8, 2146689000, -2146689000], id="int32-cube-sign"),
    pytest.param(ttnn.int32, 2, [46341, -46341, INT32_MIN], [-2147479015, -2147479015, 0], id="int32-square-wraps"),
    pytest.param(ttnn.uint32, 2, [46340, 3], [2147395600, 9], id="uint32-square"),
    pytest.param(ttnn.uint16, 0, [0, UINT16_MAX], [1, 1], id="uint16-zero-exponent"),
    # Wraps modulo 2^16 rather than saturating at 65535.
    pytest.param(ttnn.uint16, 2, [256, 257, 300, UINT16_MAX], [0, 513, 24464, 1], id="uint16-square-wraps"),
    pytest.param(ttnn.uint16, 3, [40, 41], [64000, 3385], id="uint16-cube-wraps"),
]


def height_sharded(shape):
    return ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=ttnn.ShardStrategy.HEIGHT
    )


def unevenly_height_sharded(shape):
    """Two 64-row shards, so the last shard is only partly filled unless rows % 64 == 0."""
    return ttnn.create_sharded_memory_config(
        (64, shape[-1]),
        core_grid=ttnn.CoreGrid(y=1, x=2),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


# (input memory config for a shape, output memory_config argument); None means the output inherits the input's.
MEMORY_CONFIG_CASES = [
    pytest.param(lambda shape: ttnn.L1_MEMORY_CONFIG, None, id="l1"),
    pytest.param(height_sharded, None, id="height_sharded"),
    pytest.param(lambda shape: ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, id="dram-to-l1"),
    pytest.param(lambda shape: ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, id="l1-to-dram"),
]

# 96 rows over 64-row shards: the sharded side has a partly filled last shard, the interleaved side does not.
UNEVEN_SHARD_SHAPE = (1, 1, 96, 64)
UNEVEN_SHARD_CASES = [
    pytest.param(unevenly_height_sharded(UNEVEN_SHARD_SHAPE), ttnn.DRAM_MEMORY_CONFIG, id="uneven_shards-to-dram"),
    pytest.param(ttnn.DRAM_MEMORY_CONFIG, unevenly_height_sharded(UNEVEN_SHARD_SHAPE), id="dram-to-uneven_shards"),
]


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(57306)


@pytest.mark.parametrize("ttnn_dtype", ALL_DTYPES, ids=dtype_id)
@pytest.mark.parametrize("shape", SHAPES, ids=shape_id)
def test_pow_int_shapes(shape, ttnn_dtype, device):
    torch_input = random_input_without_overflow(shape, HOST_PATH_EXPONENT, ttnn_dtype)

    actual = run_pow(torch_input, HOST_PATH_EXPONENT, ttnn_dtype, device)

    assert_equal(actual, torch_pow(torch_input, HOST_PATH_EXPONENT))


# Float-valued integer exponents must take the same integer path; the routing is dtype independent.
@pytest.mark.parametrize("exponent", [0.0, 3.0])
def test_pow_int_float_valued_exponent(exponent, device):
    torch_input = random_input_without_overflow(DEFAULT_SHAPE, int(exponent), ttnn.int32)

    actual = run_pow(torch_input, exponent, ttnn.int32, device)

    assert_equal(actual, torch_pow(torch_input, exponent))


@pytest.mark.parametrize("exponent", [0, 1])
@pytest.mark.parametrize("preallocate", [False, True], ids=["new-output", "preallocated"])
def test_pow_int_trivial_exponents_row_major(exponent, preallocate, device):
    torch_input = torch.randint(INT32_MIN, INT32_MAX, DEFAULT_SHAPE, dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = (
        ttnn.from_torch(torch.full_like(torch_input, 7), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if preallocate
        else None
    )

    actual = ttnn.pow(tt_input, exponent, output_tensor=tt_output)

    assert actual.layout == ttnn.ROW_MAJOR_LAYOUT
    assert_equal(ttnn.to_torch(actual), torch_pow(torch_input, exponent))


def test_pow_int_rejects_row_major_for_nontrivial_exponent(device, expect_error):
    skip_on_quasar(device)
    tt_input = ttnn.from_torch(
        torch.ones(DEFAULT_SHAPE, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    with expect_error(RuntimeError, "input must be in TILE layout"):
        ttnn.pow(tt_input, 2)


@pytest.mark.parametrize("ttnn_dtype, exponent", FULL_RANGE_CASES)
def test_pow_int_full_range_wraps(ttnn_dtype, exponent, device):
    torch_input = random_full_range_input(DEFAULT_SHAPE, ttnn_dtype)
    torch_input[..., 0, :8] = 0  # 0^0 == 1, as in torch

    actual = run_pow(torch_input, exponent, ttnn_dtype, device)

    assert_equal(actual, golden_int_pow(torch_input, exponent, ttnn_dtype))


@pytest.mark.parametrize("exponent", SMALL_EXPONENTS)
@pytest.mark.parametrize("ttnn_dtype", list(EDGE_VALUES), ids=dtype_id)
def test_pow_int_edge_values(exponent, ttnn_dtype, device):
    edge_values = EDGE_VALUES[ttnn_dtype]
    torch_input = torch.zeros((32, 32), dtype=torch.int32)
    torch_input.view(-1)[: len(edge_values)] = torch.tensor(edge_values, dtype=torch.int32)

    actual = run_pow(torch_input, exponent, ttnn_dtype, device)

    assert_equal(actual, golden_int_pow(torch_input, exponent, ttnn_dtype))


@pytest.mark.parametrize("ttnn_dtype, exponent, values, expected_values", KNOWN_VALUE_CASES)
def test_pow_int_known_values(ttnn_dtype, exponent, values, expected_values, device):
    torch_input = torch.zeros((32, 32), dtype=torch.int32)
    torch_input[0, : len(values)] = torch.tensor(values, dtype=torch.int32)
    expected = torch.full_like(torch_input, 1 if exponent == 0 else 0)
    expected[0, : len(values)] = torch.tensor(expected_values, dtype=torch.int32)

    actual = run_pow(torch_input, exponent, ttnn_dtype, device)

    assert_equal(actual, expected)


def test_pow_uint32_high_bit_values(device):
    skip_on_quasar(device)
    torch_input = torch.zeros((32, 32), dtype=torch.uint32)
    torch_input[0, :4] = torch.tensor([0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF], dtype=torch.uint32)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.pow(tt_input, 2), dtype=torch.uint32)

    expected = torch.zeros_like(torch_input)
    expected[0, :4] = torch.tensor([0, 1, 4, 1], dtype=torch.uint32)
    assert torch.equal(actual, expected), f"expected {expected[0, :4].tolist()}, got {actual[0, :4].tolist()}"


@pytest.mark.parametrize("ttnn_dtype", ALL_DTYPES, ids=dtype_id)
@pytest.mark.parametrize("make_input_config, output_config", MEMORY_CONFIG_CASES)
def test_pow_int_memory_configs(make_input_config, output_config, ttnn_dtype, device):
    shape = (1, 1, 256, 64)
    torch_input = random_input_without_overflow(shape, HOST_PATH_EXPONENT, ttnn_dtype)

    actual = run_pow(
        torch_input,
        HOST_PATH_EXPONENT,
        ttnn_dtype,
        device,
        memory_config=make_input_config(shape),
        output_memory_config=output_config,
    )

    assert_equal(actual, torch_pow(torch_input, HOST_PATH_EXPONENT))


@pytest.mark.parametrize("input_config, output_config", UNEVEN_SHARD_CASES)
def test_pow_int_uneven_shards_to_interleaved(input_config, output_config, device):
    torch_input = random_input_without_overflow(UNEVEN_SHARD_SHAPE, HOST_PATH_EXPONENT, ttnn.int32)

    actual = run_pow(
        torch_input,
        HOST_PATH_EXPONENT,
        ttnn.int32,
        device,
        memory_config=input_config,
        output_memory_config=output_config,
    )

    assert_equal(actual, torch_pow(torch_input, HOST_PATH_EXPONENT))


@pytest.mark.parametrize("ttnn_dtype", ALL_DTYPES, ids=dtype_id)
def test_pow_int_preallocated_output(ttnn_dtype, device):
    skip_on_quasar(device)
    torch_input = random_input_without_overflow(DEFAULT_SHAPE, HOST_PATH_EXPONENT, ttnn_dtype)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.from_torch(
        torch.full_like(torch_input, 7), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn.pow(tt_input, HOST_PATH_EXPONENT, output_tensor=tt_output)

    assert_equal(ttnn.to_torch(tt_output, dtype=torch.int32), torch_pow(torch_input, HOST_PATH_EXPONENT))


@pytest.mark.parametrize("tile_shape", [(16, 32), (32, 16), (16, 16)], ids=shape_id)
def test_pow_int_rejects_custom_input_tile(tile_shape, device, expect_error):
    skip_on_quasar(device)
    tile = ttnn.Tile(tile_shape)
    tt_input = ttnn.from_torch(
        torch.ones(DEFAULT_SHAPE, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        device=device,
    )

    with expect_error(RuntimeError, "default 32x32 tile"):
        ttnn.pow(tt_input, HOST_PATH_EXPONENT)


def test_pow_int_rejects_preallocated_output_with_different_tile(device, expect_error):
    skip_on_quasar(device)
    torch_input = torch.ones(DEFAULT_SHAPE, dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.from_torch(
        torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((16, 32)), device=device
    )

    with expect_error(RuntimeError, "output tile"):
        ttnn.pow(tt_input, 2, output_tensor=tt_output)


@pytest.mark.parametrize("ttnn_dtype", ALL_DTYPES, ids=dtype_id)
def test_pow_int_program_cache_keyed_on_exponent(ttnn_dtype, device):
    torch_input = random_input_without_overflow(DEFAULT_SHAPE, 3, ttnn_dtype)

    # Same shape, interleaved exponents: a cache entry keyed without the exponent would return wrong results.
    for exponent in [2, 3, 0, 1, 2, 3]:
        actual = run_pow(torch_input, exponent, ttnn_dtype, device)
        assert_equal(actual, torch_pow(torch_input, exponent))


def test_pow_int_rejects_quasar(device, expect_error):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("integer pow is supported on this architecture")
    tt_input = ttnn.from_torch(
        torch.ones((32, 32), dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "not supported on Quasar"):
        ttnn.pow(tt_input, 2)


@pytest.mark.parametrize("ttnn_dtype", ALL_DTYPES, ids=dtype_id)
def test_pow_int_negative_exponent_raises(ttnn_dtype, device, expect_error):
    torch_input = torch.randint(1, 10, (32, 32), dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "negative integer powers"):
        ttnn.pow(tt_input, -1)


@pytest.mark.parametrize("ttnn_dtype", ALL_DTYPES, ids=dtype_id)
def test_pow_int_non_integral_exponent_raises(ttnn_dtype, device, expect_error):
    tt_input = ttnn.from_torch(
        torch.ones((32, 32), dtype=torch.int32), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "requires an integral exponent"):
        ttnn.pow(tt_input, 2.5)


@pytest.mark.parametrize("ttnn_dtype", ALL_DTYPES, ids=dtype_id)
def test_pow_int_tensor_exponent_raises(ttnn_dtype, device, expect_error):
    torch_input = torch.ones((32, 32), dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_exponent = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "tensor exponents are not supported"):
        ttnn.pow(tt_input, tt_exponent)


@pytest.mark.parametrize("max_value", [INT32_MAX, UINT16_MAX])
@pytest.mark.parametrize("exponent", [1, 2, 3, 5, 31])
def test_max_base_without_overflow_helper(exponent, max_value):
    limit = max_base_without_overflow(exponent, max_value)

    assert limit**exponent <= max_value < (limit + 1) ** exponent, f"bad limit {limit} for exponent {exponent}"

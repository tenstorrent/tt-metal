# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def test_benchmark_from_torch_zero_copy(benchmark):
    # Zero copy from_torch: row_major + physical_shape == logical_shape2d + bfloat16->bfloat16
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)

    def from_torch():
        ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    benchmark.pedantic(from_torch, iterations=100, rounds=10, warmup_rounds=1)


def test_benchmark_from_torch_one_copy(benchmark):
    # One copy from_torch tile layout + physical_shape == logical_shape2d + bfloat16->bfloat16
    # - Copy on tilizing
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)

    def from_torch():
        ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    benchmark.pedantic(from_torch, iterations=10, rounds=5, warmup_rounds=1)


def test_benchmark_from_torch_two_copy(benchmark):
    # Two copied from_torch tile layout + physical_shape != logical_shape2d + bfloat16->bfloat16
    # - First copy on padding to tile_size
    # - Second copy on tilizing
    torch_tensor = torch.rand((8096, 8100), dtype=torch.bfloat16)

    def from_torch():
        ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    benchmark.pedantic(from_torch, iterations=10, rounds=5, warmup_rounds=1)


TORCH_BENCH_TYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.uint8,
]

TTNN_BENCH_TYPES = [
    ttnn.float32,
    ttnn.bfloat16,
    ttnn.bfloat8_b,
    ttnn.bfloat4_b,
    ttnn.uint8,
    ttnn.int32,
]


@pytest.mark.parametrize("use_device", [True, False])
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("torch_dtype", TORCH_BENCH_TYPES)
@pytest.mark.parametrize("ttnn_dtype", TTNN_BENCH_TYPES)
@pytest.mark.parametrize("size", [8192])
def test_benchmark_from_torch(benchmark, device, use_device, ttnn_dtype, torch_dtype, ttnn_layout, size, request):
    if ttnn_layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
        pytest.skip("ROW_MAJOR_LAYOUT not supported with bfloat8_b/bfloat4_b")
    if torch_dtype in [torch.int32, torch.uint8, torch.int64]:
        torch_input_tensor = torch.randint(0, 100, (size, size), dtype=torch_dtype)
    else:
        torch_input_tensor = torch.rand((size, size), dtype=torch_dtype)

    def from_torch():
        ttnn_tensor = ttnn.from_torch(
            torch_input_tensor,
            device=device if use_device else None,
            dtype=ttnn_dtype,
            layout=ttnn_layout,
        )

        if not use_device:
            ttnn.to_device(ttnn_tensor, device=device)

        ttnn.synchronize_device(device)

    benchmark.pedantic(from_torch, iterations=10, rounds=2, warmup_rounds=1)


@pytest.mark.parametrize("use_device", [True, False])
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("torch_dtype", TORCH_BENCH_TYPES)
@pytest.mark.parametrize("ttnn_dtype", TTNN_BENCH_TYPES)
@pytest.mark.parametrize("size", [8192])
def test_benchmark_to_torch(benchmark, device, use_device, ttnn_dtype, torch_dtype, size, ttnn_layout):
    if ttnn_layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
        pytest.skip("ROW_MAJOR_LAYOUT not supported with bfloat8_b/bfloat4_b")

    match ttnn_dtype:
        case ttnn.int32 | ttnn.uint8:
            tmp_torch = torch.randint(0, 100, (size, size), dtype=torch.int32)

        case _:
            tmp_torch = torch.rand(size, size, dtype=torch.float32)

    ttnn_input_tensor = ttnn.from_torch(tmp_torch, device=device, dtype=ttnn_dtype, layout=ttnn_layout)

    def to_torch():
        ttnn.to_torch(ttnn_input_tensor, device=device if use_device else None, dtype=torch_dtype)
        ttnn.synchronize_device(device)

    benchmark.pedantic(to_torch, iterations=10, rounds=2, warmup_rounds=1)


def generate_bfloat4_b_exact_tensor(shape, seed=0):
    """
    Generate a float32 torch tensor whose values survive a round-trip through
    bfloat4_b without any precision loss.

    bfloat4_b is a block floating-point format where every 16-element block
    (one face row in tile layout) shares a single 8-bit exponent, and each
    element stores 1 sign bit + 3 mantissa bits (with an implicit leading 1).

    To guarantee exact round-trip, this function ensures:
     - All 16 elements in each block share the same power-of-two exponent,
        so the shared exponent equals every element's exponent (no alignment
        shift, no mantissa bit loss).
     - Each element's mantissa is one of the 8 exactly representable values:
        {1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875}
        which correspond to the 3-bit patterns 000..111 with hidden bit.

    The last two dimensions of *shape* must be multiples of 32 (tile layout).
    """
    torch.manual_seed(seed)

    EXACT_MANTISSAS = torch.tensor([1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875], dtype=torch.float32)
    FACE_ROW_SIZE = 16

    assert len(shape) >= 2, "Shape must have at least 2 dimensions"
    H, W = shape[-2], shape[-1]
    assert H % 32 == 0 and W % 32 == 0, f"Last two dims must be multiples of 32 for tile layout, got ({H}, {W})"

    total_elements = 1
    for d in shape:
        total_elements *= d
    total_rows = total_elements // W
    num_blocks_per_row = W // FACE_ROW_SIZE
    total_blocks = total_rows * num_blocks_per_row

    exponents = torch.randint(-4, 5, (total_blocks, 1), dtype=torch.float32)
    mantissa_indices = torch.randint(0, 8, (total_blocks, FACE_ROW_SIZE))
    mantissas = EXACT_MANTISSAS[mantissa_indices]
    signs = torch.where(
        torch.randint(0, 2, (total_blocks, FACE_ROW_SIZE)).bool(),
        torch.ones(1, dtype=torch.float32),
        -torch.ones(1, dtype=torch.float32),
    )

    values = signs * torch.pow(2.0, exponents) * mantissas
    return values.reshape(shape)


def quantize_to_bf4(tensor, exp_bits=2, mant_bits=1):
    """
    Simulates a 4-bit float roundtrip.
    Default: 1 sign bit, 2 exponent bits, 1 mantissa bit (E2M1).
    """
    # 1. Capture the sign
    sign = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)

    # 2. Handle zeros to avoid log errors
    abs_tensor[abs_tensor == 0] = 1e-8

    # 3. Log-scale to find the exponent
    # bfloat4 has a very narrow range (usually 2^exponent)
    exponent = torch.floor(torch.log2(abs_tensor))

    # Clip exponent to fit in 2 bits (e.g., range -1 to 2)
    max_exp = 2 ** (exp_bits - 1)
    exponent = torch.clamp(exponent, -max_exp, max_exp)

    # 4. Quantize the mantissa
    # With 1 mantissa bit, we only have two levels: 1.0 and 1.5
    mantissa = abs_tensor / (2**exponent)
    mantissa = torch.round(mantissa * (2**mant_bits)) / (2**mant_bits)
    mantissa = torch.clamp(mantissa, 1.0, 2.0 - (1 / (2**mant_bits)))

    # 5. Reconstruct the "crushed" value
    bf4_simulated = sign * mantissa * (2**exponent)

    return bf4_simulated.to(torch.float32)


def test_benchmark_from_torch_deep_seek(benchmark, device):
    # Two copied from_torch tile layout + physical_shape != logical_shape2d + bfloat16->bfloat16
    # - First copy on padding to tile_size
    # - Second copy on tilizing
    torch.manual_seed(0)
    shape = (1, 7168, 2304)
    shard_shape = (7168, 192)
    torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)  # torch.rand(shape, dtype=torch_dtype)
    torch_input_tensor = quantize_to_bf4(torch_input_tensor)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    def from_torch():
        ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat4_b,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # memory_config
        ttnn.synchronize_device(device)

    benchmark.pedantic(from_torch, iterations=10, rounds=5, warmup_rounds=1)


def test_benchmark_from_torch_deep_seek_mc(benchmark, device):
    # Two copied from_torch tile layout + physical_shape != logical_shape2d + bfloat16->bfloat16
    # - First copy on padding to tile_size
    # - Second copy on tilizing
    torch.manual_seed(0)
    shape = (1, 7168, 2304)
    shard_shape = (7168, 192)
    torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)  # torch.rand(shape, dtype=torch_dtype)
    torch_input_tensor = quantize_to_bf4(torch_input_tensor)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    def from_torch():
        tensor = ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.float32,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
        tensor = ttnn.typecast(tensor, ttnn.bfloat4_b)
        ttnn.synchronize_device(device)

    benchmark.pedantic(from_torch, iterations=10, rounds=5, warmup_rounds=1)

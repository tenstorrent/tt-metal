# SPDX-FileCopyrightText: © 2023-24 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.python_api_testing.sweep_tests.ttnn_pytorch_ops import eltwise_typecast
from tests.ttnn.python_api_testing.typecast_test_helpers import (
    assert_integer_typecast_equal,
    make_typecast_test_input,
    narrow_to_8bit,
    typecast_test_input_bounds,
    uses_exact_integer_typecast_check,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal

mem_configs = [
    ttnn.DRAM_MEMORY_CONFIG,
    ttnn.L1_MEMORY_CONFIG,
]

TILE_HEIGHT = 32
TILE_WIDTH = 32

cpu_layout = ttnn.Layout.ROW_MAJOR
npu_layout = ttnn.Layout.TILE


def _make_fp32_to_int32_input(shape):
    in_low, in_high = typecast_test_input_bounds(ttnn.float32, ttnn.int32)
    return make_typecast_test_input(shape, torch.float32, in_low, in_high)


@pytest.mark.parametrize("num_tiles", [1, 128])
@pytest.mark.parametrize("memory_config", mem_configs)
def test_typecast_uint32_to_fp32_rounding(device, num_tiles, memory_config):
    generator = torch.Generator().manual_seed(52787)
    values = torch.randint(0, 2**32, (num_tiles * 1024,), dtype=torch.int64, generator=generator)
    # Include every uint32 binade requiring rounding, both midpoint parities,
    # and the 23-bit split boundaries used by the conversion kernel.
    boundaries = [0, 1, 2**32 - 1, 2164260993]
    for exponent in range(24, 32):
        ulp = 1 << (exponent - 23)
        for offset in (0, ulp, 1 << 23):
            midpoint = (1 << exponent) + offset + ulp // 2
            boundaries.extend((midpoint - 1, midpoint, midpoint + 1))
    for high in (1, 2, 127, 128, 255, 256, 510, 511):
        boundaries.extend(((high << 23) - 1, high << 23, (high << 23) + 1))
    values[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int64)
    expected = values.to(torch.float32).reshape(1, 1, num_tiles * 32, 32)
    # Preserve all 32 input bits, including values above INT32_MAX.
    input_tensor = ttnn.from_torch(
        values.to(torch.uint32).reshape(expected.shape),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    output = ttnn.to_torch(ttnn.typecast(input_tensor, ttnn.float32, memory_config=memory_config))
    # Compare bits: PCC and tolerances would hide one-ULP conversion errors.
    assert torch.equal(output.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize(
    "tt_input_dtype, tt_output_dtype, expected",
    [
        (ttnn.int32, ttnn.uint16, (-1000, 80000)),
        (ttnn.uint32, ttnn.uint16, (0, 80000)),
        (ttnn.int32, ttnn.uint32, (-1000, 80000)),
        (ttnn.int32, ttnn.uint8, (-512, 512)),
        (ttnn.uint16, ttnn.uint8, (0, 512)),
        (ttnn.uint32, ttnn.uint8, (0, 512)),
        (ttnn.int32, ttnn.int8, (-256, 256)),
        (ttnn.uint16, ttnn.int8, (0, 256)),
        (ttnn.bfloat16, ttnn.int8, (-256, 256)),
        (ttnn.uint16, ttnn.uint32, (0, 65535)),
        (ttnn.uint16, ttnn.int32, (0, 65535)),
        (ttnn.bfloat16, ttnn.uint8, (-512, 512)),
        (ttnn.bfloat16, ttnn.uint16, (-1000, 80000)),
        (ttnn.bfloat16, ttnn.bfloat16, (0, 100)),
    ],
)
def test_typecast_test_input_bounds(tt_input_dtype, tt_output_dtype, expected):
    assert typecast_test_input_bounds(tt_input_dtype, tt_output_dtype) == expected


def test_typecast_test_input_bounds_unsigned_high_never_exceeds_input_max():
    for tt_output_dtype in (ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8, ttnn.int8):
        low, high = typecast_test_input_bounds(ttnn.uint16, tt_output_dtype)
        assert low == 0
        assert high <= 65535


@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    (
        (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat16),
        (torch.float32, ttnn.float32, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b),
        (torch.int, ttnn.uint16, ttnn.uint32),
        (torch.int, ttnn.uint16, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.uint16),
        (torch.int, ttnn.uint32, ttnn.uint16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.uint8),
        (torch.uint8, ttnn.uint8, ttnn.bfloat16),
        (torch.float32, ttnn.float32, ttnn.uint8),
        (torch.uint8, ttnn.uint8, ttnn.float32),
        (torch.int, ttnn.int32, ttnn.uint8),
        (torch.uint8, ttnn.uint8, ttnn.int32),
        (torch.int, ttnn.uint16, ttnn.uint8),
        (torch.uint8, ttnn.uint8, ttnn.uint16),
        (torch.int, ttnn.uint32, ttnn.uint8),
        (torch.uint8, ttnn.uint8, ttnn.uint32),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.uint8),
        (torch.uint8, ttnn.uint8, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.uint8),
        (torch.uint8, ttnn.uint8, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat16, ttnn.int8),
        (torch.int8, ttnn.int8, ttnn.bfloat16),
        (torch.float32, ttnn.float32, ttnn.int8),
        (torch.int8, ttnn.int8, ttnn.float32),
        (torch.int, ttnn.int32, ttnn.int8),
        (torch.int8, ttnn.int8, ttnn.int32),
        (torch.int, ttnn.uint16, ttnn.int8),
        (torch.int8, ttnn.int8, ttnn.uint16),
        (torch.int, ttnn.uint32, ttnn.int8),
        (torch.int8, ttnn.int8, ttnn.uint32),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.int8),
        (torch.int8, ttnn.int8, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.int8),
        (torch.int8, ttnn.int8, ttnn.bfloat4_b),
        (torch.uint8, ttnn.uint8, ttnn.int8),
        (torch.int8, ttnn.int8, ttnn.uint8),
    ),
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],  # Single core
        [[1, 1, 320, 320]],  # multi core
        [[1, 3, 320, 320]],  # multi core
    ],
)
@pytest.mark.parametrize(
    "input_mem_config",
    mem_configs,
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
class TestTypecast:
    def test_run_eltwise_typecast_op(
        self,
        tt_output_dtype,
        pt_input_dtype,
        tt_input_dtype,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        if tt_input_dtype == tt_output_dtype:
            pytest.skip("Same I/O data types. Skip.")
        in_low, in_high = typecast_test_input_bounds(tt_input_dtype, tt_output_dtype)
        shape = input_shapes[0]
        torch_input = make_typecast_test_input(shape, pt_input_dtype, in_low, in_high)

        tt_input = ttnn.from_torch(
            torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem_config
        )
        tt_output = ttnn.typecast(tt_input, tt_input_dtype, tt_output_dtype, memory_config=dst_mem_config)
        torch_output = ttnn.to_torch(tt_output)

        torch_golden = eltwise_typecast(torch_input, tt_input_dtype=tt_input_dtype, tt_output_dtype=tt_output_dtype)

        if tt_output_dtype in (ttnn.float32, ttnn.bfloat16):
            assert_equal(torch_golden, torch_output)
        elif uses_exact_integer_typecast_check(tt_input_dtype, tt_output_dtype):
            assert_integer_typecast_equal(torch_golden, torch_output)
        else:
            pcc = 0.98 if tt_input_dtype == ttnn.bfloat4_b or tt_output_dtype == ttnn.bfloat4_b else 0.99
            assert_with_pcc(torch_golden, torch_output, pcc=pcc)


@pytest.mark.parametrize("tt_output_dtype", [ttnn.uint8, ttnn.int8, ttnn.uint16, ttnn.uint32, ttnn.int32])
@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
def test_typecast_zero_fp_to_int(device, pt_input_dtype, tt_input_dtype, tt_output_dtype):
    """Regression: fp32/bf16 zero must typecast to integer 0 for all int output types.

    SFPEXEXP has a hardware quirk for zero/subnormal inputs: biased_exp=0
    returns an incorrect unbiased value, causing naive kernels to produce
    incorrect results (e.g. 2 for fp32->uint8) for 0.0 inputs.  The random
    inputs in test_run_eltwise_typecast_op are vanishingly unlikely to include
    exact zero, so this dedicated test ensures the corner case is always covered
    across all supported integer output types.
    """
    shape = [1, 1, 32, 32]
    torch_input = torch.zeros(shape, dtype=pt_input_dtype)

    tt_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.typecast(tt_input, tt_input_dtype, tt_output_dtype)
    result = ttnn.to_torch(tt_output)

    assert torch.all(result == 0), f"0.0 -> {tt_output_dtype} produced non-zero values: {result.unique().tolist()}"


@pytest.mark.parametrize(
    "tt_output_dtype, torch_output_dtype", [(ttnn.int32, torch.int32), (ttnn.uint32, torch.uint32)]
)
@pytest.mark.parametrize("tt_input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_typecast_fp_to_int_saturates_out_of_range(device, tt_input_dtype, tt_output_dtype, torch_output_dtype):
    """Regression for #55933: out-of-range floats saturate to the output limit of their own sign.

    calculate_typecast_fp32_to_int32 loads INT32_MIN as the saturation constant for every
    overflowing lane and only negates negative inputs, so positive overflow used to come
    out as INT32_MIN. The random inputs in test_run_eltwise_typecast_op stay below 80000, so
    this test covers both overflow directions and the values either side of each boundary,
    including the upper half of the uint32 range (#61736). NaN is left out: it saturates by
    its sign bit like +-inf, but torch's NaN conversion is platform-dependent.
    """
    values = [
        # int32 positive overflow starting at the first float32 at 2^31; in range for uint32
        2.0**31,
        1.25 * 2.0**31,
        3e9,
        4294901760.0,
        # largest float32 below 2^32, then uint32 positive overflow
        4294967040.0,
        2.0**32,
        1e38,
        float("inf"),
        # negative overflow, starting at the first float32 below -2^31
        -2147483904.0,
        -3e9,
        -1e38,
        float("-inf"),
        # in range, including the largest float32 below 2^31 and -2^31 itself
        2147483520.0,
        -(2.0**31),
        2.0**30,
        -(2.0**30),
        65536.0,
        -65536.0,
        1.0,
        -1.0,
        0.0,
    ]
    torch_input = torch.tensor([values], dtype=torch.float32)
    tt_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Expected values come from what the device holds, because bfloat16 packing rounds some inputs.
    held = ttnn.to_torch(tt_input).to(torch.float64)
    output_info = torch.iinfo(torch_output_dtype)
    expected = held.trunc().clamp(output_info.min, output_info.max).to(torch.int64)

    tt_output = ttnn.typecast(tt_input, tt_input_dtype, tt_output_dtype)
    assert tt_output.dtype == tt_output_dtype
    # Compared in int64 so a mis-saturated lane reports its real distance instead of a 32-bit wrap.
    assert_equal(expected, ttnn.to_torch(tt_output).to(torch.int64))


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
def test_typecast_bf16_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # bf16 --> bfp8_b by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def print_mismatches(cpu, npu, num_max_print):
    different_indices = (cpu != npu).nonzero(as_tuple=True)
    count = 0
    for idx in zip(*different_indices):
        count = count + 1
        print(f"idx={idx} cpu={cpu[idx]} npu={npu[idx]}")
        if count > num_max_print:
            break


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
@pytest.mark.parametrize("seed", [0, 2, 4, 6, 8])
@pytest.mark.parametrize("scale", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("bias", [0, 1, 2, 4, 8, 16, 32, 64, 128])
def test_typecast_bf16_to_bfp8_b_various_input(seed, scale, bias, device):
    torch.manual_seed(seed)
    shape = [1024, 1024]

    bias = bias
    low = bias - scale
    high = bias + scale
    torch_bf16 = random_tensor = torch.empty(shape).uniform_(low, high).to(torch.bfloat16)

    random_signs = torch.randint(0, 2, shape) * 2 - 1
    torch_bf16 = torch_bf16 * random_signs

    # bf16 --> bfp8_b by cpu.
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    if not passed:
        print_mismatches(cpu_version, npu_version, 16)
    assert passed


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("scale", [4])
@pytest.mark.parametrize("bias", [2])
# NaN becomes -Inf when converted to bfloat8_b format, skip testing
@pytest.mark.parametrize("insert_inf, insert_nan", [[True, False]])  # , [False, True], [True, True]])
def test_typecast_bf16_to_bfp8_b_with_inf_nan(seed, scale, bias, insert_inf, insert_nan, device):
    torch.manual_seed(seed)
    shape = [1024, 1024]

    bias = bias
    low = bias - scale
    high = bias + scale

    torch_bf16 = random_tensor = torch.empty(shape).uniform_(low, high).to(torch.bfloat16)
    if insert_inf:
        num_inf = torch_bf16.numel() // 8  # 16 elements are pcked into
        inf_indices = torch.randint(0, torch_bf16.numel(), (num_inf,))
        torch_bf16.view(-1)[inf_indices] = float("inf")
    if insert_nan:
        num_nan = torch_bf16.numel() // 8
        nan_indices = torch.randint(0, torch_bf16.numel(), (num_nan,))
        torch_bf16.view(-1)[nan_indices] = float("nan")
    random_signs = torch.randint(0, 2, shape) * 2 - 1
    torch_bf16 = torch_bf16 * random_signs

    # bf16 --> bfp8_b by cpu.
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    if not passed:
        print_mismatches(cpu_version, npu_version, 16)
    assert passed


def test_typecast_bfp8_b_to_bf16(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> bf16 by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> bf16 by npu.
    bf16_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.bfloat16)
    npu_version = bf16_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def test_typecast_fp32_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # fp32 --> bfp8_b by cpu.
    torch_fp32 = torch.randn(shape, dtype=torch.float32)
    bfp8_b_by_cpu = ttnn.Tensor(torch_fp32, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # fp32 --> bfp8_b by npu
    tt_fp32 = ttnn.Tensor(torch_fp32, ttnn.float32).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_fp32, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def test_typecast_bfp8_b_to_fp32(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> fp32 by cpu.
    torch_fp32 = torch.randn(shape, dtype=torch.float32)
    bfp8_b = ttnn.Tensor(torch_fp32, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> fp32 by npu.
    fp32_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.float32)
    npu_version = fp32_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


@pytest.mark.parametrize(
    "shard_layout, core_grid, shard_shape",
    [
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [32, 128],
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [128, 32],
        ),
        (
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [64, 64],
        ),
    ],
)
@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    [
        (torch.float32, ttnn.float32, ttnn.int32),
        (torch.int8, ttnn.int8, ttnn.uint8),
        (torch.int8, ttnn.int8, ttnn.int32),
        (torch.int8, ttnn.int8, ttnn.bfloat16),
        (torch.float32, ttnn.float32, ttnn.int8),
    ],
)
def test_typecast_legacy_sharded(
    device, shard_layout, core_grid, shard_shape, pt_input_dtype, tt_input_dtype, tt_output_dtype
):
    torch.manual_seed(0)
    shape = [1, 1, 128, 128]

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)

    in_low, in_high = typecast_test_input_bounds(tt_input_dtype, tt_output_dtype)
    torch_input = make_typecast_test_input(shape, pt_input_dtype, in_low, in_high)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=tt_output_dtype)
    assert output_tensor.dtype == tt_output_dtype

    result = ttnn.to_torch(output_tensor)
    expected = eltwise_typecast(torch_input, tt_input_dtype=tt_input_dtype, tt_output_dtype=tt_output_dtype)
    if uses_exact_integer_typecast_check(tt_input_dtype, tt_output_dtype):
        assert_integer_typecast_equal(expected, result)
    else:
        assert_equal(expected, result)


@pytest.mark.parametrize(
    "tensor_shape, nd_shard_shape, shard_grid",
    [
        (
            [1, 1, 128, 128],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        ([4, 128, 128], [2, 64, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})),
        (
            [2, 2, 128, 128],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [3, 128, 128],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),  # uneven: 3 % 2 != 0
        (
            [5, 4, 160, 160],
            [2, 3, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ),  # uneven sharding in all dims
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    [
        (torch.float32, ttnn.float32, ttnn.int32),
        (torch.int8, ttnn.int8, ttnn.int32),
        (torch.float32, ttnn.float32, ttnn.int8),
    ],
)
def test_typecast_nd_sharded_int(
    device,
    tensor_shape,
    nd_shard_shape,
    shard_grid,
    shard_orientation,
    layout,
    pt_input_dtype,
    tt_input_dtype,
    tt_output_dtype,
):
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape=nd_shard_shape, grid=shard_grid, orientation=shard_orientation)
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    in_low, in_high = typecast_test_input_bounds(tt_input_dtype, tt_output_dtype)
    torch_input = make_typecast_test_input(tensor_shape, pt_input_dtype, in_low, in_high)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=tt_input_dtype, layout=layout, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=tt_output_dtype)
    assert output_tensor.dtype == tt_output_dtype

    result = ttnn.to_torch(output_tensor)
    expected = eltwise_typecast(torch_input, tt_input_dtype=tt_input_dtype, tt_output_dtype=tt_output_dtype)
    assert_integer_typecast_equal(expected, result)


@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32),
    ],
)
@pytest.mark.parametrize(
    "tensor_shape, nd_shard_shape, shard_grid",
    [
        (
            [1, 1, 128, 128],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [4, 128, 128],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [3, 128, 128],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),  # uneven in dim 0
        (
            [5, 3, 96, 160],
            [3, 2, 64, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),  # uneven in all dims: 5%3, 3%2, 96%64, 160%96
    ],
)
def test_typecast_nd_sharded_float(
    device, pt_input_dtype, tt_input_dtype, tt_output_dtype, tensor_shape, nd_shard_shape, shard_grid
):
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=nd_shard_shape, grid=shard_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_input = torch.randn(tensor_shape, dtype=pt_input_dtype)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    # Build CPU-side reference: run the same input -> tt_input_dtype -> tt_output_dtype
    cpu_tensor = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
    cpu_reference = ttnn.to_torch(cpu_tensor)

    output_tensor = ttnn.typecast(input_tensor, dtype=tt_output_dtype)
    assert output_tensor.dtype == tt_output_dtype

    npu_result = ttnn.to_torch(output_tensor)

    # Compare in a common dtype so shapes/dtypes align
    common_dtype = torch.float32
    assert torch.equal(cpu_reference.to(common_dtype), npu_result.to(common_dtype))


# Tests that legacy-sharded inputs violating TypecastShardedProgramFactory preconditions
# fall back to TypecastProgramFactory correctly.
@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    [
        # Violates legacy sharded program factory condition: tile_size(input) != tile_size(output)
        # bfloat16 tile = 2048 bytes, float32 tile = 4096 bytes
        (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
        # bfloat16 tile = 2048 bytes, bfp8_b tile = 1088 bytes
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32),
    ],
)
@pytest.mark.parametrize(
    "shard_layout, core_grid, shard_shape",
    [
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [32, 128],
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [128, 32],
        ),
        (
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [64, 64],
        ),
    ],
)
def test_typecast_legacy_sharded_tile_size_mismatch(
    device, pt_input_dtype, tt_input_dtype, tt_output_dtype, shard_layout, core_grid, shard_shape
):
    """Violates is_valid_for_sharded_optimized_typecast condition 2: tile_size(input) != tile_size(output)."""
    torch.manual_seed(0)
    shape = [1, 1, 128, 128]

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)

    torch_input = torch.randn(shape, dtype=pt_input_dtype)

    cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
    cpu_output = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
    cpu_reference = ttnn.to_torch(cpu_output)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=tt_output_dtype)
    assert output_tensor.dtype == tt_output_dtype

    npu_result = ttnn.to_torch(output_tensor)
    common_dtype = torch.float32
    assert torch.equal(cpu_reference.to(common_dtype), npu_result.to(common_dtype))


@pytest.mark.parametrize(
    "shard_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
def test_typecast_legacy_sharded_dram_buffer(device, shard_layout):
    """Violates is_valid_for_sharded_optimized_typecast condition 3: input buffer is in DRAM, not L1."""
    torch.manual_seed(0)
    shape = [1, 1, 128, 128]

    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [32, 128]
    else:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [128, 32]

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.DRAM, shard_spec)

    torch_input = _make_fp32_to_int32_input(shape)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=ttnn.int32)
    assert output_tensor.dtype == ttnn.int32

    npu_result = ttnn.to_torch(output_tensor)
    expected = eltwise_typecast(torch_input, tt_input_dtype=ttnn.float32, tt_output_dtype=ttnn.int32)
    assert_integer_typecast_equal(expected, npu_result)


def test_typecast_legacy_sharded_shard_size_not_tile_aligned(device):
    """Violates is_valid_for_sharded_optimized_typecast condition 5: shard_size_in_bytes % tile_size != 0.

    Uses ROW_MAJOR WIDTH_SHARDED with shard = [64, 8] and float32 (4 bytes).
    8 * 4 = 32 bytes, which IS a multiple of L1 alignment (16 bytes) — condition 4 satisfied.
    64 * 8 * 4 = 2048 bytes, tile_size(Float32) = 4096 — 2048 % 4096 != 0 — condition 5 violated.
    Uses float32 -> int32 (same tile size) so only condition 5 is violated.
    """
    torch.manual_seed(0)
    shape = [1, 1, 64, 32]

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_shape = [64, 8]  # 4 cores, width 32 / 4 = 8

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_input = _make_fp32_to_int32_input(shape)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=ttnn.int32)
    assert output_tensor.dtype == ttnn.int32

    npu_result = ttnn.to_torch(output_tensor)
    expected = eltwise_typecast(torch_input, tt_input_dtype=ttnn.float32, tt_output_dtype=ttnn.int32)
    assert_integer_typecast_equal(expected, npu_result)


@pytest.mark.parametrize(
    "shape, shard_layout, shard_shape",
    [
        # Rows longer than one tile (1056 elements)
        ([1, 1, 8, 2112], ttnn.TensorMemoryLayout.WIDTH_SHARDED, [8, 1056]),
        ([1, 1, 16, 1056], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [8, 1056]),
        # Rows shorter than one tile (64 elements)
        ([1, 1, 8, 128], ttnn.TensorMemoryLayout.WIDTH_SHARDED, [8, 64]),
    ],
    ids=["width_wide", "height_wide", "width_narrow"],
)
@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        (torch.float32, ttnn.float32, ttnn.int32),
        (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
    ],
)
def test_typecast_legacy_sharded_row_major_page_not_tile_sized(
    device, shape, shard_layout, shard_shape, pt_input_dtype, tt_input_dtype, tt_output_dtype
):
    """Sharded ROW_MAJOR input whose rows are shorter or longer than one tile."""
    torch.manual_seed(0)

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)

    in_low, in_high = typecast_test_input_bounds(tt_input_dtype, tt_output_dtype)
    torch_input = make_typecast_test_input(shape, pt_input_dtype, in_low, in_high)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=tt_input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=tt_output_dtype)
    assert output_tensor.dtype == tt_output_dtype

    result = ttnn.to_torch(output_tensor)
    expected = eltwise_typecast(torch_input, tt_input_dtype=tt_input_dtype, tt_output_dtype=tt_output_dtype)
    if uses_exact_integer_typecast_check(tt_input_dtype, tt_output_dtype):
        assert_integer_typecast_equal(expected, result)
    else:
        assert_equal(expected, result)


@pytest.mark.parametrize(
    "shape",
    [
        [120, 104],  # width=104, not divisible by 32 (WAN 480p per-device shard)
        [81, 120, 104],
        [3, 81, 120, 104],
        [1, 3, 81, 120, 104],
        [10, 40],  # width=40, not divisible by 32
        [5, 8],  # width=8, less than one tile row
    ],
)
def test_typecast_rm_non_aligned_width(shape, device):
    """ROW_MAJOR bfloat16→uint8 typecast with width not divisible by 32."""
    torch.manual_seed(42)
    in_low, in_high = typecast_test_input_bounds(ttnn.bfloat16, ttnn.uint8)
    torch_input = make_typecast_test_input(shape, torch.bfloat16, in_low, in_high)
    expected = eltwise_typecast(torch_input, tt_input_dtype=ttnn.bfloat16, tt_output_dtype=ttnn.uint8)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    result = ttnn.to_torch(ttnn.typecast(input_tensor, ttnn.uint8))
    assert_integer_typecast_equal(expected, result)


def test_typecast_rm_chunked_program_cache(device):
    """Regression: program cache hit on ROW_MAJOR typecast must not hang when num_rows % num_cores != 0."""
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    shape = [num_cores + 1, 64]

    in_low, in_high = typecast_test_input_bounds(ttnn.bfloat16, ttnn.uint8)
    torch_input = make_typecast_test_input(shape, torch.bfloat16, in_low, in_high)
    expected = eltwise_typecast(torch_input, tt_input_dtype=ttnn.bfloat16, tt_output_dtype=ttnn.uint8)

    for i in range(2):
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_tensor = ttnn.typecast(input_tensor, ttnn.uint8)
        result = ttnn.to_torch(output_tensor)
        assert_integer_typecast_equal(expected, result)


@pytest.mark.parametrize(
    "shape",
    [
        [16, 16],
        [256],
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    [
        (torch.uint8, ttnn.uint8, ttnn.bfloat16),
        (torch.int8, ttnn.int8, ttnn.bfloat16),
        (torch.int8, ttnn.int8, ttnn.float32),
        (torch.int8, ttnn.int8, ttnn.int32),
        (torch.int8, ttnn.int8, ttnn.uint32),
        # int8 is sign-extended before the int32 -> uint16 conversion, which clamps negatives to 0.
        (torch.int8, ttnn.int8, ttnn.uint16),
        # Both dtypes are the same raw byte, so the conversion is a pure reinterpretation.
        (torch.int8, ttnn.int8, ttnn.uint8),
    ],
)
def test_typecast_8bit_exhaustive(
    shape, layout, memory_config, pt_input_dtype, tt_input_dtype, tt_output_dtype, device
):
    """8-bit typecast covering every representable input value.

    All 256 values of either dtype are exactly representable in bfloat16, float32 and the wider
    integer dtypes, so each conversion must be bit-exact.
    """
    low = 0 if pt_input_dtype == torch.uint8 else -128
    torch_input = torch.arange(low, low + 256, dtype=pt_input_dtype).reshape(shape)
    expected = eltwise_typecast(torch_input, tt_input_dtype=tt_input_dtype, tt_output_dtype=tt_output_dtype)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=tt_input_dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )
    result = ttnn.to_torch(ttnn.typecast(input_tensor, tt_output_dtype))

    if uses_exact_integer_typecast_check(tt_input_dtype, tt_output_dtype):
        assert_integer_typecast_equal(expected, result)
    else:
        assert_equal(expected, result)


@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.int32, ttnn.int32),
        (torch.int32, ttnn.uint16),
        (torch.int32, ttnn.uint32),
    ],
)
def test_typecast_to_int8_wraps(layout, pt_input_dtype, tt_input_dtype, device):
    """Narrowing to int8 truncates towards zero and wraps modulo 256, like the uint8 narrowing."""
    if tt_input_dtype in (ttnn.uint16, ttnn.uint32):
        values = torch.arange(0, 512, dtype=torch.float32)
    else:
        values = torch.arange(-608, 608, dtype=torch.float32)
        if pt_input_dtype != torch.int32:
            values = values + 0.7
    torch_input = values.reshape(1, 1, -1, 32).to(pt_input_dtype)

    input_tensor = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=layout, device=device)
    result = ttnn.to_torch(ttnn.typecast(input_tensor, ttnn.int8))

    assert_integer_typecast_equal(narrow_to_8bit(torch_input, signed=True), result)


@pytest.mark.parametrize("tt_output_dtype", [ttnn.int32, ttnn.uint16, ttnn.float32, ttnn.bfloat16])
def test_unary_chain_typecast_int8(tt_output_dtype, device):
    torch_input = torch.arange(-128, 128, dtype=torch.int8).reshape(1, 1, -1, 32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int8, layout=ttnn.TILE_LAYOUT, device=device)

    chained = ttnn.unary_chain(
        input_tensor,
        [ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.INT8.value, tt_output_dtype.value)],
    )

    expected = ttnn.to_torch(ttnn.typecast(input_tensor, tt_output_dtype))
    assert_equal(expected, ttnn.to_torch(chained))


# ---------------------------------------------------------------------------------------------------------------------
# FP8_E4M3 typecast in TILE layout (Blackhole only).
#
# Device semantics measured on Blackhole p150b (identical for TILE and ROW_MAJOR, and for interleaved DRAM and L1):
#   * fp8 -> bf16 is exact for every normal fp8 value and for +-0.
#   * bf16 -> fp8 rounds toward zero (truncates the mantissa), unlike torch's round-to-nearest-even cast. Overflow
#     matches torch: [448, 480) -> 448, >= 480 and +-inf -> NaN (0x7F / 0xFF), NaN -> NaN.
#   * Deviations from torch, pinned as strict xfails below: -0 -> +0 (both directions), fp8 NaN widens to 512, fp8 subnormals are not
#     produced or consumed correctly.
# ---------------------------------------------------------------------------------------------------------------------
fp8_tile_shapes = [(1, 1, 32, 32), (1, 1, 64, 128), (2, 3, 64, 96), (1, 1, 45, 70)]


def _skip_unless_blackhole():
    from tt_py_test_utils_common.utility_functions import is_blackhole

    if not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")


def _all_fp8_e4m3_values():
    """All 256 fp8_e4m3fn bit patterns, as an fp8 torch tensor."""
    return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)


def _fp8_is_normal_or_zero(fp8):
    bits = fp8.view(torch.uint8)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    return ((exponent != 0) | (mantissa == 0)) & ~fp8.float().isnan()


def _tile_into(values, cols=64):
    """Repeat `values` to fill a (1, 1, rows, cols) tensor with rows a multiple of 32."""
    n = values.numel()
    rows = max(32, ((n + cols - 1) // cols + 31) // 32 * 32)
    reps = (rows * cols + n - 1) // n
    return values.repeat(reps)[: rows * cols].reshape(1, 1, rows, cols)


def _fp8_bits(t):
    return t.contiguous().view(torch.uint8).to(torch.int32)


def _bf16_bits(t):
    return t.contiguous().view(torch.int16).to(torch.int32)


def _bf16_to_fp8_round_toward_zero(x):
    """Reference for the device cast: truncate to a 3-bit mantissa, then encode (exact, so torch's rounding is moot)."""
    truncated = (x.float().view(torch.int32) & ~0xFFFFF).view(torch.float32)
    return truncated.to(torch.float8_e4m3fn)


def _random_fp8_normal_range_bf16(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    magnitude = torch.exp2(torch.empty(shape).uniform_(-6.0, 8.8, generator=generator))  # [2^-6, ~445]
    sign = torch.where(torch.rand(shape, generator=generator) < 0.5, -1.0, 1.0)
    torch_input = (sign * magnitude).to(torch.bfloat16)
    # bf16 midpoints between adjacent fp8 values.
    ties = torch.tensor([1.0625, 1.1875, 2.125, 2.375, -1.0625, -1.1875, 0.0166015625, 68.0], dtype=torch.bfloat16)
    torch_input.view(-1)[: ties.numel()] = ties
    return torch_input


@pytest.mark.parametrize("memory_config", mem_configs, ids=["dram", "l1"])
def test_typecast_fp8_e4m3_to_bf16_tile_exhaustive(device, memory_config):
    """Every normal fp8 value and +0 widens to bf16 exactly on device (fp8 TILE input built on host)."""
    _skip_unless_blackhole()
    all_fp8 = _all_fp8_e4m3_values()
    keep = _fp8_is_normal_or_zero(all_fp8) & (all_fp8.view(torch.uint8) != 0x80)  # -0 -> +0, see xfail below
    golden_fp8 = _tile_into(all_fp8[keep])

    tt_fp8 = ttnn.from_torch(
        golden_fp8.float(), dtype=ttnn.fp8_e4m3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    assert torch.equal(_fp8_bits(ttnn.to_torch(tt_fp8)), _fp8_bits(golden_fp8))

    tt_bf16 = ttnn.typecast(tt_fp8, ttnn.bfloat16, memory_config=memory_config)
    assert tt_bf16.layout == ttnn.TILE_LAYOUT and tt_bf16.dtype == ttnn.bfloat16
    mismatches = (_bf16_bits(ttnn.to_torch(tt_bf16)) != _bf16_bits(golden_fp8.to(torch.bfloat16))).sum().item()
    assert mismatches == 0, f"fp8 -> bf16: {mismatches} bit mismatches"

    tt_fp32 = ttnn.typecast(tt_fp8, ttnn.float32, memory_config=memory_config)
    assert torch.equal(ttnn.to_torch(tt_fp32), golden_fp8.float())


@pytest.mark.parametrize("memory_config", mem_configs, ids=["dram", "l1"])
@pytest.mark.parametrize("shape", fp8_tile_shapes)
def test_typecast_bf16_to_fp8_e4m3_tile(device, shape, memory_config):
    """bf16 -> fp8 in TILE layout: round toward zero, identical to the ROW_MAJOR path, and widening back is exact."""
    _skip_unless_blackhole()
    torch_input = _random_fp8_normal_range_bf16(shape, seed=1234)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    tt_fp8 = ttnn.typecast(tt_input, ttnn.fp8_e4m3, memory_config=memory_config)
    assert tt_fp8.layout == ttnn.TILE_LAYOUT and tt_fp8.dtype == ttnn.fp8_e4m3
    actual = ttnn.to_torch(tt_fp8)
    assert actual.shape == torch_input.shape

    expected = _bf16_to_fp8_round_toward_zero(torch_input)
    mismatches = (_fp8_bits(actual) != _fp8_bits(expected)).sum().item()
    rne_mismatches = (_fp8_bits(actual) != _fp8_bits(torch_input.to(torch.float8_e4m3fn))).sum().item()
    print(f"\nbf16 -> fp8 {shape}: {mismatches} mismatches vs round-toward-zero, {rne_mismatches} vs torch RNE")
    assert mismatches == 0, f"bf16 -> fp8: {mismatches} mismatches vs round-toward-zero of {actual.numel()}"

    # Same bytes as the ROW_MAJOR (chunked) typecast path.
    tt_input_rm = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    actual_rm = ttnn.to_torch(ttnn.typecast(tt_input_rm, ttnn.fp8_e4m3, memory_config=memory_config))
    assert torch.equal(_fp8_bits(actual_rm), _fp8_bits(actual))

    # And back: the fp8 bytes the device produced widen exactly.
    tt_back = ttnn.typecast(tt_fp8, ttnn.bfloat16, memory_config=memory_config)
    assert torch.equal(_bf16_bits(ttnn.to_torch(tt_back)), _bf16_bits(actual.to(torch.bfloat16)))


@pytest.mark.xfail(strict=True, reason="Blackhole bf16 -> fp8 typecast rounds toward zero, not to nearest even")
def test_typecast_bf16_to_fp8_e4m3_tile_matches_torch_rne(device):
    _skip_unless_blackhole()
    torch_input = _random_fp8_normal_range_bf16((1, 1, 64, 128), seed=1234)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.typecast(tt_input, ttnn.fp8_e4m3))
    assert torch.equal(_fp8_bits(actual), _fp8_bits(torch_input.to(torch.float8_e4m3fn)))


def test_typecast_bf16_fp8_e4m3_tile_all_normal_values_round_trip(device):
    """Every normal fp8 value (and +0) given as bf16 survives bf16 -> fp8 -> bf16 on device bit-exactly."""
    _skip_unless_blackhole()
    all_fp8 = _all_fp8_e4m3_values()
    keep = _fp8_is_normal_or_zero(all_fp8) & (all_fp8.view(torch.uint8) != 0x80)  # -0 -> +0, see xfail below
    golden_fp8 = _tile_into(all_fp8[keep])
    torch_input = golden_fp8.to(torch.bfloat16)  # exact: every fp8 value is representable in bf16

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_fp8 = ttnn.typecast(tt_input, ttnn.fp8_e4m3)
    assert torch.equal(_fp8_bits(ttnn.to_torch(tt_fp8)), _fp8_bits(golden_fp8))
    tt_back = ttnn.to_torch(ttnn.typecast(tt_fp8, ttnn.bfloat16))
    assert torch.equal(_bf16_bits(tt_back), _bf16_bits(torch_input))


def test_typecast_bf16_to_fp8_e4m3_tile_overflow_and_nan(device):
    """Overflow / inf / NaN inputs map like torch's cast: [448, 480) -> +-448, larger and +-inf -> NaN."""
    _skip_unless_blackhole()
    values = torch.tensor([448.0, 449.0, 464.0, 479.0, 480.0, 1000.0, 65280.0, float("inf")], dtype=torch.bfloat16)
    values = torch.cat([values, -values, torch.tensor([float("nan")], dtype=torch.bfloat16)])
    torch_input = _tile_into(values, cols=32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.typecast(tt_input, ttnn.fp8_e4m3)).reshape(-1)[: values.numel()]
    expected = values.to(torch.float8_e4m3fn)
    assert torch.equal(_fp8_bits(actual), _fp8_bits(expected)), list(zip(values.tolist(), actual.float().tolist()))


@pytest.mark.xfail(strict=True, reason="Blackhole bf16 -> fp8 typecast with fp32 DEST turns -0 into +0")
def test_typecast_bf16_to_fp8_e4m3_tile_negative_zero(device):
    _skip_unless_blackhole()
    torch_input = torch.full((1, 1, 32, 32), -0.0, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.typecast(tt_input, ttnn.fp8_e4m3))
    assert torch.all(_fp8_bits(actual) == 0x80)


@pytest.mark.xfail(strict=True, reason="Blackhole fp8 -> bf16 typecast with fp32 DEST turns -0 into +0")
def test_typecast_fp8_e4m3_to_bf16_tile_negative_zero(device):
    _skip_unless_blackhole()
    tt_fp8 = ttnn.from_torch(
        torch.full((1, 1, 32, 32), -0.0), dtype=ttnn.fp8_e4m3, layout=ttnn.TILE_LAYOUT, device=device
    )
    assert torch.all(_fp8_bits(ttnn.to_torch(tt_fp8)) == 0x80)
    assert torch.all(
        _bf16_bits(ttnn.to_torch(ttnn.typecast(tt_fp8, ttnn.bfloat16)))
        == _bf16_bits(torch.tensor(-0.0, dtype=torch.bfloat16))
    )


@pytest.mark.xfail(strict=True, reason="Blackhole fp8 -> bf16 typecast widens fp8 NaN (0x7F) to 512, not NaN")
def test_typecast_fp8_e4m3_to_bf16_tile_nan(device):
    _skip_unless_blackhole()
    tt_fp8 = ttnn.from_torch(
        torch.full((1, 1, 32, 32), float("nan")), dtype=ttnn.fp8_e4m3, layout=ttnn.TILE_LAYOUT, device=device
    )
    assert torch.all(_fp8_bits(ttnn.to_torch(tt_fp8)) == 0x7F)
    assert torch.all(ttnn.to_torch(ttnn.typecast(tt_fp8, ttnn.bfloat16)).isnan())


@pytest.mark.xfail(strict=True, reason="Blackhole fp8 typecast does not produce fp8 subnormals (2^-9 .. 7 * 2^-9)")
def test_typecast_bf16_to_fp8_e4m3_tile_subnormals(device):
    _skip_unless_blackhole()
    all_fp8 = _all_fp8_e4m3_values()
    subnormals = all_fp8[~_fp8_is_normal_or_zero(all_fp8) & ~all_fp8.float().isnan()]
    golden_fp8 = _tile_into(subnormals, cols=32)
    tt_input = ttnn.from_torch(
        golden_fp8.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    actual = ttnn.to_torch(ttnn.typecast(tt_input, ttnn.fp8_e4m3))
    assert torch.equal(_fp8_bits(actual), _fp8_bits(golden_fp8))

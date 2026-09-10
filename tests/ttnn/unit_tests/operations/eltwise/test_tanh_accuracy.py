# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose, assert_with_ulp

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008), (torch.float32, ttnn.float32, 0.003)]
)
def test_tanh_range(device, torch_dtype, ttnn_dtype, atol):
    torch_input_tensor_a = torch.tensor(
        [
            [
                [
                    [
                        -1.8125,
                        -2.828125,
                        -3.125,
                        -3.234375,
                        -2.765625,
                        -1.890625,
                        -3.359375,
                        -2.0625,
                        -3.015625,
                        -2.203125,
                        -2.015625,
                        -2.9375,
                        -1.3046875,
                        -1.359375,
                        -1.3984375,
                        -1.2265625,
                        -2,
                        -3,
                        -1.5,
                        -2.5,
                        -3.5,
                        -3.75,
                        -3.359375,
                        -1.8828125,
                        -3.255,
                        -0.9,
                        -0.1,
                        0.25,
                        0.75,
                        -0.8359375,
                        -0.5,
                        0.9,
                    ]
                ]
            ]
        ],
        dtype=torch_dtype,
    )
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.tanh(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_allclose(output_tensor, torch_output_tensor, rtol=1e-05, atol=atol)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    # PCC and max abs error against torch.tanh over this test's 32-value input,
    # measured on a Wormhole n150 (2026-09-10) after the approximate-tanh SFPLUT
    # retune -- see APPROX_TANH_RETUNE.md:
    #
    #   bfloat16  approx=False  pcc 0.9999898975945355   max|err| 0.003906
    #   bfloat16  approx=True   pcc 0.9992522964571828   max|err| 0.056641
    #   float32   approx=False  pcc 0.9999999999999984   max|err| 0.000000
    #   float32   approx=True   pcc 0.9992540536236625   max|err| 0.055867
    #
    # The retune lifted approximate mode from pcc 0.9978378297942829 (bfloat16) and
    # 0.9977552960423647 (float32). Its max|err| here is set by x = -0.5, the sampled
    # point nearest the fit's error peak (|x| = 0.4636, where the error is 0.056339):
    # |tanh(0.5) - 0.8125*0.5| = 0.0558672, which the float32 row reproduces exactly.
    #
    # The accurate-path numbers also differ from the previously recorded
    # 0.9999663646890817 (bfloat16) / 0.9999829606828651 (float32). That is NOT from
    # the retune, which only touches the APPROXIMATION_MODE branch -- those predate
    # later work on the accurate path, from the era of the fpu-vs-sfpu arithmetic
    # split recorded as pcc 0.9999583453515977 (fpu) vs 0.9999669593009368 (sfpu).
    #
    # Not re-measured: the timing below. The retune moves two SFPLOADI immediates and
    # changes no instructions, so it cannot affect it.
    # Single-tile tanh: accurate = 7886ns, approx = 1789ns (~77% faster)
    assert pcc


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008), (torch.float32, ttnn.float32, 0.003)]
)
@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),
        (100, -100),
        (4, -4),
    ],
)
def test_tanh_inplace(device, high, low, torch_dtype, ttnn_dtype, atol):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand([1, 9, 8192], dtype=torch_dtype) * (high - low) + low
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.tanh(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=input_tensor_a)
    output_tensor = ttnn.to_torch(input_tensor_a)

    assert_allclose(output_tensor, torch_output_tensor, rtol=1e-05, atol=atol)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    assert pcc


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008), (torch.float32, ttnn.float32, 0.003)]
)
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([2, 4, 320, 1024])),
        (torch.Size([1, 9, 8192])),
    ),
)
@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),  # pcc_msg 0.99998
        (100, -100),
        (10000, -10000),
        (4, -4),  # pcc_msg 0.9999948671754642
    ],
)
def test_tanh_accuracy(device, input_shapes, high, low, torch_dtype, ttnn_dtype, atol):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((input_shapes), dtype=torch_dtype) * (high - low) + low
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.tanh(input_tensor)
    output_tensor = ttnn.to_torch(output)

    assert_allclose(output_tensor, torch_output_tensor, rtol=1e-05, atol=atol)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    # pcc_msg 0.9999 or above
    assert pcc


@pytest.mark.parametrize("torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008)])
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 89600, 32])),),
)
@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),
        (100, -100),
        (10000, -10000),
        (4, -4),
    ],
)
def test_tanh_height_sharded(device, input_shapes, high, low, torch_dtype, ttnn_dtype, atol):
    torch.manual_seed(0)

    in_data = torch.rand((input_shapes), dtype=torch_dtype) * (high - low) + low
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 6),
            ),
        }
    )
    n_cores = 56
    N, C, H, W = in_data.shape
    shard_spec = ttnn.ShardSpec(shard_grid, [N * C * H // n_cores, W], ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor1 = ttnn.from_torch(
        in_data,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    output_tensor = ttnn.tanh(input_tensor1)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden_tensor = golden_function(in_data)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)
    assert_with_ulp(expected_result=golden_tensor, actual_result=output_tensor, ulp_threshold=1)


def return_mem_config(mem_config_string):
    if mem_config_string == "l1_height_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 8, 512),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_height_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512, 512 // 8),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(512, 512 // 8),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 8, 512),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 2, 512 // 4),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 2, 512 // 4),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    raise ("Input mem_config_string is not valid!")


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008), (torch.float32, ttnn.float32, 0.003)]
)
@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),
        (100, -100),
        (10000, -10000),
        (4, -4),
    ],
)
@pytest.mark.parametrize(
    "input_mem_config",
    [
        "l1_height_sharded_rm",
        "l1_height_sharded_cm",
        "l1_width_sharded_rm",
        "l1_width_sharded_cm",
        "l1_block_sharded_rm",
        "l1_block_sharded_cm",
    ],
)
def test_tanh_sharded(device, high, low, input_mem_config, torch_dtype, ttnn_dtype, atol):
    torch.manual_seed(0)

    in_data = torch.rand([1, 1, 512, 512], dtype=torch_dtype) * (high - low) + low

    input_tensor1 = ttnn.from_torch(
        in_data,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=return_mem_config(input_mem_config),
    )
    output_tensor = ttnn.tanh(input_tensor1)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden_tensor = golden_function(in_data)

    pcc, pcc_msg = assert_with_pcc(golden_tensor, output_tensor, 0.999)
    assert pcc


def test_tanh_fp32_special_values(device):
    input_tensor = torch.tensor(
        [
            float("nan"),
            -0.0,
            0.0,
            float("inf"),
            float("-inf"),
            1.0,
            -1.0,
            10.0,
            -10.0,
        ],
        dtype=torch.float32,
    )

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.tanh(tt_in)
    result = ttnn.to_torch(tt_result)

    # tanh(NaN) == NaN
    assert torch.equal(torch.isnan(result), torch.isnan(golden))
    # tanh(+Inf) != 1.0
    assert torch.equal(torch.isposinf(result), torch.isposinf(golden))
    # tanh(-Inf) != -1.0
    assert torch.equal(torch.isneginf(result), torch.isneginf(golden))

    # tanh(-0.0) == -0.0
    finite_mask = ~torch.isnan(golden)
    assert torch.equal(
        torch.signbit(result)[finite_mask], torch.signbit(golden)[finite_mask]
    ), f"Sign bit mismatch: result={result.tolist()} golden={golden.tolist()}"

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=5, allow_nonfinite=True)

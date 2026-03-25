# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch
from functools import partial

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics, assert_with_pcc
from models.common.utility_functions import torch_random
from tests.ttnn.unit_tests.operations.reduce.numeric_check import (
    collect_and_dump_numeric_metrics,
)

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [32, 48, 64, 80, 96, 112, 128])
@pytest.mark.parametrize("c", [9 * 64])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_3D_tensor(device, batch_size, h, w, c, n, dim, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((n, c, h, w))
    golden_function = ttnn.get_golden_function(ttnn.sum)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=output_memory_config)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    if dim:
        rank = 4
        if isinstance(dim, tuple):
            for d in dim:
                if d < 0:
                    d += rank
        else:
            if dim < 0:
                dim += rank
        output_tensor = ttnn.to_torch(output_tensor).squeeze(dim=dim)
    else:
        output_tensor = ttnn.to_torch(output_tensor).squeeze()
    # Collect numeric metrics and dump to CSV using reusable function
    test_name = f"test_3D_tensor[batch_size={batch_size},h={h},w={w},c={c},n={n},dim={dim},input_dtype={input_dtype},input_memory_config={input_memory_config},output_memory_config={output_memory_config}]"
    collect_and_dump_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        test_name=test_name,
        csv_filename="test_reduction_h_interleaved_numeric_results.csv",
        test_params=None,
    )
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.9999,
        rtol=8.160001,
        atol=8.160001,
        frobenius_threshold=0.000377427112,
    )
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize(
    "w", [7 * 64 * 32, 7 * 64 * 48, 7 * 64 * 64, 7 * 64 * 80, 7 * 64 * 96, 7 * 64 * 112, 7 * 64 * 128]
)
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_2D_tensor_full_grid(
    device, batch_size, h, w, c, n, dim, input_dtype, input_memory_config, output_memory_config
):
    torch.manual_seed(0)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((n, c, h, w))
    golden_function = ttnn.get_golden_function(ttnn.sum)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=output_memory_config)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    if dim:
        rank = 4
        if isinstance(dim, tuple):
            for d in dim:
                if d < 0:
                    d += rank
        else:
            if dim < 0:
                dim += rank
        output_tensor = ttnn.to_torch(output_tensor).squeeze(dim=dim)
    else:
        output_tensor = ttnn.to_torch(output_tensor).squeeze()
    # Collect numeric metrics and dump to CSV using reusable function
    test_name = f"test_2D_tensor_full_grid[batch_size={batch_size},h={h},w={w},c={c},n={n},dim={dim},input_dtype={input_dtype},input_memory_config={input_memory_config},output_memory_config={output_memory_config}]"
    collect_and_dump_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        test_name=test_name,
        csv_filename="test_reduction_h_interleaved_numeric_results.csv",
        test_params=None,
    )
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.9999,
        rtol=1.36265776,
        atol=8.160001,
        frobenius_threshold=0.000400773076,
    )
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [32, 64, 96, 128])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("dim", [-2])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_2D_tensor(device, batch_size, h, w, c, n, dim, input_dtype, input_memory_config, output_memory_config):
    torch.manual_seed(0)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )((n, c, h, w))
    golden_function = ttnn.get_golden_function(ttnn.sum)
    torch_output_tensor = golden_function(torch_input_tensor, dim=dim, memory_config=output_memory_config)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    if dim:
        rank = 4
        if isinstance(dim, tuple):
            for d in dim:
                if d < 0:
                    d += rank
        else:
            if dim < 0:
                dim += rank
        output_tensor = ttnn.to_torch(output_tensor).squeeze(dim=dim)
    else:
        output_tensor = ttnn.to_torch(output_tensor).squeeze()
    # Collect numeric metrics and dump to CSV using reusable function
    test_name = f"test_2D_tensor[batch_size={batch_size},h={h},w={w},c={c},n={n},dim={dim},input_dtype={input_dtype},input_memory_config={input_memory_config},output_memory_config={output_memory_config}]"
    collect_and_dump_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        test_name=test_name,
        csv_filename="test_reduction_h_interleaved_numeric_results.csv",
        test_params=None,
    )
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999869,
        rtol=1.020001,
        atol=12.240001,
        frobenius_threshold=0.00895955554,
    )
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc, skip_for_blackhole
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)


def run_clone(
    shape,
    input_memory_config,
    output_memory_config,
    input_dtype,
    output_dtype,
    tilized,
    compute_kernel_options,
    device,
):
    """
    Function to test the clone operation on NPU. Generates random input data, clones it on NPU,
    and compares the output with the CPU clone for correctness.

    Parameters:
    shape: tuple
        Shape of the input tensor.
    input_memory_config: MemoryConfig
        Memory configuration for the input tensor on NPU.
    output_memory_config: MemoryConfig
        Memory configuration for the output tensor on NPU.
    input_dtype: str
        Data type of the input tensor ('int32' or other).
    output_dtype: str or None
        Data type of the output tensor (must be None or match input_dtype when not tilized).
    tilized: bool
        Whether to use TILE_LAYOUT or ROW_MAJOR_LAYOUT for NPU tensor.
    compute_kernel_options:
        Configuration options for the compute kernel.
    device: ttnn.device
        Device where the operation is performed (e.g., NPU device).

    Raises:
    pytest.skip: When certain conditions on dtype mismatch or layout are not met.
    """
    if input_dtype == "int32":
        torch_input = torch.randint(low=-10, high=11, size=shape, dtype=get_lib_dtype(torch, input_dtype))
    else:
        torch_input = 2 * torch.rand(size=shape, dtype=get_lib_dtype(torch, input_dtype)) - 1

    if input_dtype == "int32":
        if output_dtype and output_dtype != "int32":
            pytest.skip("For int32 input, output_dtype must be None or int32.")
    if output_dtype == "int32" and input_dtype != "int32":
        pytest.skip("For int32 output, input_dtype must also be int32.")
    if output_dtype != input_dtype and output_dtype and not tilized:
        pytest.skip("When not tilized, dtype conversion is not supported.")
    if not tilized and shape and shape[-1] % 2 == 1:
        pytest.skip("TTNN failed to create a ROW_MAJOR_LAYOUT tensor from a PyTorch tensor with an odd last dimension.")

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=get_lib_dtype(ttnn, input_dtype),
        layout=ttnn.TILE_LAYOUT if tilized else ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config,
    )

    ttnn_output = ttnn.clone(
        ttnn_input,
        dtype=get_lib_dtype(ttnn, output_dtype),
        memory_config=output_memory_config,
        compute_kernel_config=get_compute_kernel_options(compute_kernel_options),
    )

    torch_output = ttnn.to_torch(ttnn_output).reshape(shape)

    passing, out = comp_allclose_and_pcc(torch.ops.aten.clone(torch_input), torch_output, rtol=0.01, atol=0.01)
    logger.info(out)
    assert passing


memory_config_list = [
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
]


@pytest.mark.parametrize(
    "shape",
    [
        [10],  # 1d
        [10, 10],  # 2d
        [10, 10, 10],  # 3d
        [10, 10, 10, 10],  # 4d
        [1, 1, 1, 30],  # Single core
        [1, 1, 300, 380],  # Multi core
        [1, 3, 320, 380],  # Multi core
        [1, 1, 32, 32],  # Single core
        [1, 1, 320, 384],  # Multi core
        [1, 3, 320, 384],  # Multi core
        [38, 2, 99, 181],  # Odd last Dim
        [5, 33, 319, 381],  # Odd last Dim
    ],
)
@pytest.mark.parametrize(
    "tilized",
    [True, False],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        "bfloat16",
        "int32",
    ],
)
def test_clone_shape(
    shape,
    tilized,
    input_dtype,
    device,
):
    """
    Test case to verify the clone operation on different tensor shapes and layouts (tilized or not).
    """
    torch.manual_seed(2024)
    run_clone(
        shape,
        memory_config_list[0],
        memory_config_list[0],
        input_dtype,
        None,
        tilized,
        None,
        device,
    )


@pytest.mark.parametrize(
    "input_memory_config",
    memory_config_list,
)
@pytest.mark.parametrize(
    "output_memory_config",
    [*memory_config_list, None],
)
@pytest.mark.parametrize(
    "tilized",
    [True, False],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        "bfloat16",
        "int32",
    ],
)
def test_clone_memory_config(
    input_memory_config,
    output_memory_config,
    tilized,
    input_dtype,
    device,
):
    """
    Test case to verify the clone operation with different memory configurations (input/output)
    and layout configurations (tilized or not).
    """
    torch.manual_seed(2024)
    run_clone(
        [1, 3, 320, 384],
        input_memory_config,
        output_memory_config,
        input_dtype,
        None,
        tilized,
        None,
        device,
    )


@pytest.mark.parametrize(
    "input_dtype",
    [
        "bfloat16",
        "float32",
        "int32",
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        "bfloat16",
        "float32",
        "int32",
        None,
    ],
)
@pytest.mark.parametrize(
    "tilized",
    [True, False],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_clone_dtype_conversion(
    input_dtype,
    output_dtype,
    tilized,
    compute_kernel_options,
    device,
):
    """
    Test case to verify the clone operation with various input/output dtype combinations and compute kernel configs.
    """
    torch.manual_seed(2024)
    run_clone(
        [1, 3, 320, 384],
        memory_config_list[0],
        memory_config_list[0],
        input_dtype,
        output_dtype,
        tilized,
        compute_kernel_options,
        device,
    )


@pytest.mark.parametrize(
    "tilized",
    [True, False],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        "bfloat16",
        "int32",
    ],
)
def test_clone_callback(
    tilized,
    input_dtype,
    device,
):
    """
    Test case to verify the clone operation with various input/output dtype combinations.
    """
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_clone(
            [1, 3, 320, 384],
            memory_config_list[0],
            memory_config_list[0],
            input_dtype,
            None,
            tilized,
            None,
            device,
        )
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 64 * 32, 64 * 32],
    ],
)
@pytest.mark.parametrize(
    "shard_strategy",
    [
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardStrategy.BLOCK,
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
    "input_dtype",
    [
        "bfloat16",
        "float32",
    ],
)
def test_clone_sharded_tilized(
    shape,
    shard_strategy,
    shard_orientation,
    input_dtype,
    device,
):
    """
    Test case to verify the clone operation with tilized sharded tensors.
    Tests various sharding strategies (HEIGHT, WIDTH, BLOCK) with identical input/output shard specs.
    """
    torch.manual_seed(2024)

    compute_grid_size = device.compute_with_storage_grid_size()
    x_grid_size = compute_grid_size.x
    y_grid_size = compute_grid_size.y
    while shape[-1] % x_grid_size != 0:
        x_grid_size = x_grid_size - 1
    while shape[-2] % y_grid_size != 0:
        y_grid_size = y_grid_size - 1
    shard_grid = ttnn.CoreGrid(y=y_grid_size, x=x_grid_size)

    shard_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=shard_grid,
        strategy=shard_strategy,
        orientation=shard_orientation,
    )

    run_clone(
        shape=shape,
        input_memory_config=shard_memory_config,
        output_memory_config=shard_memory_config,
        input_dtype=input_dtype,
        output_dtype=None,
        tilized=True,
        compute_kernel_options=None,
        device=device,
    )


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 128, 64],
    ],
)
@pytest.mark.parametrize(
    "shard_strategy",
    [
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardStrategy.BLOCK,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        "bfloat16",
        "float32",
    ],
)
def test_clone_sharded_row_major(
    shape,
    shard_strategy,
    input_dtype,
    device,
):
    """
    Test case to verify the clone operation with sharded tensors in ROW_MAJOR layout.
    """
    torch.manual_seed(2024)

    compute_grid_size = device.compute_with_storage_grid_size()
    x_grid_size = compute_grid_size.x
    y_grid_size = compute_grid_size.y
    while shape[-1] % x_grid_size != 0:
        x_grid_size = x_grid_size - 1
    while shape[-2] % y_grid_size != 0:
        y_grid_size = y_grid_size - 1

    shard_grid = ttnn.CoreGrid(y=y_grid_size, x=x_grid_size)

    shard_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=shard_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    run_clone(
        shape=shape,
        input_memory_config=shard_memory_config,
        output_memory_config=shard_memory_config,
        input_dtype=input_dtype,
        output_dtype=None,
        tilized=False,
        compute_kernel_options=None,
        device=device,
    )


@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        ("bfloat16", "float32"),
        ("float32", "bfloat16"),
    ],
)
def test_clone_sharded_dtype_conversion(
    input_dtype,
    output_dtype,
    device,
):
    """
    Test case to verify the clone operation with sharded tensors and dtype conversion.
    Dtype conversion is only supported with TILE layout.
    """
    torch.manual_seed(2024)

    shape = [1, 1, 64 * 32, 32]

    compute_grid_size = device.compute_with_storage_grid_size()

    x_grid_size = compute_grid_size.x
    y_grid_size = compute_grid_size.y
    while shape[-1] % x_grid_size != 0:
        x_grid_size = x_grid_size - 1
    while shape[-2] % y_grid_size != 0:
        y_grid_size = y_grid_size - 1

    shard_grid = ttnn.CoreGrid(y=y_grid_size, x=x_grid_size)

    shard_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    run_clone(
        shape=shape,
        input_memory_config=shard_memory_config,
        output_memory_config=shard_memory_config,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        tilized=True,
        compute_kernel_options=None,
        device=device,
    )

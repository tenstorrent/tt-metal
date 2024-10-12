# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import copy

import ttnn
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger


def to_ttnn(
    torch_tensor,
    device,
    *,
    ttnn_layout=ttnn.TILE_LAYOUT,
):
    """
    Converts a PyTorch tensor to a TTNN tensor with the specified device and layout.

    This function handles tensors with 0 or 1 dimensions by reshaping them to have at least 2 dimensions:
    - If the input tensor is a scalar (0-dimensional), it is reshaped to shape `[1, 1]`.
    - If the input tensor is 1-dimensional, it is reshaped to shape `[1, length]`.

    Parameters:
    torch_tensor: torch.Tensor or None
        The PyTorch tensor to convert. If None, the function returns None.
    device: ttnn.device
        The device to which the TTNN tensor will be moved.
    ttnn_layout: ttnn.TILE_LAYOUT or ttnn.ROW_MAJOR_LAYOUT, optional
        The desired layout for the TTNN tensor. Defaults to `ttnn.TILE_LAYOUT`.

    Returns:
    ttnn.Tensor or None
        The converted TTNN tensor, with the specified layout and moved to the specified device.
    """
    if torch_tensor is None:
        return None
    if len(torch_tensor.shape) == 0:
        torch_tensor = torch.reshape(torch_tensor, shape=[1, 1])
    if len(torch_tensor.shape) == 1:
        torch_tensor = torch.reshape(torch_tensor, shape=[1, len(torch_tensor)])
    ttnn_tensor = ttnn.from_torch(torch_tensor)
    if ttnn_layout == ttnn.TILE_LAYOUT:
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout=ttnn.TILE_LAYOUT)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device=device)
    return ttnn_tensor


def to_torch(
    ttnn_tensor,
    shape,
):
    """
    Converts a TTNN tensor back to a PyTorch tensor with the specified shape.

    This function handles tensors with 0 or 1 dimensions by adjusting the unpadding shape:
    - If the specified shape is empty (`len(shape) == 0`), `unpad_shape` is set to `[1, 1]`.
    - If the specified shape is 1-dimensional, `unpad_shape` is set to `[1, shape[0]]`.

    The TTNN tensor is converted to `ROW_MAJOR_LAYOUT` and unpadded using `unpad_shape`.
    It is then converted to a PyTorch tensor and reshaped to the specified shape.

    Parameters:
    ttnn_tensor: ttnn.Tensor or None
        The TTNN tensor to convert. If None, the function returns None.
    shape: list
        The desired shape of the output PyTorch tensor.

    Returns:
    torch.Tensor or None
        The converted PyTorch tensor reshaped to the specified shape.
    """
    if ttnn_tensor is None:
        return None
    unpad_shape = copy.deepcopy(shape)
    if len(shape) == 0:
        unpad_shape = [1, 1]
    if len(shape) == 1:
        unpad_shape = [1, shape[0]]
    ttnn_tensor = ttnn_tensor.cpu()
    if ttnn_tensor.layout == ttnn.TILE_LAYOUT:
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout=ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(unpad_shape)
    torch_tensor = ttnn.to_torch(ttnn_tensor)
    torch_tensor = torch.reshape(torch_tensor, shape=shape)
    return torch_tensor


def get_lib_dtype(lib, dtype):
    """
    Maps string-based data types to their corresponding library-specific dtypes.

    Parameters:
    lib: library module (e.g., torch, ttnn)
        The library for which the dtype mapping is required.
    dtype: str
        The string representation of the data type (e.g., 'bfloat16', 'float32', 'int32').

    Returns:
    Corresponding library-specific dtype or None if not found.
    """
    dtype_map = {
        "bfloat16": lib.bfloat16,
        "float32": lib.float32,
        "int32": lib.int32,
    }
    return dtype_map.get(dtype, None)


def run_clone(
    shape,
    input_memory_config,
    output_memory_config,
    input_dtype,
    output_dtype,
    tilized,
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

    ttnn_input = to_ttnn(
        torch_input,
        device,
        ttnn_layout=ttnn.TILE_LAYOUT if tilized else ttnn.ROW_MAJOR_LAYOUT,
    ).to(device, input_memory_config)

    ttnn_output = ttnn.clone(
        ttnn_input,
        dtype=get_lib_dtype(ttnn, output_dtype),
        memory_config=output_memory_config,
    )

    torch_output = to_torch(ttnn_output, shape)

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
def test_clone_shape(
    shape,
    tilized,
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
        "bfloat16",
        None,
        tilized,
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
def test_clone_memory_config(
    input_memory_config,
    output_memory_config,
    tilized,
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
        "bfloat16",
        None,
        tilized,
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
def test_clone_dtype_conversion(
    input_dtype,
    output_dtype,
    tilized,
    device,
):
    """
    Test case to verify the clone operation with various input/output dtype combinations.
    """
    torch.manual_seed(2024)
    run_clone(
        [1, 3, 320, 384],
        memory_config_list[0],
        memory_config_list[0],
        input_dtype,
        output_dtype,
        tilized,
        device,
    )


@pytest.mark.parametrize(
    "tilized",
    [True, False],
)
def test_clone_callback(
    tilized,
    device,
    use_program_cache,
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
            "bfloat16",
            None,
            tilized,
            device,
        )
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = to_ttnn(torch_dummy, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]

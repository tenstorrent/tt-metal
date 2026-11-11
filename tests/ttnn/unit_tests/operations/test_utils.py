# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
import math
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


compute_kernel_options = [False, True]
compute_kernel_ids = ["fp32_dest_acc_en=False", "fp32_dest_acc_en=True"]


def round_up(a, b):
    """
    Round up a to the nearest multiple of b
    """
    return b * math.ceil(a / b)


def get_compute_kernel_options(fp32_dest_acc_en):
    if fp32_dest_acc_en is None:
        return None
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


def to_torch(ttnn_tensor, *, shape=None):
    """
    Converts a ttnn tensor to a torch tensor. If ttnn tensor is None, returns None.
    If shape specified, reshapes the resulting torch tensor to the given shape.
    """
    if ttnn_tensor is None:
        return None
    torch_tensor = ttnn.to_torch(ttnn_tensor)
    if shape is not None:
        torch_tensor = torch_tensor.reshape(shape)
    return torch_tensor


def to_ttnn(
    torch_tensor,
    *,
    device=None,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
    shape=None,
):
    """
    Converts a torch tensor to a ttnn tensor, with optional arguments to control
    the device, data type, memory layout, and memory configuration. The tensor can
    also be reshaped if a shape is provided. If the torch tensor is a scalar (a tensor
    with zero dimensions), it will be automatically reshaped to have a shape of [1, 1].
    """
    if torch_tensor is None:
        return None
    if shape is not None:
        torch_tensor = torch_tensor.reshape(shape)
    if len(torch_tensor.shape) == 0:
        torch_tensor = torch_tensor.reshape([1, 1])
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


# For keepdim in torch
def filter_indices(output_shape, dims):
    def not_in_dims(index_value_pair):
        index, value = index_value_pair
        return index not in dims

    filtered_elements = list(filter(not_in_dims, enumerate(output_shape)))
    filtered_values = [value for index, value in filtered_elements]

    return filtered_values


# For keepdim in tt
def filter_indices_with_last_two(output_shape, dims):
    last_two_elements = output_shape[-2:]
    remaining_elements = output_shape[:-2]

    def not_in_dims(index_value_pair):
        index, _ = index_value_pair
        return index not in dims

    filtered_remaining_elements = list(filter(not_in_dims, enumerate(remaining_elements)))
    filtered_remaining_values = [value for index, value in filtered_remaining_elements]
    final_output_shape = filtered_remaining_values + last_two_elements

    return final_output_shape


def compute_output_shape(input_shape, dim, keepdim=False):
    if dim is None or dim == []:
        dim = list(range(len(input_shape)))

    if isinstance(dim, int):
        dim = [dim]

    output_shape = list(input_shape)

    for d in dim:
        output_shape[d] = 1

    if keepdim:
        torch_output_shape = output_shape.copy()
        tt_output_shape = output_shape.copy()
    else:
        torch_output_shape = filter_indices(output_shape, dim)
        tt_output_shape = filter_indices_with_last_two(output_shape, dim)

    return torch_output_shape, tt_output_shape


def check_dim(input_shape, dim, keepdim):
    if type(dim) == int and dim >= len(input_shape):
        pytest.skip("dim bigger than input rank")

    if type(dim) == list:
        for i in dim:
            if i >= len(input_shape):
                pytest.skip("dim bigger than input rank")

    if keepdim == False:
        if dim in [None, []]:
            pytest.skip("`keepdim == false` don't support last 2-dim")

        if type(dim) == int and len(input_shape) - 2 <= dim:
            pytest.skip("`keepdim == false` don't support last 2-dim")

        if type(dim) == list:
            for i in dim:
                if len(input_shape) - 2 <= i:
                    pytest.skip("`keepdim == false` don't support last 2-dim")


def get_lib_dtype(lib, dtype):
    """
    Maps string-based data types to their corresponding library-specific dtypes.

    Parameters:
    lib: library module (e.g., torch, ttnn)
        The library for which the dtype mapping is required.
    dtype: str
        The string representation of the data type (e.g., 'bfloat16', 'float32', 'int32', 'bfloat8_b').

    Returns:
    Corresponding library-specific dtype or None if not found.
    """
    if hasattr(lib, "bfloat8_b") and dtype == "bfloat8_b":
        return lib.bfloat8_b
    else:
        dtype_map = {
            "bfloat16": lib.bfloat16,
            "float32": lib.float32,
            "int32": lib.int32,
        }
        return dtype_map.get(dtype, None)


def get_ttnn_torch_dtype(ttnn_dtype: ttnn.DataType) -> torch.dtype:
    """
    Maps a ttnn.DataType to the corresponding torch dtype that can handle them.
    Parameters:
    ttnn_dtype: ttnn.DataType
        The ttnn data type to be mapped.
    Returns:
    torch.dtype or None
        The corresponding torch dtype if the mapping exists, otherwise None.
    """
    dtype_map = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.bfloat8_b: torch.bfloat16,
        ttnn.int32: torch.int32,
    }
    return dtype_map.get(ttnn_dtype, None)


def create_ttnn_tilized_tensor(torch_tensor, device, dtype, pad_value=float("nan")):
    return ttnn.from_torch(torch_tensor, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, pad_value=pad_value)


def check_second_call_through_cache(device, build, run):
    """Run an op on tensors A, then on new tensors B at other addresses, and check the second output.

    On a program cache hit only the buffer addresses are patched, so a stale address makes the second call
    read or write the buffer of the first. The second call must not add cache entries, and its output must
    match torch and, bit for bit, the same call on a cold cache.

    build(seed, second_call) allocates fresh device tensors from seeded data. It returns the inputs for
    run() and the torch reference for them. second_call is False for the first run and True for the
    others, so a test can change how the operands alias between the two calls.
    run(inputs) runs the op and returns its output as a float32 torch tensor.
    """
    # first_inputs and the spacer stay referenced until return, so the new tensors cannot reuse their addresses.
    first_inputs, _ = build(0, second_call=False)
    run(first_inputs)

    spacer = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    second_inputs, golden = build(1, second_call=True)

    entries_before = device.num_program_cache_entries()
    output = run(second_inputs)
    assert device.num_program_cache_entries() == entries_before, "second call missed the program cache"

    # Loose on purpose: a stale address gives garbage, and the exact check is the cold-cache comparison.
    assert_with_pcc(golden, output, 0.99)

    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    cold_inputs, _ = build(1, second_call=True)
    cold_output = run(cold_inputs)
    assert torch.equal(output, cold_output), "cached second call differs from the same call on a cold cache"

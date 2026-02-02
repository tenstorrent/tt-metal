# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import json
import time

from loguru import logger
from models.common.utility_functions import comp_pcc, comp_allclose, comp_ulp, comp_equal, divup, roundup
from typing import Tuple, Union

import ttnn
import torch
import numpy as np


# Dictionaries for converting dtypes
tt_dtype_to_torch_dtype = {
    ttnn.uint8: torch.uint8,
    ttnn.uint16: torch.int16,
    ttnn.uint32: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.float32: torch.float,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.float,
    ttnn.bfloat4_b: torch.float,
}

tt_dtype_to_np_dtype = {
    ttnn.uint8: np.ubyte,
    ttnn.uint16: np.int16,
    ttnn.uint32: np.int32,
    ttnn.int32: np.int32,
    ttnn.float32: np.float32,
    ttnn.bfloat8_b: np.float32,
    ttnn.bfloat4_b: np.float32,
}


def construct_pcc_assert_message(message, expected_pytorch_result, actual_pytorch_result):
    messages = []
    messages.append(message)
    # messages.append("Expected")
    # messages.append(str(expected_pytorch_result))
    # messages.append("Actual")
    # messages.append(str(actual_pytorch_result))
    messages = [str(m) for m in messages]
    return "\n".join(messages)


def assert_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    """
    Assert that two PyTorch tensors are similar within a specified Pearson Correlation Coefficient (PCC) threshold.

    This function compares two tensors using PCC, which measures the linear correlation between them.
    It's particularly useful for floating-point comparisons where exact equality is not expected due to
    numerical precision differences.

    Args:
        expected_pytorch_result (torch.Tensor): The expected reference tensor
        actual_pytorch_result (torch.Tensor): The actual tensor to compare against the reference
        pcc (float, optional): The minimum PCC threshold for the comparison to pass. Defaults to 0.9999.
                              Values closer to 1.0 indicate stronger correlation.

    Returns:
        tuple: A tuple containing:
            - pcc_passed (bool): True if the PCC check passed, False otherwise
            - pcc_message (str): A message describing the PCC comparison result

    Raises:
        AssertionError: If the tensor shapes don't match or if the PCC is below the specified threshold
    """
    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}"
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    assert pcc_passed, construct_pcc_assert_message(pcc_message, expected_pytorch_result, actual_pytorch_result)
    return pcc_passed, pcc_message


def assert_allclose(
    expected_result: Union[ttnn.Tensor, torch.Tensor],
    actual_result: Union[ttnn.Tensor, torch.Tensor],
    rtol=1e-05,
    atol=1e-08,
):
    r"""
     Assert that two tensors are similar.

     Two tensors are considered close if
     ``
     |actual - expected| \leq atol + rtol \cdot |expected|
     ``

    Args:
         expected_result (Union[ttnn.Tensor, torch.Tensor]): The expected reference tensor
         actual_result (Union[ttnn.Tensor, torch.Tensor]): The actual tensor to compare against the reference
         rtol (float, optional): Relative tolerance. Defaults to 1e-05.
         atol (float, optional): Absolute tolerance. Defaults to 1e-08

     Returns:
         tuple: A tuple containing:
             - allclose_passed (bool): True if allclose check passed, False otherwise
             - allclose_message (str): A message describing comparison result

     Raises:
         AssertionError: If the tensor shapes don't match or if tensors are not close enough according to
                         the aforementioned formula.
    """
    if isinstance(expected_result, ttnn.Tensor):
        expected_result = ttnn.to_torch(expected_result)
    if isinstance(actual_result, ttnn.Tensor):
        actual_result = ttnn.to_torch(actual_result)

    assert list(expected_result.shape) == list(
        actual_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_result.shape)}"
    allclose_passed, allclose_message = comp_allclose(expected_result, actual_result, rtol, atol)
    assert allclose_passed, allclose_message
    return allclose_passed, allclose_message


def assert_with_ulp(
    expected_result: Union[ttnn.Tensor, torch.Tensor],
    actual_result: Union[ttnn.Tensor, torch.Tensor],
    ulp_threshold=10,
    allow_nonfinite=False,
):
    """
    Assert that two tensors are similar within a given distance expressed in Units of Least Precision (ULP)

    The error is measured using the following formula:
    ``
        | expected - actual | / ULP(expected)
    ``

    Where ULP(expected) returns, for each element, the length of a single Unit of Least Precision (ULP).


    Args:
        expected_result (Union[ttnn.Tensor, torch.Tensor]): The expected reference tensor
        actual_result (Union[ttnn.Tensor, torch.Tensor]): The actual tensor to compare against the reference
        ulp_threshold (float, optional): Maximum tolerated ULP distance. Defaults to 10.
        allow_nonfinite (bool, optional): If disabled, any non-finite value (NaN, +inf, -inf) will trigger an assertion. If enabled, differences between non-finite values at the same positions will trigger an assertion.

    Notes:
        The length of a single ULP is measured using the difference between two consecutive floating point numbers.

        ULP should be preferred when errors between `calculated` and `golden` outputs are known to be small (difference < 10s of ULPs).
        This is typically the case for element-wise operations that approximate common numerical functions (e.g. exp, pow, log, ...).

        For more significant differences, where `calculated` and `golden` differ by orders of magnitude, ULPs may be harder to compare
        Indeed, with current definition, on bfloat16:
        - ULP-Delta(4, 0) = 128
        - ULP-Delta(0, 4) = 4.36e+40

        Generally, if the ULP error exceeds the 2**(#mantissa bits) (128-ULP for bfloat16, 8388608 for float32), then it means that both outputs are different by more than an order of magnitude.
        For these cases, functions such as `assert_allclose(golden, calculated, rtol, atol)` should be used instead.

        To measure the accuracy in ULP of operations on bfloat8_b data type, the ttnn bfloat8_b tensor should be either passed directly to the
        function, or converted to bfloat16 beforehand (bfloat16 has the 'same' resolution as bfloat8_b).
        Indeed, ttnn.to_torch() converts bfloat8_b to float32 by default, which would lead to assert_with_ulp() measuring ULP error as if
        data type was computed as float32.

    Returns:
        tuple: A tuple containing:
            - ulp_passed (bool): True if ulp check passed, False otherwise
            - ulp_message (str): A message describing comparison result

    Raises:
        AssertionError: If the tensor shapes don't match or if tensor difference is greater than ulp_threshold.
    """

    def tt_dtype_to_torch_dtype_for_ulp(tt_dtype):
        # By default, ttnn converts ttnn.bfloat8_b to torch.float
        # However, the resolution of a bfloat8_b value is the same as bfloat16
        # (assuming all elements within the block share the same exponent)
        # Thus, for ULP measurement, we convert bfloat8_b to bfloat16 instead of float32
        if tt_dtype == ttnn.bfloat8_b:
            return torch.bfloat16
        if tt_dtype not in tt_dtype_to_torch_dtype:
            raise ValueError(f"Trying to measure ULP on unknown dtype: {tt_dtype}")
        return tt_dtype_to_torch_dtype[tt_dtype]

    if isinstance(expected_result, ttnn.Tensor):
        expected_result = ttnn.to_torch(expected_result, dtype=tt_dtype_to_torch_dtype_for_ulp(expected_result.dtype))
    if isinstance(actual_result, ttnn.Tensor):
        actual_result = ttnn.to_torch(actual_result, dtype=tt_dtype_to_torch_dtype_for_ulp(actual_result.dtype))

    assert list(expected_result.shape) == list(
        actual_result.shape
    ), f"list(expected_result.shape)={list(expected_result.shape)} vs list(actual_result.shape)={list(actual_result.shape)}"

    maximum_meaningful_ulp_thresholds = {
        torch.float64: 2**52,
        torch.float32: 2**23,
        torch.float16: 2**10,
        torch.bfloat16: 2**7,
    }
    maximum_meaningful_ulp_threshold = (
        maximum_meaningful_ulp_thresholds[torch.float32]
        if expected_result.dtype in maximum_meaningful_ulp_thresholds
        else maximum_meaningful_ulp_thresholds[expected_result.dtype]
    )

    if ulp_threshold > maximum_meaningful_ulp_threshold:
        logger.warning(
            f"ULP threshold {ulp_threshold} is greater than the maximum meaningful ULP threshold of {maximum_meaningful_ulp_threshold} for dtype {expected_result.dtype}"
        )

    ulp_passed, ulp_message = comp_ulp(expected_result, actual_result, ulp_threshold, allow_nonfinite)
    assert ulp_passed, ulp_message
    return ulp_passed, ulp_message


def assert_equal(expected_pytorch_result, actual_pytorch_result):
    """
    Assert that two PyTorch tensors are exactly equal.

    This function performs an exact equality comparison between two tensors, checking that
    all corresponding elements are identical. Both tensor shapes and values must match exactly.

    Args:
        expected_pytorch_result (torch.Tensor): The expected reference tensor
        actual_pytorch_result (torch.Tensor): The actual tensor to compare against the reference

    Returns:
        tuple: A tuple containing:
            - equal_passed (bool): True if the tensors are exactly equal, False otherwise
            - equal_message (str): A message describing the equality comparison result

    Raises:
        AssertionError: If the tensor shapes don't match or if the tensors are not exactly equal
    """
    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}"
    equal_passed, equal_message = comp_equal(expected_pytorch_result, actual_pytorch_result)
    assert equal_passed, equal_message
    return equal_passed, equal_message


def comp_relative_frobenius(expected_pytorch_result, actual_pytorch_result):
    """
    Compute the relative Frobenius norm of the difference between two tensors.
    Uses relative Frobenius norm: ||error||_F / ||expected||_F.
    If ||expected||_F == 0, returns the absolute Frobenius error.

    Args:
        expected_pytorch_result (torch.Tensor or ttnn.Tensor): The expected reference tensor.
        actual_pytorch_result (torch.Tensor or ttnn.Tensor): The actual tensor to compare against the reference.

    Returns:
        float: The (relative or absolute) Frobenius norm of the error.
        bool: True if the expected norm is zero, False otherwise.
    """
    if isinstance(expected_pytorch_result, ttnn.Tensor):
        expected_pytorch_result = ttnn.to_torch(expected_pytorch_result)
    if isinstance(actual_pytorch_result, ttnn.Tensor):
        actual_pytorch_result = ttnn.to_torch(actual_pytorch_result)

    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"Shape mismatch: expected {list(expected_pytorch_result.shape)} vs actual {list(actual_pytorch_result.shape)}"

    error = expected_pytorch_result - actual_pytorch_result
    frob_error = torch.norm(error, p="fro")
    frob_expected = torch.norm(expected_pytorch_result, p="fro")

    expected_norm_is_zero = frob_expected == 0
    rel_norm_value = float(frob_error / frob_expected) if not expected_norm_is_zero else float(frob_error)

    return rel_norm_value, expected_norm_is_zero


def assert_relative_frobenius(expected_pytorch_result, actual_pytorch_result, threshold=0.01):
    """
    Assert that the relative Frobenius norm of the difference between two tensors is below a specified threshold.
    Uses relative Frobenius norm: ||error||_F / ||expected||_F. If ||expected||_F == 0, uses absolute Frobenius error.

    Args:
        expected_pytorch_result (torch.Tensor or ttnn.Tensor): The expected reference tensor.
        actual_pytorch_result (torch.Tensor or ttnn.Tensor): The actual tensor to compare against the reference.
        threshold (float): The maximum allowed relative Frobenius norm of the error.

    Returns:
        tuple: A tuple containing:
            - relative_frobenius_passed (bool): True if the relative Frobenius norm is below the threshold, False otherwise
            - relative_frobenius_message (str): A message describing the relative Frobenius norm comparison result

    Raises:
        AssertionError: If the tensor shapes don't match or if the relative Frobenius norm is above the threshold.
    """
    rel_norm_value, expected_norm_is_zero = comp_relative_frobenius(expected_pytorch_result, actual_pytorch_result)

    relative_frobenius_passed = rel_norm_value <= threshold
    relative_frobenius_message = f"Relative Frobenius norm {rel_norm_value} is below threshold {threshold}."
    if not relative_frobenius_passed:
        if expected_norm_is_zero:
            relative_frobenius_message = (
                f"Frobenius norm of expected is 0. Absolute error {rel_norm_value} exceeds threshold {threshold}."
            )
        else:
            relative_frobenius_message = (
                f"Relative Frobenius norm of error {rel_norm_value} exceeds threshold {threshold}."
            )
    assert relative_frobenius_passed, relative_frobenius_message
    return relative_frobenius_passed, relative_frobenius_message


def check_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    if expected_pytorch_result.shape != actual_pytorch_result.shape:
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
        )
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, construct_pcc_assert_message(pcc_message, expected_pytorch_result, actual_pytorch_result)


def check_with_pcc_without_tensor_printout(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    if expected_pytorch_result.shape != actual_pytorch_result.shape:
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
        )
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, pcc_message


def set_slow_dispatch_mode(set_var):
    prev_value = os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)

    if set_var != "" and set_var is not None:
        os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = set_var
        logger.info("Setting slow dispatch mode")
    else:
        logger.info("Setting fast dispatch mode")

    return prev_value


def update_process_id():
    print(f"Debugging PID: {os.getpid()}")
    cwd = os.getcwd()
    launch_json_path = os.path.join(cwd, ".vscode", "launch.json")
    with open(launch_json_path, "r") as f:
        launch_data = json.load(f)

    for config in launch_data.get("configurations", []):
        if config.get("name") == "C++: Attach to Python":
            config["processId"] = str(os.getpid())
            break

    with open(launch_json_path, "w") as f:
        json.dump(launch_data, f, indent=4)


def get_per_core_size_and_num_cores(
    size: int, num_cores_choices: Tuple[int, ...], min_per_core_size: int = 32, max_per_core_size: int = None
) -> Tuple[int, int]:
    if max_per_core_size is None:
        max_per_core_size = size

    for num_cores in num_cores_choices:
        per_core_size = roundup(divup(size, num_cores), 32)  # Divide, round up, then round up to nearest 32
        if per_core_size > min_per_core_size and per_core_size < max_per_core_size:
            # Actual num_cores might be less after we round up to nearest 32
            num_cores_actual = divup(size, per_core_size)  # Divide and round up
            yield per_core_size, num_cores_actual


def start_measuring_time() -> int:
    return time.time_ns()


def stop_measuring_time(start_time) -> int:
    return time.time_ns() - start_time


def maybe_trace(op_func, enable_trace, device):
    if enable_trace:
        # Compile the op
        output = op_func()
        ttnn.synchronize_device(device)

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        output = op_func()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)

        # Execute trace
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.release_trace(device, trace_id)
        ttnn.synchronize_device(device)
    else:
        output = op_func()
    return output


def generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16):
    """
    Generate all possible bfloat16 bit patterns as a test tensor.

    This function creates an exhaustive test dataset by generating all 65,536 (2^16) possible
    bfloat16 bit patterns. This is useful for comprehensive testing of operations across the
    entire bfloat16 value space, including edge cases like infinities, NaNs, and subnormals.

    Args:
        dtype (torch.dtype, optional): The target dtype to cast the bit patterns to.
                                       Defaults to torch.bfloat16.

    Returns:
        torch.Tensor: A tensor of shape (256, 256) containing all possible bfloat16 values,
                     cast to the specified dtype. The tensor is shaped as a square grid for
                     convenient TILE_LAYOUT compatibility (32x32 tile divisibility).

    Notes:
        - The function generates values by iterating through all 16-bit integer patterns
          and reinterpreting them as bfloat16 values.
        - The resulting tensor includes all special values: +/-0, +/-infinity, NaNs,
          subnormals, and all normal values in the bfloat16 range.
        - When dtype is set to a higher precision format (e.g., torch.float32), the bfloat16
          values are promoted without loss of information.

    Example:
        >>> all_bf16 = generate_all_bfloat16_bitpatterns(torch.float32)
        >>> all_bf16.shape
        torch.Size([256, 256])
    """
    # Generate all possible bfloat16 bit patterns (2^16 = 65536 values)
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    bf16_bitpatterns = all_bitpatterns.view(torch.bfloat16)  # Reinterpret as bfloat16

    # Cast to target dtype
    bitpatterns = bf16_bitpatterns.to(dtype)

    # Reshape tensor to 256 x 256 for tile layout compatibility
    bitpatterns = bitpatterns.reshape(256, 256)

    return bitpatterns


def flush_subnormal_values_to_zero(tensor):
    """
    Flush subnormal (denormalized) floating-point values to zero.

    Subnormal numbers are floating-point values smaller than the smallest normalized number.
    Tenstorrent hardware flushes subnormals to zero for performance reasons.
    This function replicates that behavior for testing purposes.

    Args:
        tensor (torch.Tensor): Input tensor with floating-point values.

    Returns:
        torch.Tensor: The input tensor with all subnormal values replaced by zero.
                     The tensor is modified in-place.

    Notes:
        - This function only works for float32 and bfloat16 as they share the same exponent range.
        - For float32 and bfloat16, subnormal values are those where the exponent bits are all zero,
          which corresponds to absolute values less than 2^(-126).
    """
    # Float32 and bfloat16 numbers are subnormal if exponent == 0
    # This corresponds to absolute values < 2^(-126)
    SUBNORMAL_THRESHOLD = 2.0 ** (-126)
    mask = torch.abs(tensor) < SUBNORMAL_THRESHOLD
    tensor[mask] = 0.0
    return tensor

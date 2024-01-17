# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc, set_slow_dispatch_mode
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_eltwise_scalar_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    scalar,
    data_seed,
    dispatch_mode,
    pt_op,
    tt_op,
    device,
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[1]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = pt_op(x, y, alpha=scalar)

        x = ttnn_ops.torch_to_ttnn(x, device, dlayout[0], in_mem_config, dtype[0])
        y = ttnn_ops.torch_to_ttnn(y, device, dlayout[1], in_mem_config, dtype[1])

        tt_result = tt_op(x, y, alpha=scalar)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        set_slow_dispatch_mode(prev_dispatch_mode)
        raise e

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(5, 11, 241, 88), (5, 11, 241, 88)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        -76.0,
        4171614,
        "1",
    ),
    (
        [(6, 12, 161, 76), (6, 12, 161, 76)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        32.75,
        4265074,
        "1",
    ),
    (
        [(4, 5, 236, 192), (4, 5, 236, 192)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        50.75,
        4078621,
        "1",
    ),
    (
        [(5, 4, 11, 202), (5, 4, 11, 202)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        98.5,
        13154222,
        "1",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, scalar, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_add(
    input_shape, dtype, dlayout, in_mem_config, output_mem_config, scalar, data_seed, dispatch_mode, device
):
    run_eltwise_scalar_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        output_mem_config,
        scalar,
        data_seed,
        dispatch_mode,
        torch.add,
        ttnn.add,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, scalar, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_sub(
    input_shape, dtype, dlayout, in_mem_config, output_mem_config, scalar, data_seed, dispatch_mode, device
):
    run_eltwise_scalar_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        output_mem_config,
        scalar,
        data_seed,
        dispatch_mode,
        torch.sub,
        ttnn.sub,
        device,
    )


def run_eltwise_binary_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    dispatch_mode,
    pt_op,
    tt_op,
    device,
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[1]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = pt_op(x, y)

        x = ttnn_ops.torch_to_ttnn(x, device, dlayout[0], in_mem_config, dtype[0])
        y = ttnn_ops.torch_to_ttnn(y, device, dlayout[1], in_mem_config, dtype[1])

        tt_result = tt_op(x, y)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        set_slow_dispatch_mode(prev_dispatch_mode)
        raise e

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, scalar, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_mul(
    input_shape, dtype, dlayout, in_mem_config, output_mem_config, scalar, data_seed, dispatch_mode, device
):
    run_eltwise_binary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        output_mem_config,
        data_seed,
        dispatch_mode,
        torch.mul,
        ttnn.mul,
        device,
    )


def run_eltwise_unary_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    dispatch_mode,
    pt_op,
    tt_op,
    device,
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = pt_op(x)
        x = ttnn_ops.torch_to_ttnn(x, device, dlayout[0], in_mem_config, dtype[0])

        tt_result = tt_op(x)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        set_slow_dispatch_mode(prev_dispatch_mode)
        raise e

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_exp(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, dispatch_mode, device
):
    run_eltwise_unary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        dispatch_mode,
        torch.exp,
        ttnn.exp,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_tanh(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, dispatch_mode, device
):
    run_eltwise_unary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        dispatch_mode,
        torch.tanh,
        ttnn.tanh,
        device,
    )


# torch.nn.functional.gelu(x, approximate=approximate)


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_gelu(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, dispatch_mode, device
):
    run_eltwise_unary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        dispatch_mode,
        torch.nn.functional.gelu,
        ttnn.gelu,
        device,
    )


def run_permute_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    permute_dims,
    data_seed,
    dispatch_mode,
    device,
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.permute(x, permute_dims)
        x = ttnn_ops.torch_to_ttnn(x, device, dlayout[0], in_mem_config, dtype[0])

        tt_result = ttnn.permute(x, permute_dims)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        set_slow_dispatch_mode(prev_dispatch_mode)
        raise e

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_permute(input_shape, dtype, dlayout, in_mem_config, out_mem_config, scalar, data_seed, dispatch_mode, device):
    run_permute_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, (3, 1, 0, 2), data_seed, dispatch_mode, device
    )

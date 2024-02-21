# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import complex_abs as pt_complex_abs
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import complex_abs as tt_complex_abs
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_complex


def run_complex_unary_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device, pt_op, tt_op
):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand_complex(size=input_shape, low=-100, high=100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = pt_op(x_ref)

    tt_result = tt_op(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


def run_complex_binary_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device, pt_op, tt_op
):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand_complex(size=input_shape, low=-100, high=100)
    y = gen_rand_complex(size=input_shape, low=-100, high=100)

    # get ref result
    ref_value = pt_op(x, y)

    tt_result = tt_op(
        x=x,
        y=y,
        device=device,
        dtype=[dtype, dtype],
        layout=[dlayout, dlayout],
        input_mem_config=[in_mem_config, in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (4, 7, 32, 128),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        38346,
    ),
    (
        (4, 6, 160, 64),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        38346,
    ),
    (
        (4, 6, 160, 128),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        38346,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_abs_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_unary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        device,
        pytorch_ops.complex_abs,
        tt_lib_ops.complex_abs,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_conj_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_unary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        device,
        pytorch_ops.complex_conj,
        tt_lib_ops.complex_conj,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_div_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_binary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        device,
        pytorch_ops.complex_div,
        tt_lib_ops.complex_div,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_imag_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_unary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        device,
        pytorch_ops.complex_imag,
        tt_lib_ops.complex_imag,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_mul_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_binary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        device,
        pytorch_ops.complex_mul,
        tt_lib_ops.complex_mul,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_real_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_unary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        device,
        pytorch_ops.complex_real,
        tt_lib_ops.complex_real,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_recip_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_unary_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        device,
        pytorch_ops.complex_recip,
        tt_lib_ops.complex_recip,
    )

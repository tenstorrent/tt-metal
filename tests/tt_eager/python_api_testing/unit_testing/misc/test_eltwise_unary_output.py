# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "mem_configs",
    (
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1),
    ),
)
@pytest.mark.parametrize("scalar", (3.2, -2.0))
def test_unary_mul(input_shapes, mem_configs, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    mem_cfg = mem_configs

    tt_output_tensor_on_device = tt_lib.tensor.mul_unary(input_tensor, scalar, output_mem_config=mem_cfg)
    golden_tensor = torch.mul(in_data, scalar)

    comp_pass = compare_pcc([tt_output_tensor_on_device], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", (3.2, -2.0))
def test_unary_mul_output(input_shapes, device, scalar):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.mul_unary(input_tensor, scalar, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.mul(in_data, scalar)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", (3.2, -2.0))
def test_unary_mul_output_scalar(input_shapes, device, scalar):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.mul_unary(scalar, input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.mul(scalar, in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "mem_configs",
    (
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1),
    ),
)
def test_recip(input_shapes, mem_configs, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device)

    mem_cfg = mem_configs

    tt_output_tensor_on_device = tt_lib.tensor.recip(input_tensor, output_mem_config=mem_cfg)
    golden_tensor = torch.reciprocal(in_data)

    comp_pass = compare_pcc([tt_output_tensor_on_device], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_recip_output(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.recip(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.reciprocal(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", (3.2, -2.0))
def test_unary_div_output(input_shapes, device, scalar):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.div_unary(input_tensor, scalar, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.div(in_data, scalar)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", (3.2, -2.0))
def test_unary_div_output_scalar(input_shapes, device, scalar):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.div_unary(scalar, input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.div(scalar, in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", (3.2, -2.0))
def test_unary_add_output(input_shapes, device, scalar):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.add_unary(input_tensor, scalar, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.add(in_data, scalar)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", (3.2, -2.0))
def test_unary_add_output_scalar(input_shapes, device, scalar):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.add_unary(scalar, input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.add(scalar, in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_gtz_output(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.gtz(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.gt(in_data, 0)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_lez_output(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.lez(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.le(in_data, 0)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_eqz_output(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.eqz(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.eq(in_data, 0)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_log_output(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1e-6, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, 1e-6, 1, device)

    cq_id = 0
    tt_lib.tensor.log(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.log(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_sqrt_output(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 0, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, 0, 1, device)

    cq_id = 0
    tt_lib.tensor.sqrt(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.sqrt(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "unary_op_fn", [[tt_lib.tensor.neg, torch.neg], [tt_lib.tensor.sign, torch.sign], [tt_lib.tensor.tanh, torch.tanh]]
)
def test_unary_ops_output(input_shapes, device, unary_op_fn):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_op = unary_op_fn[0]
    torch_op = unary_op_fn[1]
    tt_op(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch_op(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("exponent", (3, 0.5))
def test_unary_pow(input_shapes, device, exponent):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.pow(input_tensor, exponent, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.pow(in_data, exponent)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("fast_and_approx", (True, False))
def test_unary_exp(input_shapes, device, fast_and_approx):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_lib.tensor.exp(input_tensor, fast_and_approx=fast_and_approx, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.exp(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass

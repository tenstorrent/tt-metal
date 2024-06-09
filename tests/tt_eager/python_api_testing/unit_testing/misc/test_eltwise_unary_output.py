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

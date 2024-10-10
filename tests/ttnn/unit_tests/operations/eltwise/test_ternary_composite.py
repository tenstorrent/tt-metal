# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_val,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_ternary_addcmul_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -80, 80, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -90, 90, device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

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
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_ternary_addcdiv_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.addcdiv(input_tensor1, input_tensor2, input_tensor3, value=value)
    golden_fn = ttnn.get_golden_function(ttnn.addcdiv)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

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
@pytest.mark.parametrize("predicate", [0, 1])
def test_ternary_where_opt_output(input_shapes, predicate, device):
    in_data1, input_tensor1 = data_gen_with_val(input_shapes, device, val=predicate)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -90, 90, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    ttnn.where(input_tensor1, input_tensor2, input_tensor3, output_tensor=output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.where)
    golden_tensor = golden_fn(in_data1.bool(), in_data2, in_data3)

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
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_lerp_overload_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.lerp(input_tensor1, input_tensor2, value)
    golden_fn = ttnn.get_golden_function(ttnn.lerp)
    golden_tensor = golden_fn(in_data1, in_data2, value)

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
def test_lerp_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.lerp(input_tensor1, input_tensor2, input_tensor3)
    golden_fn = ttnn.get_golden_function(ttnn.lerp)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3)

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
@pytest.mark.parametrize("value1", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("value2", [1.0, 5.0, 10.0])
def test_mac_overload_ttnn(input_shapes, value1, value2, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.mac(input_tensor1, value1, value2)
    golden_fn = ttnn.get_golden_function(ttnn.mac)
    golden_tensor = golden_fn(in_data1, value1, value2)

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
def test_mac_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.mac(input_tensor1, input_tensor2, input_tensor3)
    golden_fn = ttnn.get_golden_function(ttnn.mac)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass

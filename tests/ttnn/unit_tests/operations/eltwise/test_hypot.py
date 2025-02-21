# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from models.utility_functions import torch_random
from functools import partial
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_device_hypot_no_bcast(input_shapes, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(input_shapes, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(input_shapes, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
    ),
)
def test_device_hypot_scalar(a_shape, b_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(a_shape, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(b_shape, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([2, 3, 1, 4]), torch.Size([2, 3, 5, 4])),  # ROW_A
        (torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 1, 4])),  # ROW_B
    ),
)
def test_device_hypot_bcast_row(a_shape, b_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(a_shape, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(b_shape, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([2, 3, 5, 1]), torch.Size([2, 3, 5, 4])),  # COL_A
        (torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 5, 1])),  # COL_B
    ),
)
def test_device_hypot_bcast_col(a_shape, b_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(a_shape, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(b_shape, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998


@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 31, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 2, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([2, 3, 128, 1])),
    ),
)
def test_device_hypot_invalid_bcast(a_shape, b_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(a_shape, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(b_shape, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as e:
        output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)
        assert "Broadcasting rule violation" in str(e.value)


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.float32]),
)
def test_device_hypot_sfpu_no_bcast(input_shapes, dtype, device):
    torch.manual_seed(0)
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(input_shapes)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(input_shapes)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    out_tt = ttnn.experimental.hypot(a_tt, b_tt, queue_id=cq_id)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.hypot)
    out_pt = golden_fn(a_pt, b_pt)
    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.9998


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.float32]),
)
def test_device_hypot_sfpu_scalar(a_shape, b_shape, dtype, device):
    torch.manual_seed(0)
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    out_tt = ttnn.experimental.hypot(a_tt, b_tt, queue_id=cq_id)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.hypot)
    out_pt = golden_fn(a_pt, b_pt)
    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.9998


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([2, 3, 1, 4]), torch.Size([2, 3, 5, 4])),  # ROW_A
        (torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 1, 4])),  # ROW_B
    ),
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.float32]),
)
def test_device_hypot_sfpu_bcast_row(a_shape, b_shape, dtype, device):
    torch.manual_seed(0)
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    out_tt = ttnn.experimental.hypot(a_tt, b_tt, queue_id=cq_id)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.hypot)
    out_pt = golden_fn(a_pt, b_pt)
    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.9998


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([2, 3, 5, 1]), torch.Size([2, 3, 5, 4])),  # COL_A
        (torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 5, 1])),  # COL_B
    ),
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.float32]),
)
def test_device_hypot_sfpu_bcast_col(a_shape, b_shape, dtype, device):
    torch.manual_seed(0)
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    out_tt = ttnn.experimental.hypot(a_tt, b_tt, queue_id=cq_id)
    tt_out = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.hypot)
    out_pt = golden_fn(a_pt, b_pt)
    status = ttnn.pearson_correlation_coefficient(out_pt, tt_out)
    assert status >= 0.9998


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 31, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 2, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([2, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.float32]),
)
def test_device_hypot_sfpu_invalid_bcast(a_shape, b_shape, dtype, device):
    torch.manual_seed(0)
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_pt,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    with pytest.raises(RuntimeError) as e:
        output_tensor = ttnn.experimental.hypot(a_tt, b_tt, queue_id=cq_id)
        assert "Broadcasting rule violation" in str(e.value)


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.hypot,
    ],
)
def test_hypot_fp32(device, ttnn_function):
    x_torch = torch.tensor([[2.3454653]], dtype=torch.float32)
    y_torch = torch.tensor([[5.00030171126]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.experimental.hypot(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status

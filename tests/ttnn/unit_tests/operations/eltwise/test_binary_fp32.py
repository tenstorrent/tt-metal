# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.sub,
        ttnn.rsub,
        ttnn.add,
    ],
)
def test_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_sub = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_sub)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.add,
        ttnn.sub,
    ],
)
def test_int32(device, ttnn_function):
    x_torch = torch.tensor([[11, 23, 0, -23, -1, -100]], dtype=torch.int32)
    y_torch = torch.tensor([[78, 99, 34, -33, -1, 100]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_add)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.mul,
    ],
)
def test_mul_fp32(device, ttnn_function):
    x_torch = torch.tensor([[2]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.divide,
    ],
)
# Torch num/ 0 = inf and 0/0  nan; TT num/ 0 = inf and 0/0=nan; in fp32  tile
# Torch num/ 0 = inf and 0/0  nan; TT num/ 0 = inf and 0/0=0; in chained (mul * recip) div op
def test_div_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.00030171126, -3, 16, -5, 14, -12, 0, 0, 1, 15, 0.0, float("inf")]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, -4, -5, 0, 0, 0, 1, 0, 10, 0.0, float("inf")]], dtype=torch.float32)
    # torch out in ttnn TorchTensor([[ 0.500150859355927, -1.000000000000000, -4.000000000000000,  1.000000000000000,                inf,               -inf,                nan,  0.000000000000000,                inf,
    #            1.500000000000000]])
    # tt out in torch TorchTensor([[ 0.500150859355927, -1.000000000000000, -4.000000000000000,  1.000000000000000,                inf,               -inf,                nan,  0.000000000000000,                inf,
    #            1.499999880790710]])
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)

    assert_with_ulp(expected_result=z_torch, actual_result=tt_out, ulp_threshold=0, allow_nonfinite=True)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.divide,
    ],
)
# Test division when input_b is non-zero
def test_div_bf16_nonzero(device, ttnn_function):
    x_torch = torch.tensor(
        [
            [
                1.00030171126,
                -3,
                16,
                -5,
                0,
                15,
            ]
        ],
        dtype=torch.bfloat16,
    )
    y_torch = torch.tensor(
        [
            [
                2,
                3,
                -4,
                -5,
                1,
                10,
            ]
        ],
        dtype=torch.bfloat16,
    )
    # torch out in ttnn TorchTensor([[ 0.500000000000000, -1.000000000000000, -4.000000000000000,  1.000000000000000, 0.000000000000000,  1.500000000000000]],
    #         dtype=torch.bfloat16)
    # tt out in torch TorchTensor([[ 0.500000000000000, -1.000000000000000, -4.000000000000000,  1.000000000000000, 0.000000000000000,  1.500000000000000]],
    #         dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn_function(x_tt, y_tt)  # bf16 runs FPU
    tt_out = ttnn.to_torch(z_tt_div)

    assert_with_ulp(expected_result=z_torch, actual_result=tt_out, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.pow,
    ],
)
def test_pow_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.55, 2.25, -3.6]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, -2.2]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_pow = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_pow)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


def test_squared_sum_fp32_activ(device):
    x_torch = torch.ones([1, 1, 64, 64], dtype=torch.float32)
    y_torch = torch.ones([1, 1, 64, 64], dtype=torch.float32) * 4
    z_torch = torch.square(x_torch + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(x_tt, y_tt, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)])
    tt_out = ttnn.to_torch(z_tt_add)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "activation, standalone, torch_fn",
    [
        (
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
            lambda t: ttnn.softplus(t, beta=1.0, threshold=20.0),
            lambda x: torch.nn.functional.softplus(x, beta=1.0, threshold=20.0),
        ),
        (
            ttnn.UnaryWithParam(ttnn.UnaryOpType.ERF, False),
            lambda t: ttnn.erf(t, fast_and_approximate_mode=False),
            torch.erf,
        ),
    ],
    ids=["softplus", "erf"],
)
def test_add_fp32_activ_matches_standalone(device, activation, standalone, torch_fn):
    # A fused activation on float32 must run the float32 SFPU variant, as the standalone op does: the bf16
    # variant of softplus returns 0 below -5 and the bf16 variant of erf is about 1e4 times less accurate.
    x_torch = torch.linspace(-16.0, 8.0, 1024, dtype=torch.float32).reshape(1, 1, 32, 32)
    y_torch = torch.zeros_like(x_torch)
    z_torch = torch_fn(x_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn.add(x_tt, y_tt, activations=[activation]))
    tt_alone = ttnn.to_torch(standalone(x_tt))

    assert not (tt_out == 0).any()
    assert torch.allclose(z_torch, tt_out, atol=1e-4, rtol=0)
    assert torch.allclose(tt_alone, tt_out, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.add,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 16, 16],
        [1, 3, 320, 384],
    ],
)
def test_add_fp32_input_activ(device, ttnn_function, shape):
    x_torch = torch.ones(shape, dtype=torch.float32) * 2
    y_torch = torch.ones(shape, dtype=torch.float32) * 4
    z_torch = torch.pow(torch.nn.functional.silu(x_torch) + y_torch, 4)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(
        x_tt,
        y_tt,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 4)],
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    )
    tt_out = ttnn.to_torch(z_tt_add)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logaddexp,
    ],
)
def test_logaddexp_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.logaddexp(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logaddexp2,
    ],
)
def test_logaddexp2_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.logaddexp2(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.ldexp,
    ],
)
def test_ldexp_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bias_gelu,
    ],
)
def test_bias_gelu_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.squared_difference,
    ],
)
def test_squared_difference_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2.009, 3.11, 4.22, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
    ],
)
def test_logical_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.509009, 2, 3.33, 4, 0, -11]], dtype=torch.float32)
    y_torch = torch.tensor([[0, 3, 4, 5, 0, -9999]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.eq,
        ttnn.ne,
        ttnn.gt,
        ttnn.ge,
        ttnn.lt,
        ttnn.le,
    ],
)
def test_relational_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.99999999991, 0, 345.1234568999130, -1]], dtype=torch.float32)
    y_torch = torch.tensor([[1.99999999990, 0, 345.1234568999131, -1]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
    ],
)
def test_bitwise(device, ttnn_function):
    x_torch = torch.tensor([[1, 2, 3, 4, 5, 0]], dtype=torch.int32)
    y_torch = torch.tensor([[9, 3, 0, 1, 7, 0]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_xlogy_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.xlogy(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.xlogy)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize("fast_and_approximate_mode", [True, False])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)])
def test_binary_div_edge_case_ttnn(fast_and_approximate_mode, rounding_mode, device, torch_dtype, ttnn_dtype):
    if torch_dtype == torch.bfloat16 and rounding_mode is None and fast_and_approximate_mode is True:
        pytest.skip(
            "Skipping test case due to division by zero not being handled properly in bfloat16 with rounding_mode=None and fast_and_approximate_mode=True"
        )
    in_data1 = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 7.0, 9.75], dtype=torch_dtype)
    in_data2 = torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0, 2.5, -14.25], dtype=torch_dtype)
    input_tensor1 = ttnn.from_torch(
        in_data1, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor2 = ttnn.from_torch(
        in_data2, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = ttnn.div(
        input_tensor1, input_tensor2, fast_and_approximate_mode=fast_and_approximate_mode, rounding_mode=rounding_mode
    )
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, in_data2, rounding_mode)
    output_tensor = ttnn.to_torch(output_tensor)

    if ttnn_dtype == ttnn.bfloat16:
        golden_tensor = torch.where(
            torch.isnan(golden_tensor), torch.tensor(float("inf"), dtype=golden_tensor.dtype), golden_tensor
        )
    assert_with_ulp(expected_result=golden_tensor, actual_result=output_tensor, ulp_threshold=0, allow_nonfinite=True)


@pytest.mark.parametrize(
    "a, b",
    [
        (100.0, 0.0),  # exact answer is 100; exp(100) is not representable
        (89.0, 0.0),  # just past log(FLT_MAX) = 88.72
        (90.0, 89.0),
        (200.0, 199.0),
        (1000.0, 999.0),
        (100.0, 100.0),
        (-100.0, -100.0),  # both exponentials underflow to zero
        (-1000.0, -1000.0),
        (5.0, 3.0),  # inside the currently-working band, as a control
        (0.0, 0.0),
    ],
)
def test_logaddexp_beyond_exp_range_fp32(device, a, b):
    # logaddexp is bounded by its own inputs:
    #     max(a, b) <= logaddexp(a, b) <= max(a, b) + ln 2
    # so a finite pair always has a finite result. Composing it as
    # log(exp(a) + exp(b)) breaks that: exp() saturates above 88.72 and flushes
    # to zero below -87, and the composition returned +/-inf on both sides.
    #
    # The existing coverage draws from [-64, 64] in
    # tests/sweep_framework/sweeps/eltwise/binary/logaddexp, and from [1, 4] in
    # test_logaddexp_fp32 above, so this range was never exercised.
    x_torch = torch.tensor([[a]], dtype=torch.float32)
    y_torch = torch.tensor([[b]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.logaddexp)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn.logaddexp(x_tt, y_tt))

    assert torch.isfinite(tt_out).all(), (
        f"logaddexp({a}, {b}) returned {tt_out.flatten()[0].item()}; "
        f"the exact result is {z_torch.flatten()[0].item()}"
    )
    assert_allclose(tt_out, z_torch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "a, b",
    [
        (89.0, 0.0),  # ijankowskiTT's repro: inf on main, because bfloat16 never reached the kernel
        (100.0, 0.0),
        (128.0, 127.0),
        (-100.0, -100.0),
        (5.0, 3.0),  # inside the working band, as a control
    ],
)
def test_logaddexp_beyond_exp_range_bf16(device, a, b):
    # The kernel is templated on the destination precision and has always had a bfloat16
    # path, but is_binary_sfpu_op gated logaddexp to fp32 only, so ttnn.logaddexp on
    # bfloat16 kept going down the composed exp/add/log route and kept overflowing.
    # The LLK sweep sits below that gate, so it exercised the kernel without exercising
    # the routing. This test goes through the ttnn op, which is what the gate decides.
    x_torch = torch.tensor([[a]], dtype=torch.bfloat16)
    y_torch = torch.tensor([[b]], dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn.logaddexp)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn.logaddexp(x_tt, y_tt))

    got = tt_out.flatten()[0].item()
    want = z_torch.flatten()[0].item()
    assert torch.isfinite(tt_out).all(), f"logaddexp({a}, {b}) on bfloat16 returned {got}; the exact result is {want}"
    assert_allclose(tt_out, z_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "a, b",
    [
        (float("inf"), float("inf")),
        (float("-inf"), float("-inf")),
        (float("inf"), float("-inf")),  # different signs: the difference is well defined
        (float("inf"), 0.0),  # one infinite operand, as a control
        (float("-inf"), 0.0),
    ],
)
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)])
def test_logaddexp_infinities(device, a, b, torch_dtype, ttnn_dtype):
    # max(a, b) + log1p(exp(-|a - b|)) needs matching infinities handled
    # separately: inf - inf is NaN, and the NaN then swallows the result. The
    # composed form it replaces returns +/-inf on these two points, so leaving them
    # out would trade an overflow bug for a NaN one.
    #
    # torch.logaddexp is the reference here: logaddexp(inf, inf) is inf and
    # logaddexp(-inf, -inf) is -inf.
    x_torch = torch.tensor([[a]], dtype=torch_dtype)
    y_torch = torch.tensor([[b]], dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.logaddexp)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn.logaddexp(x_tt, y_tt))

    # Compared as scalars on purpose: torch.equal also compares shape, and would report
    # False rather than raise if to_torch ever came back padded. == is exact here, which
    # is what these five points need -- an allclose against +/-inf says nothing.
    got = tt_out.flatten()[0].item()
    want = z_torch.flatten()[0].item()
    assert got == want, f"logaddexp({a}, {b}) returned {got}; the exact result is {want}"


@pytest.mark.parametrize(
    "a, b",
    [
        (200.0, 0.0),  # exact answer is 200; 2**200 is not representable
        (128.0, 0.0),  # just past log2(FLT_MAX) = 128
        (129.0, 128.0),
        (200.0, 199.0),
        (1000.0, 999.0),
        (127.0, 127.0),  # 2**127 + 2**127 = 2**128 overflows; the answer is 128
        (-150.0, -150.0),  # both powers flush to zero; the answer is -149
        (-1000.0, -1000.0),
        (5.0, 3.0),  # inside the currently-working band, as a control
        (0.0, 0.0),
    ],
)
def test_logaddexp2_beyond_exp2_range_fp32(device, a, b):
    # logaddexp2 is bounded by its own inputs, exactly like logaddexp:
    #     max(a, b) <= logaddexp2(a, b) <= max(a, b) + 1
    # so a finite pair always has a finite result. Composing it as
    # log2(2**a + 2**b) breaks that at the base-2 thresholds: 2**x saturates
    # above 128 and flushes to zero below -149, and the composition returned
    # +/-inf on both sides.
    #
    # The existing coverage draws from [-60, 100] in
    # tests/sweep_framework/sweeps/eltwise/binary/logaddexp2, and from [1, 4] in
    # test_logaddexp2_fp32 above, so this range was never exercised.
    x_torch = torch.tensor([[a]], dtype=torch.float32)
    y_torch = torch.tensor([[b]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.logaddexp2)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn.logaddexp2(x_tt, y_tt))

    assert torch.isfinite(tt_out).all(), (
        f"logaddexp2({a}, {b}) returned {tt_out.flatten()[0].item()}; "
        f"the exact result is {z_torch.flatten()[0].item()}"
    )
    assert_allclose(tt_out, z_torch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "a, b",
    [
        (128.0, 0.0),  # inf on main: bfloat16 shares the base-2 threshold, since log2(BF16_MAX) = 128
        (200.0, 0.0),
        (256.0, 255.0),
        (-150.0, -150.0),
        (5.0, 3.0),  # inside the working band, as a control
    ],
)
def test_logaddexp2_beyond_exp2_range_bf16(device, a, b):
    # Same routing question as test_logaddexp_beyond_exp_range_bf16: the kernel has a
    # bfloat16 path, and this test goes through the ttnn op so that it is the
    # is_binary_sfpu_op gate, not only the kernel, that is exercised.
    x_torch = torch.tensor([[a]], dtype=torch.bfloat16)
    y_torch = torch.tensor([[b]], dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn.logaddexp2)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn.logaddexp2(x_tt, y_tt))

    got = tt_out.flatten()[0].item()
    want = z_torch.flatten()[0].item()
    assert torch.isfinite(tt_out).all(), f"logaddexp2({a}, {b}) on bfloat16 returned {got}; the exact result is {want}"
    assert_allclose(tt_out, z_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "a, b",
    [
        (float("inf"), float("inf")),
        (float("-inf"), float("-inf")),
        (float("inf"), float("-inf")),  # different signs: the difference is well defined
        (float("inf"), 0.0),  # one infinite operand, as a control
        (float("-inf"), 0.0),
    ],
)
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)])
def test_logaddexp2_infinities(device, a, b, torch_dtype, ttnn_dtype):
    # max(a, b) + log2(1 + 2**-|a - b|) needs matching infinities handled separately
    # for the same reason as logaddexp: inf - inf is NaN, and the NaN then swallows the
    # result. The kernel classifies infinity from the exponent/mantissa fields and
    # requires identical bit patterns, exactly as ckernel_sfpu_logaddexp.h does.
    #
    # torch.logaddexp2 is the reference here: logaddexp2(inf, inf) is inf and
    # logaddexp2(-inf, -inf) is -inf.
    x_torch = torch.tensor([[a]], dtype=torch_dtype)
    y_torch = torch.tensor([[b]], dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.logaddexp2)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn.logaddexp2(x_tt, y_tt))

    got = tt_out.flatten()[0].item()
    want = z_torch.flatten()[0].item()
    assert got == want, f"logaddexp2({a}, {b}) returned {got}; the exact result is {want}"


# ---------------------------------------------------------------------------------------------
# logaddexp / logaddexp2: IEEE special values, the scalar and in-place APIs, and broadcasting.
#
# Every IEEE case below is checked on both operations and both float dtypes, in both operand
# orders, against the PyTorch reference:
#     (+inf, +inf) -> +inf    (-inf, -inf) -> -inf    (+inf, -inf) -> +inf
#     (+inf, x)    -> +inf    (-inf, x)    -> x       NaN in either operand -> NaN
# ---------------------------------------------------------------------------------------------

_INF = float("inf")
_NAN = float("nan")

_LOGADDEXP_OPS = [ttnn.logaddexp, ttnn.logaddexp2]
_LOGADDEXP_INPLACE_OPS = [(ttnn.logaddexp, ttnn.logaddexp_), (ttnn.logaddexp2, ttnn.logaddexp2_)]
_LOGADDEXP_DTYPES = [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)]

# (rtol, atol) for finite results. fp32 uses the tolerance of the fp32 overflow tests above,
# bf16 the one of the bf16 overflow tests: one ULP of bf16 is 1/128 of the magnitude.
_LOGADDEXP_TOLERANCE = {torch.float32: (1e-5, 1e-5), torch.bfloat16: (1e-2, 1e-2)}

_LOGADDEXP_SPECIAL_PAIRS = [
    (_INF, _INF),
    (-_INF, -_INF),
    (_INF, -_INF),
    (-_INF, _INF),
    (_INF, 2.5),
    (2.5, _INF),
    (_INF, -150.0),
    (-150.0, _INF),
    (-_INF, 2.5),
    (2.5, -_INF),
    (-_INF, 150.0),
    (150.0, -_INF),
    (-150.0, -_INF),
    (-_INF, -150.0),
    (_NAN, 1.0),
    (1.0, _NAN),
    (_NAN, _NAN),
    (_NAN, _INF),
    (_INF, _NAN),
    (-_INF, _NAN),
    (_NAN, -_INF),
    # A negative NaN: max() orders a NaN by its sign, so this is the case where the NaN
    # would lose to the finite operand if the kernel did not propagate it explicitly.
    (-_NAN, 1.0),
    (1.0, -_NAN),
]

_LOGADDEXP_FINITE_PAIRS = [
    (100.0, 0.0),  # past the exp() overflow of the composed form
    (200.0, -200.0),
    (1000.0, 999.0),
    (-1000.0, -1000.0),  # both exponentials underflow in the composed form
    (5.0, 3.0),
    (1.0, 1.0),
    (-0.5, 0.25),
    (0.0, 0.0),
]


def _assert_logaddexp_matches(got, want, torch_dtype, what):
    got = got.to(torch.float32).flatten()
    want = want.to(torch.float32).flatten()
    assert got.shape == want.shape, f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"

    nan_want = torch.isnan(want)
    if torch_dtype == torch.bfloat16:
        # A NaN result leaves a bfloat16 tensor as an infinity: the bf16 pack cannot hold it.
        # test_binary_div_edge_case_ttnn records the same thing for div. What is asserted here
        # is that the NaN did not turn into a finite value.
        assert (~torch.isfinite(got[nan_want])).all(), f"{what}: NaN inputs gave finite {got[nan_want].tolist()}"
    else:
        assert torch.isnan(got[nan_want]).all(), f"{what}: NaN inputs gave {got[nan_want].tolist()}"

    inf_want = torch.isinf(want)
    assert torch.equal(
        got[inf_want], want[inf_want]
    ), f"{what}: expected {want[inf_want].tolist()}, got {got[inf_want].tolist()}"

    finite = ~(nan_want | inf_want)
    assert torch.isfinite(got[finite]).all(), f"{what}: finite reference, non-finite result {got[finite].tolist()}"
    rtol, atol = _LOGADDEXP_TOLERANCE[torch_dtype]
    torch.testing.assert_close(got[finite], want[finite], rtol=rtol, atol=atol, msg=what)


@pytest.mark.parametrize("ttnn_function", _LOGADDEXP_OPS)
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", _LOGADDEXP_DTYPES)
def test_logaddexp_ops_special_values(device, ttnn_function, torch_dtype, ttnn_dtype):
    pairs = _LOGADDEXP_SPECIAL_PAIRS + _LOGADDEXP_FINITE_PAIRS
    x_torch = torch.tensor([[a for a, _ in pairs]], dtype=torch_dtype)
    y_torch = torch.tensor([[b for _, b in pairs]], dtype=torch_dtype)
    z_torch = ttnn.get_golden_function(ttnn_function)(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn_function(x_tt, y_tt))

    _assert_logaddexp_matches(tt_out, z_torch, torch_dtype, f"{ttnn_function.__name__} {torch_dtype}")


_LOGADDEXP_TENSOR_VALUES = [_INF, -_INF, _NAN, 1000.0, 150.0, 2.5, 0.0, -0.5, -150.0, -1000.0]


@pytest.mark.parametrize("ttnn_function, inplace_function", _LOGADDEXP_INPLACE_OPS)
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", _LOGADDEXP_DTYPES)
@pytest.mark.parametrize("scalar", [0.5, 150.0, -150.0, _INF, -_INF, _NAN])
@pytest.mark.parametrize("api", ["tensor_scalar", "inplace_scalar"])
def test_logaddexp_ops_scalar_api(device, ttnn_function, inplace_function, torch_dtype, ttnn_dtype, scalar, api):
    x_torch = torch.tensor([_LOGADDEXP_TENSOR_VALUES], dtype=torch_dtype)
    z_torch = ttnn.get_golden_function(ttnn_function)(x_torch, torch.full_like(x_torch, scalar))

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if api == "tensor_scalar":
        tt_out = ttnn.to_torch(ttnn_function(x_tt, scalar))
    else:
        inplace_function(x_tt, scalar)
        tt_out = ttnn.to_torch(x_tt)

    _assert_logaddexp_matches(
        tt_out, z_torch, torch_dtype, f"{api} {ttnn_function.__name__}(x, {scalar}) {torch_dtype}"
    )


@pytest.mark.parametrize("ttnn_function, inplace_function", _LOGADDEXP_INPLACE_OPS)
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", _LOGADDEXP_DTYPES)
def test_logaddexp_ops_inplace_tensor(device, ttnn_function, inplace_function, torch_dtype, ttnn_dtype):
    pairs = _LOGADDEXP_SPECIAL_PAIRS + _LOGADDEXP_FINITE_PAIRS
    x_torch = torch.tensor([[a for a, _ in pairs]], dtype=torch_dtype)
    y_torch = torch.tensor([[b for _, b in pairs]], dtype=torch_dtype)
    z_torch = ttnn.get_golden_function(ttnn_function)(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    inplace_function(x_tt, y_tt)

    _assert_logaddexp_matches(ttnn.to_torch(x_tt), z_torch, torch_dtype, f"{inplace_function.__name__} {torch_dtype}")


@pytest.mark.parametrize("ttnn_function", _LOGADDEXP_OPS)
@pytest.mark.parametrize("torch_dtype, ttnn_dtype", _LOGADDEXP_DTYPES)
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((1, 3, 32, 32), (1, 3, 32, 32)),  # no broadcast, as a control
        ((1, 3, 32, 32), (1, 1, 32, 32)),
        ((2, 3, 16, 16), (2, 3, 1, 16)),
        ((2, 3, 16, 16), (1, 3, 16, 1)),
    ],
)
def test_logaddexp_ops_broadcast(device, ttnn_function, torch_dtype, ttnn_dtype, shape_a, shape_b):
    # Finite operands only, drawn wide enough that most pairs are past the 88.7 (logaddexp)
    # and 128 (logaddexp2) thresholds where the composed form overflowed, with the rest in the
    # band where the correction term matters. Every result must be finite and match torch.
    torch.manual_seed(0)
    x_torch = (torch.rand(shape_a) * 600.0 - 300.0).to(torch_dtype)
    y_torch = (torch.rand(shape_b) * 600.0 - 300.0).to(torch_dtype)
    z_torch = ttnn.get_golden_function(ttnn_function)(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.to_torch(ttnn_function(x_tt, y_tt))

    assert list(tt_out.shape) == list(z_torch.shape)
    _assert_logaddexp_matches(
        tt_out, z_torch, torch_dtype, f"{ttnn_function.__name__} {shape_a} x {shape_b} {torch_dtype}"
    )


@pytest.mark.parametrize("ttnn_function", _LOGADDEXP_OPS)
@pytest.mark.parametrize("ttnn_dtype, pcc", [(ttnn.bfloat8_b, 0.999), (ttnn.bfloat4_b, 0.97)])
def test_logaddexp_ops_block_float(device, ttnn_function, ttnn_dtype, pcc):
    # bfloat8_b and bfloat4_b take the same fused kernel as bfloat16. Before they were gated
    # in they stayed on the composed exp/add/log route, whose exp() overflowed at these
    # magnitudes and returned inf. The goldens use the dequantized inputs and the PCC bounds
    # of test_bf4b_bf8b in test_binary_bcast.py; finiteness is what the fix is about.
    torch.manual_seed(0)
    x_torch = (torch.rand((1, 1, 32, 32)) * 600.0 - 300.0).to(torch.bfloat16)
    y_torch = (torch.rand((1, 1, 32, 32)) * 600.0 - 300.0).to(torch.bfloat16)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_torch = ttnn.get_golden_function(ttnn_function)(ttnn.to_torch(x_tt), ttnn.to_torch(y_tt))

    tt_out = ttnn.to_torch(ttnn_function(x_tt, y_tt))

    assert torch.isfinite(tt_out).all(), f"{ttnn_function.__name__} on {ttnn_dtype} returned a non-finite value"
    assert ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= pcc

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib as ttl
from functools import partial

import ttnn._ttnn
import ttnn.operations
import ttnn.operations.matmul
import ttnn.operations.reduction
from models.helper_funcs import Linear as tt_Linear
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, ttl_complex_2_torch_complex
from models.demos.metal_BERT_large_11.tt import custom_matmuls
from tests.ttnn.utils_for_testing import assert_with_pcc


def setup_tt_tensor(x, device, layout, input_mem_config, dtype):
    if input_mem_config is None:
        device = None

    return torch2tt_tensor(x, device, layout, input_mem_config, dtype)


# pcie slot arg will eventually be fully deprecated in favour of pytest uplift
# and passing device from fixture
def setup_host_and_device(func):
    def wrap(*args, device, **kwargs):
        output = func(*args, device=device, **kwargs)
        ttnn.device.DeallocateBuffers(device)
        return output

    return wrap


################################################
################## Helper-Funcs ################
################################################


@setup_host_and_device
def linear(
    x,
    weight,
    bias=None,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    tt_weight = setup_tt_tensor(weight, device, layout[1], input_mem_config[1], dtype[1])
    tt_bias = None

    if bias is not None:
        tt_bias = setup_tt_tensor(bias, device, layout[2], input_mem_config[2], dtype[2])

    _, __, out_features, in_features = tt_weight.shape.with_tile_padding()
    tt_linear = tt_Linear(in_features, out_features, tt_weight, tt_bias)

    t1 = tt_linear(t0)
    return tt2torch_tensor(t1)


################################################
#################### TT-LIB ####################
################################################
@setup_host_and_device
def copy(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.copy(t0, t1)

    return tt2torch_tensor(t2)


@setup_host_and_device
def clone(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.clone(t0, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def typecast(
    x,
    *args,
    device,
    tt_input_dtype,
    tt_output_dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], tt_input_dtype[0])

    # Copy op kernel typecast - yet to migrate to TTNN
    t1 = ttnn.experimental.typecast(t0, tt_output_dtype[0], memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def concat(
    x,
    y,
    *args,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    xtt = (t0, t1)
    t2 = ttnn.concat(xtt, dim, memory_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def move(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    dummy_tensor = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    # Free up dummy tensor from memory to make available to move
    dummy_tensor.deallocate()
    t1 = ttnn.move(t0, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_exp(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.exp(t0, fast_and_approximate_mode=fast_and_approx, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_erf(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.erf(t0, fast_and_approximate_mode=fast_and_approx, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_erfc(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.erfc(t0, fast_and_approximate_mode=fast_and_approx, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_hardtanh(
    x,
    *args,
    low,
    high,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardtanh(t0, min=low, max=high, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_leaky_relu(
    x,
    *args,
    negative_slope,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.leaky_relu(t0, negative_slope, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_elu(
    x,
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.elu(t0, alpha, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_gelu(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.gelu(t0, fast_and_approximate_mode=fast_and_approx, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_rsqrt(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    if t0.layout == ttnn.TILE_LAYOUT:
        t1 = ttnn.rsqrt(t0, fast_and_approximate_mode=fast_and_approx, memory_config=output_mem_config)
    else:
        # this case is for test_eltwise_rsqrt_in_depth.py with shape (3, 11, 92, 100) RM
        # either use this format or move the test to non-working as ttnn does not use run_with_autoformat
        input_shape = t0.shape
        t0 = t0.cpu().pad_to_tile(0)
        t0 = t0.to(ttnn.TILE_LAYOUT)
        t0 = t0.to(device)
        t1 = ttnn.rsqrt(t0, fast_and_approximate_mode=fast_and_approx, memory_config=output_mem_config)
        t1 = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(input_shape)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_prelu(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    weight = kwargs["weight"]
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.prelu(t0, weight, memory_config=output_mem_config)
    return tt2torch_tensor(t1)


# stats ops
@setup_host_and_device
def std_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.std_hw(t0)

    output = tt2torch_tensor(t1)
    output = output.max(2, True)[0].max(3, True)[0]

    return output


@setup_host_and_device
def var_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.var_hw(t0)

    output = tt2torch_tensor(t1)
    output = output.max(2, True)[0].max(3, True)[0]

    return output


@setup_host_and_device
def mean_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.mean(t0, [2, 3], memory_config=output_mem_config)
    output = tt2torch_tensor(t1)
    output = output[:, :, 0, 0]

    return output


@setup_host_and_device
def eltwise_polyval(
    x,
    *args,
    coeffs,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.polyval(t0, coeffs, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_mac(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.mac(t0, t1, t2, memory_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_addcmul(
    x,
    y,
    z,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.addcmul(t0, t1, t2, value=scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_addcdiv(
    x,
    y,
    z,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.addcdiv(t0, t1, t2, value=scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def unary_div_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t3 = ttnn.div_bw(t0, t1, scalar, output_mem_config=output_mem_config)[0]

    return tt2torch_tensor(t3)


@setup_host_and_device
def div_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.div_bw(t0, t1, t2, output_mem_config=output_mem_config)

    return [tt2torch_tensor(t3[0]), tt2torch_tensor(t3[1])]


@setup_host_and_device
def unary_mul_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t3 = ttnn.mul_bw(t0, t1, scalar, output_mem_config)[0]

    return tt2torch_tensor(t3)


@setup_host_and_device
def unary_assign_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t3 = ttnn.assign_bw(t0, t1, output_mem_config)[0]
    return tt2torch_tensor(t3)


@setup_host_and_device
def binary_assign_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.assign_bw(t0, t1, t2, output_mem_config)

    return tt2torch_tensor(t3[0])


@setup_host_and_device
def eltwise_logical_and_(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t3 = ttnn.logical_and_(t0, t1)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_lerp_binary(
    x,
    y,
    *args,
    weight,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.lerp(t0, t1, weight, memory_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_softplus(
    x,
    *args,
    beta,
    threshold,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.softplus(t0, beta=beta, threshold=threshold, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def conv(
    x,
    y,
    conv_params,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    # t2 = ttnn.experimental.tensor.conv(t0, t1, conv_params, 0, 0, 0, 0, 0, conv_params[0])
    t2 = ttnn.conv2d(t0, t1, conv_params, 0, 0, 0, 0, 0, conv_params[0])

    return tt2torch_tensor(t2)


@setup_host_and_device
def layernorm_noweights(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.layer_norm(t0, epsilon=1e-5, weight=None, bias=None, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def add_layernorm_noweights(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.layer_norm(
        t0, residual_input_tensor=t1, epsilon=1e-5, weight=None, bias=None, memory_config=output_mem_config
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def layernorm(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[1] == ttnn.TILE_LAYOUT:
        y = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    if layout[2] == ttnn.TILE_LAYOUT:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.layer_norm(t0, epsilon=1e-5, weight=t1, bias=t2, memory_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def add_layernorm(
    x,
    y,
    z,
    w,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_tt_tensor(w, device, layout[2], input_mem_config[2], dtype[2])
    t4 = ttnn.layer_norm(
        t0, residual_input_tensor=t1, epsilon=1e-5, weight=t2, bias=t3, memory_config=output_mem_config
    )

    return tt2torch_tensor(t4)


@setup_host_and_device
def eltwise_lerp_ternary(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.lerp(t0, t1, t2, memory_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_subalpha(
    x,
    y,
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.subalpha(t0, t1, alpha, memory_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_celu(
    x,
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t2 = ttnn.celu(t0, alpha=alpha, memory_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_logit(x, *args, eps, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.logit(t0, eps=eps, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_polygamma(x, *args, k, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.polygamma(t0, k=k, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_div(
    x,
    y,
    *args,
    accurate_mode,
    round_mode,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.div(t0, t1, accurate_mode=accurate_mode, round_mode=round_mode, memory_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_fmod(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.fmod(t0, t1, memory_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_floor_div(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.floor_div(t0, t1, memory_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_rfloor_div(
    x,
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.rdiv(t0, value, round_mode="floor", memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_round(
    x,
    *args,
    decimals,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.round(t0, decimals=decimals, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_unary_rdiv_trunc(
    x,
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.rdiv(t0, value, round_mode="trunc", memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_bitwise_xor(
    x,
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.bitwise_xor(t0, value, memory_config=output_mem_config, queue_id=0)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_bitwise_not(
    x,
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.bitwise_not(t0, value, memory_config=output_mem_config, queue_id=0)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_right_shift(
    x,
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.bitwise_right_shift(t0, value, memory_config=output_mem_config, queue_id=0)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_left_shift(
    x,
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.bitwise_left_shift(t0, value, memory_config=output_mem_config, queue_id=0)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_heaviside(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.heaviside(t0, scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_unary_ne(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.ne(t0, scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def repeat_interleave(
    x,
    *args,
    repeat,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.repeat_interleave(t0, repeat, dim, memory_config=output_mem_config)
    output_tensor = ttnn.from_device(t1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)
    return output_tensor


@setup_host_and_device
def repeat(
    x,
    *args,
    repeat,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.repeat(t0, ttnn.Shape(repeat), memory_config=output_mem_config)
    output_tensor = ttnn.to_torch(t1)
    return output_tensor


@setup_host_and_device
def eltwise_isclose(
    x,
    y,
    *args,
    rtol,
    atol,
    equal_nan,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.isclose(t0, t1, rtol=rtol, atol=atol, equal_nan=equal_nan, memory_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_unary_gt(
    x,
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.gt(t0, value, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_unary_lt(
    x,
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.lt(t0, value, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def full_like(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.full_like(t0, scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def ones(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = ttnn.ones(
        x.shape,
        layout=layout[0],
        device=device if input_mem_config[0] is not None else None,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def zeros(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = ttnn.zeros(
        x.shape,
        layout=layout[0],
        device=device if input_mem_config[0] is not None else None,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def triu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    tx = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    diag = kwargs.get("diag", 0)
    t1 = ttnn.triu(tx, diagonal=diag, memory_config=output_mem_config)
    return tt2torch_tensor(t1)


@setup_host_and_device
def tril(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    tx = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    diag = kwargs.get("diag", 0)
    t1 = ttnn.tril(tx, diagonal=diag, memory_config=output_mem_config)
    return tt2torch_tensor(t1)


@setup_host_and_device
def empty(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = ttnn.empty(
        x.shape,
        dtype,
        layout[0],
        device if input_mem_config[0] is not None else None,
        output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def full(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t1 = ttnn.full(
        x.shape,
        scalar,
        layout=layout[0],
        device=device if input_mem_config[0] is not None else None,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def fill_rm(
    x,
    *args,
    hOnes,
    wOnes,
    val_hi,
    val_lo,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.fill_rm(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        hOnes,
        wOnes,
        t0,
        val_hi,
        val_lo,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def fill_ones_rm(
    x,
    *args,
    hOnes,
    wOnes,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.fill_ones_rm(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        hOnes,
        wOnes,
        t0,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def arange(
    x,
    *args,
    start,
    end,
    step=1,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t1 = ttnn.arange(
        start,
        end,
        step,
        device=device if input_mem_config[0] is not None else None,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def prod(
    x,
    *args,
    all_dimensions,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.prod(t0, all_dimensions, dim, memory_config=output_mem_config)
    output_tensor = ttnn.from_device(t1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)
    if all_dimensions:
        return output_tensor[:1, :1, :1, :1]
    else:
        return output_tensor


@setup_host_and_device
def clip(
    x,
    *args,
    low,
    high,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.clip(t0, low, high, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def where(x, y, z, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.where(t0, t1, t2, memory_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def where_optional(x, y, z, out, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_tt_tensor(out, device, layout[3], input_mem_config[3], dtype[3])
    cq_id = 0
    ttnn.where(t0, t1, t2, output_tensor=t3, queue_id=cq_id)

    return tt2torch_tensor(t3)


@setup_host_and_device
def where_scalar_optional(
    x, out, device, dtype, layout, input_mem_config, output_mem_config, scalar_true, scalar_false, **kwargs
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t3 = setup_tt_tensor(out, device, layout[1], input_mem_config[1], dtype[1])
    cq_id = 0
    ttnn.where(t0, scalar_true, scalar_false, output_tensor=t3, queue_id=cq_id)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_div_unary(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.multiply(t0, 1 / scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_unary_div(
    x,
    *args,
    scalar,
    accurate_mode,
    round_mode,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.div(t0, scalar, accurate_mode=accurate_mode, round_mode=round_mode, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_mul_unary(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.multiply(t0, scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_sub_unary(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.subtract(t0, scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_add_unary(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.add(t0, scalar, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def bcast_add_h(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)
    t2 = ttnn.add(
        t0,
        t1,
        memory_config=output_mem_config,
    )
    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_add_w(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)
    t2 = ttnn.add(
        t0,
        t1,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_add_hw(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)

    t2 = ttnn.add(
        t0,
        t1,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_sub_h(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)

    t2 = ttnn.subtract(
        t0,
        t1,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_sub_w(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)

    t2 = ttnn.subtract(
        t0,
        t1,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_sub_hw(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)

    t2 = ttnn.subtract(
        t0,
        t1,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_mul_h(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)

    t2 = ttnn.multiply(
        t0,
        t1,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_mul_w(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)

    t2 = ttnn.multiply(
        t0,
        t1,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_mul_hw(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    t1 = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT)

    t2 = ttnn.multiply(
        t0,
        t1,
        memory_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def reduce_sum_h(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sum(t0, 2, memory_config=output_mem_config)
    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :]


@setup_host_and_device
def reduce_sum_w(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sum(t0, 3, memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :, :1]


@setup_host_and_device
def reduce_sum_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sum(t0, [2, 3], memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :1]


@setup_host_and_device
def reduce_max_h(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.max(t0, 2, memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :]


@setup_host_and_device
def reduce_max_w(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.max(t0, 3, memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1]


@setup_host_and_device
def reduce_max_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.max(t0, [2, 3], memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :1]


@setup_host_and_device
def reduce_min_h(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.min(t0, 2, memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :]


@setup_host_and_device
def reduce_min_w(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.min(t0, 3, memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1]


@setup_host_and_device
def reduce_min_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.min(t0, [2, 3], memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :1]


@setup_host_and_device
def sum(x, *args, dim, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    assert dim >= 0 and dim <= 3
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sum(t0, dim, memory_config=output_mem_config)

    output = tt2torch_tensor(t1)

    if dim == 2:
        output = output[:, :, :1, :]
    elif dim == 3:
        output = output[:, :, :, :1]
    return output


@setup_host_and_device
def permute(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    permute_dims,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.permute(t0, permute_dims, memory_config=output_mem_config)
    output_tensor = ttnn.from_device(t1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)
    return output_tensor


@setup_host_and_device
def reshape(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    memory_config,
    reshape_dims,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.reshape_on_device(t0, *reshape_dims, memory_config=memory_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def split_last_dim_two_chunks_tiled(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.split(t0, 2, 3, memory_config=output_mem_config)

    output0 = ttnn.to_torch(t1[0])
    output1 = ttnn.to_torch(t1[1])

    return [output0, output1]


@setup_host_and_device
def tilize(x, *args, device, dtype, layout, input_mem_config, output_mem_config, use_multicore, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.tilize(t0, memory_config=output_mem_config, use_multicore=use_multicore)

    return t1.cpu().to_torch()


@setup_host_and_device
def tilize_with_zero_padding(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.tilize_with_zero_padding(t0, memory_config=output_mem_config)

    return t1.cpu().to_torch()


@setup_host_and_device
def tilize_with_val_padding(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_shape,
    pad_value,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.tilize_with_val_padding(
        t0,
        output_tensor_shape,
        pad_value,
        memory_config=output_mem_config,
    )

    return t1.cpu().to_torch()


@setup_host_and_device
def untilize(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if input_mem_config[0] is None:
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype[0],
            ttnn.TILE_LAYOUT,
        )
    else:
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype[0],
            ttnn.TILE_LAYOUT,
            device,
            input_mem_config[0],
        )

    t1 = ttnn.untilize(t0, memory_config=output_mem_config)
    return t1.cpu().to_torch()


@setup_host_and_device
def untilize_with_unpadding(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_end,
    **kwargs,
):
    if input_mem_config[0] is None:
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype[0],
            ttnn.TILE_LAYOUT,
        )
    else:
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype[0],
            ttnn.TILE_LAYOUT,
            device,
            input_mem_config[0],
        )

    t1 = ttnn.untilize_with_unpadding(t0, output_tensor_end=output_tensor_end, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_typecast(
    x,
    *args,
    device,
    tt_input_dtype,
    tt_output_dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], tt_input_dtype[0])

    # Currently ttnn.typecast has 2 variations of unary kernel typecast - with and without input dtype
    # t1 = ttnn.typecast(t0, tt_output_dtype[0], memory_config=output_mem_config)

    t1 = ttnn.typecast(t0, tt_input_dtype[0], tt_output_dtype[0], memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_rpow(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    factor = kwargs["factor"]
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.rpow(t0, factor, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_pow(
    x,
    *args,
    exponent,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.pow(t0, exponent, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_bias_gelu_unary(x, *args, bias, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.bias_gelu(t0, bias, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


def make_unary_op(ttl_tensor_unop):
    @setup_host_and_device
    def unary_op(
        x,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        output_mem_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        t1 = ttl_tensor_unop(t0, output_mem_config=output_mem_config)

        return tt2torch_tensor(t1)

    return unary_op


def make_ttnn_unary_op(ttl_tensor_unop):
    @setup_host_and_device
    def unary_op(
        x,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        output_mem_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        t1 = ttl_tensor_unop(t0, memory_config=output_mem_config)

        return tt2torch_tensor(t1)

    return unary_op


@setup_host_and_device
def transpose(
    x,
    *args,
    dim0,
    dim1,
    device,
    dtype,
    layout,
    input_mem_config,
    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transpose(t0, dim0, dim1, memory_config=memory_config)
    output_tensor = ttnn.from_device(t1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)
    return output_tensor


def make_unary_op_optional_output(ttl_tensor_unop):
    @setup_host_and_device
    def unary_op_optional_output(
        x,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        cq_id = 0

        t2 = ttl_tensor_unop(t0, queue_id=cq_id)

        return tt2torch_tensor(t2)

    return unary_op_optional_output


def make_unary_op_composite_ttnn(ttl_tensor_unop):
    @setup_host_and_device
    def unary_op_optional_output(
        x,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

        t2 = ttl_tensor_unop(t0)

        return tt2torch_tensor(t2)

    return unary_op_optional_output


def make_unary_op_optional_output_with_scalar(ttl_tensor_unop):
    @setup_host_and_device
    def unary_op_optional_output_with_scalar(
        x,
        *args,
        device,
        scalar,
        dtype,
        layout,
        input_mem_config,
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        cq_id = 0

        t2 = ttl_tensor_unop(t0, scalar, queue_id=cq_id)

        return tt2torch_tensor(t2)

    return unary_op_optional_output_with_scalar


def make_unary_op_optional_output_with_fast_approx(ttl_tensor_unop):
    @setup_host_and_device
    def unary_op_optional_output_with_fast_approx(
        x,
        *args,
        device,
        fast_and_approx,
        dtype,
        layout,
        input_mem_config,
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        cq_id = 0

        t2 = ttl_tensor_unop(t0, fast_and_approximate_mode=fast_and_approx, queue_id=cq_id)

        return tt2torch_tensor(t2)

    return unary_op_optional_output_with_fast_approx


# eltwise_softmax_in_place = make_unary_op(ttl.tensor.softmax_in_place)
eltwise_cos = make_unary_op_optional_output(ttnn.cos)
eltwise_sin = make_unary_op_optional_output(ttnn.sin)
eltwise_tan = make_unary_op_optional_output(ttnn.tan)
eltwise_acos = make_unary_op_optional_output(ttnn.acos)
eltwise_asin = make_unary_op_optional_output(ttnn.asin)
eltwise_atan = make_unary_op_optional_output(ttnn.atan)
eltwise_i0 = make_unary_op_optional_output(ttnn.i0)
eltwise_log = make_unary_op_optional_output(ttnn.log)
eltwise_log2 = make_unary_op_optional_output(ttnn.log2)
eltwise_log10 = make_unary_op_optional_output(ttnn.log10)
eltwise_logical_not_unary = make_unary_op_optional_output(ttnn.logical_not)
eltwise_isfinite = make_unary_op_optional_output(ttnn.isfinite)
eltwise_isinf = make_unary_op_optional_output(ttnn.isinf)
eltwise_isposinf = make_unary_op_optional_output(ttnn.isposinf)
eltwise_isneginf = make_unary_op_optional_output(ttnn.isneginf)
eltwise_isnan = make_unary_op_optional_output(ttnn.isnan)
eltwise_erfinv = make_unary_op_optional_output(ttnn.erfinv)
eltwise_erf = make_unary_op_optional_output_with_fast_approx(ttnn.erf)
eltwise_erfc = make_unary_op_optional_output_with_fast_approx(ttnn.erfc)
eltwise_gelu = make_unary_op_optional_output_with_fast_approx(ttnn.gelu)
eltwise_exp = make_unary_op_optional_output_with_fast_approx(ttnn.exp)
eltwise_softplus = make_unary_op_optional_output(ttnn.softplus)
eltwise_tanh = make_unary_op_optional_output(ttnn.tanh)
eltwise_softsign = make_ttnn_unary_op(ttnn.softsign)
eltwise_relu = make_unary_op_optional_output(ttnn.relu)
eltwise_relu6 = make_unary_op_optional_output(ttnn.relu6)

eltwise_sqrt = make_unary_op_optional_output(ttnn.sqrt)
eltwise_cbrt = make_ttnn_unary_op(ttnn.cbrt)
eltwise_rad2deg = make_unary_op_composite_ttnn(ttnn.rad2deg)
eltwise_deg2rad = make_unary_op_composite_ttnn(ttnn.deg2rad)
eltwise_sign = make_unary_op_optional_output(ttnn.sign)
eltwise_signbit = make_unary_op_optional_output(ttnn.signbit)
eltwise_abs = make_unary_op_optional_output(ttnn.abs)
eltwise_exp2 = make_unary_op_optional_output(ttnn.exp2)
eltwise_expm1 = make_unary_op_optional_output(ttnn.expm1)
eltwise_neg = make_unary_op_optional_output(ttnn.neg)
eltwise_recip = make_unary_op_optional_output(ttnn.reciprocal)
eltwise_sigmoid = make_unary_op_optional_output(ttnn.sigmoid)
eltwise_sigmoid_accurate = make_unary_op_optional_output(ttnn.sigmoid_accurate)
eltwise_log_sigmoid = make_unary_op_optional_output(ttnn.log_sigmoid)
eltwise_square = make_unary_op_optional_output(ttnn.square)
eltwise_heaviside = make_unary_op_optional_output_with_scalar(ttnn.heaviside)
eltwise_ltz = make_unary_op_optional_output(ttnn.ltz)
eltwise_gtz = make_unary_op_optional_output(ttnn.gtz)
eltwise_lez = make_unary_op_optional_output(ttnn.lez)
eltwise_gez = make_unary_op_optional_output(ttnn.gez)
eltwise_nez = make_unary_op_optional_output(ttnn.nez)
eltwise_eqz = make_unary_op_optional_output(ttnn.eqz)
zeros_like = make_ttnn_unary_op(ttnn.zeros_like)
ones_like = make_ttnn_unary_op(ttnn.ones_like)


def make_binary_op(ttl_tensor_binop):
    @setup_host_and_device
    def binary_op(
        x,
        y,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        output_mem_config,
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
        t2 = ttl_tensor_binop(t0, t1, memory_config=output_mem_config)

        return tt2torch_tensor(t2)

    return binary_op


def make_binary_op_ttnn(ttnn_tensor_binop):
    @setup_host_and_device
    def binary_op(
        x,
        y,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        output_mem_config,
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
        t2 = ttnn_tensor_binop(t0, t1, memory_config=output_mem_config)

        return tt2torch_tensor(t2)

    return binary_op


eltwise_add = make_binary_op_ttnn(ttnn.add)
eltwise_sub = make_binary_op_ttnn(ttnn.sub)
eltwise_mul = make_binary_op_ttnn(ttnn.mul)
eltwise_squared_difference = make_binary_op_ttnn(ttnn.squared_difference)
eltwise_atan2 = make_binary_op_ttnn(ttnn.atan2)
eltwise_ne = make_binary_op_ttnn(ttnn.ne)
eltwise_eq = make_binary_op_ttnn(ttnn.eq)
eltwise_gt = make_binary_op_ttnn(ttnn.gt)
eltwise_lt = make_binary_op_ttnn(ttnn.lt)
eltwise_gte = make_binary_op_ttnn(ttnn.ge)
eltwise_lte = make_binary_op_ttnn(ttnn.le)
eltwise_ldexp = make_binary_op_ttnn(ttnn.ldexp)
eltwise_logaddexp = make_binary_op_ttnn(ttnn.logaddexp)
eltwise_logaddexp2 = make_binary_op_ttnn(ttnn.logaddexp2)
eltwise_logical_xor = make_binary_op_ttnn(ttnn.logical_xor)
eltwise_logical_and = make_binary_op_ttnn(ttnn.logical_and)
eltwise_logical_or = make_binary_op_ttnn(ttnn.logical_or)
eltwise_bias_gelu = make_binary_op_ttnn(ttnn.bias_gelu)

eltwise_min = make_binary_op(ttnn.minimum)
eltwise_max = make_binary_op(ttnn.maximum)

matmul = make_binary_op_ttnn(ttnn.matmul)
outer = make_binary_op(ttnn.outer)

eltwise_scatter = make_binary_op(ttnn.scatter)
eltwise_nextafter = make_binary_op_ttnn(ttnn.nextafter)


def make_binary_op_optional_output(ttl_tensor_binop):
    @setup_host_and_device
    def binary_op(
        x,
        y,
        z,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
        t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

        cq_id = 0

        ttl_tensor_binop(t0, t1, output_tensor=t2, queue_id=cq_id)

        return tt2torch_tensor(t2)

    return binary_op


eltwise_add_optional = make_binary_op_optional_output(ttnn.add)
eltwise_sub_optional = make_binary_op_optional_output(ttnn.sub)
eltwise_mul_optional = make_binary_op_optional_output(ttnn.mul)
eltwise_bias_gelu_optional = make_binary_op_optional_output(ttnn.bias_gelu)
eltwise_squared_difference_optional = make_binary_op_optional_output(ttnn.squared_difference)
eltwise_ne_optional = make_binary_op_optional_output(ttnn.ne)
eltwise_eq_optional = make_binary_op_optional_output(ttnn.eq)
eltwise_gt_optional = make_binary_op_optional_output(ttnn.gt)
eltwise_lt_optional = make_binary_op_optional_output(ttnn.lt)
eltwise_gte_optional = make_binary_op_optional_output(ttnn.ge)
eltwise_lte_optional = make_binary_op_optional_output(ttnn.le)
eltwise_ldexp_optional = make_binary_op_optional_output(ttnn.ldexp)
eltwise_logaddexp_optional = make_binary_op_optional_output(ttnn.logaddexp)
eltwise_logaddexp2_optional = make_binary_op_optional_output(ttnn.logaddexp2)
eltwise_logical_and_optional = make_binary_op_optional_output(ttnn.logical_and)
eltwise_logical_or_optional = make_binary_op_optional_output(ttnn.logical_or)


################################################
#################### Tensor ####################
################################################


@setup_host_and_device
def datacopy(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    device_tensor = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    host_tensor = tt2torch_tensor(device_tensor)

    return host_tensor


@setup_host_and_device
def tensor_pad(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = t0.pad(output_tensor_shape, input_tensor_start, pad_value)

    return tt2torch_tensor(t1)


@setup_host_and_device
def tensor_unpad(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    t0 = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttnn.ROW_MAJOR_LAYOUT,
    )

    t0 = t0.to(layout[0])

    if (device is not None) and (input_mem_config[0] is not None):
        t0 = t0.to(device, input_mem_config[0])

    t1 = t0.unpad(output_tensor_start, output_tensor_end)
    return tt2torch_tensor(t1)


@setup_host_and_device
def pad_to_tile(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    pad_value,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = t0.pad_to_tile(pad_value)

    return tt2torch_tensor(t1)


@setup_host_and_device
def unpad_from_tile(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_shape,
    **kwargs,
):
    t0 = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttnn.ROW_MAJOR_LAYOUT,
    )

    t0 = t0.to(layout[0])

    if (device is not None) and (input_mem_config[0] is not None):
        t0 = t0.to(device, input_mem_config[0])

    t1 = t0.unpad_from_tile(output_tensor_shape)
    return tt2torch_tensor(t1)


@setup_host_and_device
def activation_glu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    dim = kwargs.get("dim", -1)
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.glu(t0, dim, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def activation_geglu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    dim = kwargs.get("dim", -1)
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.geglu(t0, dim, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def activation_reglu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    dim = kwargs.get("dim", -1)
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.reglu(t0, dim, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def activation_swiglu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    dim = kwargs.get("dim", -1)
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.swiglu(t0, dim, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


def bert_large_fused_qkv_matmul(
    x,
    y,
    z,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    a_t = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    b_t = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    bias_t = (
        ttnn.Tensor(
            z.flatten().tolist(),
            z.shape,
            dtype[2],
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .pad([1, 1, 32, 3072], [0, 0, 0, 0], 0)
        .to(layout[2])
        .to(device, input_mem_config[2])
    )

    t3 = custom_matmuls.bert_large_fused_qkv_matmul(a_t, b_t, bias_t, output_mem_config=output_mem_config)
    return tt2torch_tensor(t3)


@setup_host_and_device
def bert_large_selfout_matmul(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[2] == ttnn.TILE_LAYOUT:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = custom_matmuls.bert_large_selfout_matmul(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def bert_large_ff2_matmul(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[2] == ttnn.TILE_LAYOUT:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = custom_matmuls.bert_large_ff2_matmul(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def bert_large_ff1_matmul(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[2] == ttnn.TILE_LAYOUT:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = custom_matmuls.bert_large_ff1_matmul(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def bert_large_pre_softmax_bmm(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = custom_matmuls.bert_large_pre_softmax_bmm(t0, t1, output_mem_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def bert_large_post_softmax_bmm(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = custom_matmuls.bert_large_post_softmax_bmm(t0, t1, output_mem_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def embeddings(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    x = x.int()
    x_shape = x.shape
    y_shape = y.shape

    batch_size = x_shape[0]
    num_rows = x_shape[3]
    embedding_dim = y_shape[3]

    x_ref = x.detach().clone()

    t0 = torch.clamp(x_ref, min=0, max=y.shape[-2] - 1)
    t0 = ttnn.Tensor(t0, dtype[0]).to(device, input_mem_config[0])

    t1 = ttnn.Tensor(y, dtype[1]).to(device, input_mem_config[1])

    t2 = ttnn.embedding(t0, t1, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=output_mem_config)
    t2 = ttnn.reshape(t2, [t2.shape[0], 1, t2.shape[1], t2.shape[2]])

    tt_data = t2.cpu().to_torch()

    tt_got_back = torch.Tensor(tt_data).reshape((batch_size, 1, num_rows, embedding_dim))

    return tt_got_back


@setup_host_and_device
def rmsnorm_noweights(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.rms_norm(t0, epsilon=1e-5, weight=None, bias=None, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def rmsnorm(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[1] == ttnn.TILE_LAYOUT:
        y = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    if layout[2] == ttnn.TILE_LAYOUT:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t1 = ttnn.rms_norm(t0, epsilon=1e-5, weight=t1, bias=t2, memory_config=output_mem_config)

    return tt2torch_tensor(t1)


def complex_real(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    temp = torch.cat([x.real, x.imag], -1)
    t0 = setup_tt_tensor(temp, device, layout[0], input_mem_config[0], dtype[0])

    tt_result = ttnn.real(t0, memory_config=output_mem_config)
    tt_result = tt2torch_tensor(tt_result)

    return tt_result


@setup_host_and_device
def complex_div(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    tempx = torch.cat([x.real, x.imag], -1)
    t0 = setup_tt_tensor(tempx, device, layout[0], input_mem_config[0], dtype[0])

    tempy = torch.cat([y.real, y.imag], -1)
    t1 = setup_tt_tensor(tempy, device, layout[1], input_mem_config[1], dtype[1])

    tt_result = ttnn.experimental.tensor.complex_div(t0, t1, output_mem_config=output_mem_config)
    result = ttl_complex_2_torch_complex(tt_result)

    return result


@setup_host_and_device
def complex_mul(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    tempx = torch.cat([x.real, x.imag], -1)
    t0 = setup_tt_tensor(tempx, device, layout[0], input_mem_config[0], dtype[0])

    tempy = torch.cat([y.real, y.imag], -1)
    t1 = setup_tt_tensor(tempy, device, layout[1], input_mem_config[1], dtype[1])

    tt_result = ttnn.experimental.tensor.complex_mul(t0, t1, output_mem_config=output_mem_config)
    result = ttl_complex_2_torch_complex(tt_result)

    return result


@setup_host_and_device
def complex_imag(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    temp = torch.cat([x.real, x.imag], -1)
    t0 = setup_tt_tensor(temp, device, layout[0], input_mem_config[0], dtype[0])

    tt_result = ttnn.imag(t0, memory_config=output_mem_config)
    tt_result = tt2torch_tensor(tt_result)

    return tt_result


@setup_host_and_device
def abs_bw(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.abs_bw(t0, t1, output_mem_config)[0]

    return tt2torch_tensor(t2)


@setup_host_and_device
def binary_le_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t3 = ttnn.binary_le_bw(t0, t1, output_mem_config)[0]

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_max_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.max_bw(t0, t1, t2, output_mem_config)

    return [tt2torch_tensor(t3[0]), tt2torch_tensor(t3[1])]


@setup_host_and_device
def eltwise_min_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.min_bw(t0, t1, t2, output_mem_config)

    return [tt2torch_tensor(t3[0]), tt2torch_tensor(t3[1])]


@setup_host_and_device
def eltwise_sub_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.sub_bw(t0, t1, t2, output_mem_config)

    return [tt2torch_tensor(t3[0]), tt2torch_tensor(t3[1])]


@setup_host_and_device
def eltwise_exp_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.exp_bw(t0, t1, output_mem_config)[0]

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_tanh_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.tanh_bw(t0, t1, output_mem_config)[0]

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_mul_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.mul_bw(t0, t1, t2, output_mem_config)

    return [tt2torch_tensor(t3[0]), tt2torch_tensor(t3[1])]


@setup_host_and_device
def eltwise_tan_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    y.requires_grad = True
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.tan_bw(t0, t1, output_mem_config)[0]

    return tt2torch_tensor(t2)


@setup_host_and_device
def unary_pow_bw(
    x,
    y,
    *args,
    exponent,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.pow_bw(t0, t1, exponent, memory_config=output_mem_config)[0]

    return tt2torch_tensor(t2)


@setup_host_and_device
def sub_unary_bw(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.sub_bw(t0, t1, memory_config=output_mem_config)[0]

    return tt2torch_tensor(t2)


@setup_host_and_device
def fill_zero_bw(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.fill_zero_bw(t0, memory_config=output_mem_config)[0]

    return tt2torch_tensor(t1)


@setup_host_and_device
def tt_embedding_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    x.requires_grad = True
    z.requires_grad = True

    batch_size = y.shape[0]
    no_of_embeddings = y.shape[3]

    grad_tensor = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    input_tensor = torch.reshape(torch.arange(0, batch_size * no_of_embeddings), shape=y.shape)
    input_tensor = setup_tt_tensor(input_tensor, device, layout[1], input_mem_config[1], dtype[1])

    weights_tensor = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.embedding_bw(grad_tensor, input_tensor, weights_tensor)[0]

    return tt2torch_tensor(t3)


@setup_host_and_device
def interleaved_to_sharded_partial(
    x,
    *args,
    num_slices,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    grid_size = (8, 8)
    W = x.shape[-1]
    H = x.shape[-2]
    height_shard_spec = [H // 2, W]

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_initial = torch.randn(x.shape).bfloat16().float()

    t2 = torch2tt_tensor(out_initial, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype[0])

    for slice_index in range(num_slices):
        t1 = ttnn.interleaved_to_sharded_partial(
            t0,
            grid_size,
            height_shard_spec,
            num_slices,
            slice_index,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        ttnn.sharded_to_interleaved_partial(
            t1,
            t2,
            num_slices,
            slice_index,
            memory_config=interleaved_mem_config,
        )

    returned_res = tt2torch_tensor(t2)
    return returned_res


@setup_host_and_device
def interleaved_to_sharded_partial_coregrid(
    x,
    *args,
    num_slices,
    x_core,
    y_core,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    grid_size = (x_core, y_core)
    W = x.shape[-1]
    H = x.shape[-2]
    height_shard_spec = [H // 2, W]

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_initial = torch.randn(x.shape).bfloat16().float()

    t2 = torch2tt_tensor(out_initial, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype[0])

    for slice_index in range(num_slices):
        t1 = ttnn.interleaved_to_sharded_partial(
            t0,
            grid_size,
            height_shard_spec,
            num_slices,
            slice_index,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        ttnn.sharded_to_interleaved_partial(
            t1,
            t2,
            num_slices,
            slice_index,
            memory_config=interleaved_mem_config,
        )

    returned_res = tt2torch_tensor(t2)
    return returned_res

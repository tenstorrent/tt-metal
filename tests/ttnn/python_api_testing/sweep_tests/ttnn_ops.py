# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ttnn.model_preprocessing import preprocess_model, preprocess_model_parameters
import transformers
from models.demos.bert.tt import ttnn_bert


from tests.tt_eager.python_api_testing.sweep_tests.model_tests import (
    TorchConvReluConv,
    TTNNConvReluConv,
    run_conv_relu_conv,
    TorchConvConv,
    TTNNConvConv,
    run_conv_conv,
    BertFeedForward,
)


def setup_ttnn_tensor(x, device, layout, input_mem_config, dtype):
    # Check if input is scalar -> Then return scalar
    if len(x.shape) == 1 and x.shape[0] == 1:
        return x.item()

    input_tensor = ttnn.from_torch(
        x,
        dtype=dtype,
        layout=layout,
        device=device if input_mem_config is not None else None,
        memory_config=input_mem_config,
    )

    return input_tensor


def ttnn_tensor_to_torch(x, output_mem_config=None):
    output_tensor = x
    # output_tensor = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    # output_tensor = ttnn.from_device(output_tensor)
    return ttnn.to_torch(output_tensor)


def ones(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.ones(
        x.shape,
        device=device if input_mem_config[0] is not None else None,
        dtype=dtype[0],
        layout=layout[0],
        memory_config=output_mem_config,
    )

    return ttnn_tensor_to_torch(t0)


def ones_like(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.ones_like(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = ttnn.full(
        x.shape,
        fill_value=scalar,
        device=device if input_mem_config[0] is not None else None,
        dtype=dtype[0],
        layout=layout[0],
        memory_config=output_mem_config,
    )

    return ttnn_tensor_to_torch(t0)


def eltwise_hardswish(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardswish(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_hardtanh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardtanh(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.heaviside(t0, scalar, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_hypot(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.hypot(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_i0(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.i0(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_isfinite(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isfinite(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_isinf(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isinf(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_isnan(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isnan(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_isneginf(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isneginf(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_isposinf(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isposinf(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.leaky_relu(t0, negative_slope, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def activation_glu(
    x,
    *args,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.glu(t0, dim=dim, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.lerp(t0, t1, weight, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_addalpha(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.addalpha(t0, t1, alpha=alpha, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_lerp_ternary(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.lerp(t0, t1, t2, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t3)


def eltwise_lgamma(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.lgamma(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_log(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.log(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_log10(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.log10(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_log1p(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.log1p(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_log2(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.log2(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_log_sigmoid(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.log_sigmoid(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.celu(t0, alpha=alpha, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_logit(
    x,
    *args,
    eps,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.logit(t0, eps=eps, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_mish(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.mish(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_multigammaln(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.multigammaln(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_neg(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.neg(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_prelu(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.prelu(t0, scalar, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_relu(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.relu(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_logical_not(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.logical_not(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_xlogy(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.xlogy(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_squared_difference(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.squared_difference(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_add_and_apply_activation(
    x,
    y,
    *args,
    activation,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    if activation is not None:
        activations = [activation]
    else:
        activations = None

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.add(t0, t1, activations=activations, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_add_and_apply_activation_(
    x,
    y,
    *args,
    activation,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    if activation is not None:
        activations = [activation]
    else:
        activations = None

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.add_(t0, t1, activations=activations)

    return ttnn_tensor_to_torch(t2)


def eltwise_gtz(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.gtz(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_ltz(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.ltz(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_gez(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.gez(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_lez(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.lez(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_nez(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.nez(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_add(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.add(t0, t1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t2)


def eltwise_exp(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.exp(t0, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.permute(t0, permute_dims)  # , memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def reshape(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    reshape_dims,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.reshape(t0, reshape_dims)  # , memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def transpose_01(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transpose(t0, 0, 1)  # , memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def transpose_02(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transpose(t0, 0, 2)  # , memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def transpose_03(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transpose(t0, 0, 3)  # , memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def transpose_12(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transpose(t0, 1, 2)  # , memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def transpose_13(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transpose(t0, 1, 3)  # , memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def transpose_23(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transpose(t0, 2, 3)  # , memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def gelu(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.gelu(t0, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def activation_geglu(
    x,
    *args,
    dim=-1,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.geglu(t0, dim, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def activation_swiglu(
    x,
    *args,
    dim=-1,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.swiglu(t0, dim, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def eltwise_sub(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.subtract(t0, t1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t2)


def embeddings(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    x = x.int()
    x = torch.clamp(x, min=0, max=y.shape[0] - 1)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.embedding(t0, t1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t2)


def eltwise_tanh(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.tanh(t0, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def softmax(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.softmax(t0, dim=-1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def softplus(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.softplus(t0, beta=beta, threshold=threshold, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def mul(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.multiply(t0, t1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t2)


def linear(x, weight, bias=None, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    # tensor preprocessing
    if bias is not None:
        bias = bias.repeat(1, 1, 32, 1)

    weight = torch.transpose(weight, 2, 3)
    batch_size = x.shape[0]
    num_cores_x = 12

    # ttnn setup
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    tt_weight = setup_ttnn_tensor(weight, device, layout[1], input_mem_config[1], dtype[1])

    if bias is not None:
        tt_bias = setup_ttnn_tensor(bias, device, layout[2], input_mem_config[2], dtype[2])
    else:
        tt_bias = None

    t1 = ttnn.linear(t0, tt_weight, bias=tt_bias, dtype=ttnn.bfloat16, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_softmax_in_place(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.softmax(t0, -1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


def matmul(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    xcoregrid=-1,
    ycoregrid=-1,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    if xcoregrid != -1 and ycoregrid != -1:
        t2 = ttnn.matmul(
            t0,
            t1,
            memory_config=output_mem_config,
            core_grid=ttnn.CoreGrid(y=ycoregrid, x=xcoregrid),
        )
    else:
        t2 = ttnn.matmul(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def layernorm(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.layer_norm(t0, weight=t1, bias=t2, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t3)


def layernorm_residual(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_ttnn_tensor(w, device, layout[3], input_mem_config[3], dtype[3])

    t4 = ttnn.layer_norm(t0, residual_input_tensor=t1, weight=t2, bias=t3, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t4)


def layernorm_noweights(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.layer_norm(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def attention_softmax_nomask(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t2 = ttnn.transformer.attention_softmax(t0, head_size=None, attention_mask=None, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def attention_softmax(
    x,
    y,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    y[y <= 0.50] = 0
    y[y > 0.50] = 1

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    if scalar < 0:
        scalar = -scalar

    t2 = ttnn.transformer.attention_softmax(t0, head_size=scalar, attention_mask=t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def rmsnorm(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.rms_norm(t0, residual_input_tensor=t1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t2)


def transformer_concatenate_heads(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transformer.concatenate_heads(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.full_like(t0, fill_value=scalar, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_relu6(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.relu6(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_rsqrt(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.rsqrt(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_sigmoid(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sigmoid(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_sign(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sign(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_silu(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.silu(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_sigmoid_accurate(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sigmoid_accurate(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_sin(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sin(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_sinh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sinh(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_softshrink(
    x,
    *args,
    _lambda,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.softshrink(t0, lambd=_lambda, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_softsign(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.softsign(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_tan(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.tan(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_tanh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.tanh(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_swish(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.swish(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_signbit(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.signbit(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_rad2deg(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.rad2deg(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_polygamma(
    x,
    *args,
    k,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.polygamma(t0, k=k, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_recip(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.reciprocal(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_sqrt(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sqrt(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_square(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.square(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def tril(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.tril(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def triu(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.triu(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_tanhshrink(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.tanhshrink(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_threshold(
    x,
    *args,
    threshold,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.threshold(t0, threshold, value, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_eqz(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.eqz(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_gt(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.gt(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_gte(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.ge(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_lt(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.lt(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_lte(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.le(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_eq(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.eq(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_ne(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.ne(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_isclose(
    x,
    y,
    *args,
    rtol,
    atol,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.isclose(t0, t1, rtol=rtol, atol=atol, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.polyval(t0, coeffs, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def abs(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.abs(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def acos(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.acos(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def acosh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.acosh(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def asin(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.asin(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def asinh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.asinh(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def atan(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.atan(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def atan2(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[0], input_mem_config[0], dtype[0])

    t2 = ttnn.atan2(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def atanh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.atanh(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def cos(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.cos(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def cosh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.cosh(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def exp(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.exp(t0)

    return ttnn_tensor_to_torch(t1)


def exp2(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.exp2(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def expm1(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.expm1(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def elu(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.elu(t0, alpha, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def erf(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.erf(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def erfc(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.erfc(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def erfinv(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.erfinv(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def hardsigmoid(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardsigmoid(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def deg2rad(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.deg2rad(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def hardshrink(
    x,
    *args,
    _lambda,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardshrink(t0, lambd=_lambda, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.clone(t0, output_mem_config, dtype=dtype[0])

    return ttnn_tensor_to_torch(t1)


def cbrt(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.cbrt(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def digamma(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.digamma(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.clip(t0, low, high, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.repeat_interleave(t0, repeat, dim)  # memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def addcmul(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.addcmul(t0, t1, t2, value=scalar, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t3)


def groupnorm_noweights(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.group_norm(t0, num_groups=1, weight=None, bias=None, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.addcdiv(t0, t1, t2, value=scalar, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t3)


def where(x, y, z, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.where(t0, t1, t2, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t3)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    xtt = (t0, t1)
    t2 = ttnn.concat(xtt, dim, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def global_avg_pool2d(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    input_tensor = torch.permute(x, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(
        input_tensor, dtype=dtype[0], layout=layout[0], device=device, memory_config=input_mem_config[0]
    )
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    return output_tensor.to(torch.float32)


def upsample(
    x,
    *args,
    scale_factor,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.upsample(t0, scale_factor=scale_factor, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def l1_loss_sum(
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
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.l1_loss(t0, t1, loss_mode="sum", memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def l1_loss_mean(
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
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.l1_loss(t0, t1, loss_mode="mean", memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def l1_loss(
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
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.l1_loss(t0, t1, loss_mode="none", memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def mse_loss_sum(
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
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.mse_loss(t0, t1, loss_mode="sum", memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def mse_loss_mean(
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
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.mse_loss(t0, t1, loss_mode="mean", memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def mse_loss(
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
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.mse_loss(t0, t1, loss_mode="none", memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def ldexp(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.ldexp(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def logical_and(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.logical_and(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def logical_or(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.logical_or(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def logical_xor(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.logical_xor(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def pow(
    x,
    y,
    *args,
    exponent=None,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    if exponent is not None:
        t1 = ttnn.pow(t0, exponent, memory_config=output_mem_config)
        return ttnn_tensor_to_torch(t1)
    else:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
        t2 = ttnn.pow(t0, t1, memory_config=output_mem_config)
        return ttnn_tensor_to_torch(t2)


def logaddexp(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.logaddexp(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def logaddexp2(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.logaddexp2(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def sum(
    x,
    *args,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.sum(t0, dim=dim, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def max(
    x,
    *args,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.max(t0, dim=dim, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def min(
    x,
    *args,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.min(t0, dim=dim, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_max(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.maximum(t0, t1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t2)


def eltwise_min(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.minimum(t0, t1, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t2)


def rotary_embedding(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    torch.manual_seed(0)

    cache_size = 2048
    input_dtype = ttnn.bfloat16
    sincos_dtype = ttnn.bfloat16

    sin_cos_shape = (1, 1, cache_size, 64)

    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    # TTNN -----------------------------------------------
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    cost = setup_ttnn_tensor(cos_cached, device, layout[0], input_mem_config[0], dtype[0])
    sint = setup_ttnn_tensor(sin_cached, device, layout[0], input_mem_config[0], dtype[0])
    t2 = ttnn.experimental.rotary_embedding(t0, cost, sint, None, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def activation_reglu(
    x,
    *args,
    dim=-1,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.reglu(t0, dim, memory_config=output_mem_config)
    return ttnn_tensor_to_torch(t1)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.arange(start, end, step, device)
    return ttnn_tensor_to_torch(t1)


def nextafter(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.nextafter(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def empty(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t1 = ttnn.empty(x.shape, dtype, layout, device, output_mem_config)

    result = ttnn_tensor_to_torch(t1)
    return ttnn_tensor_to_torch(t1)


def attention_softmax_nomask_2(
    x,
    y,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    y[y <= 0.50] = 0
    y[y > 0.50] = 1

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    if scalar < 0:
        scalar = -scalar

    t2 = ttnn.transformer.attention_softmax_(t0, head_size=scalar, attention_mask=None)

    return ttnn_tensor_to_torch(t2)


def attention_softmax_2(
    x,
    y,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    y[y <= 0.50] = 0
    y[y > 0.50] = 1

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    if scalar < 0:
        scalar = -scalar

    t2 = ttnn.transformer.attention_softmax_(t0, head_size=scalar, attention_mask=t1)

    return ttnn_tensor_to_torch(t2)


def zeros(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    # t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.zeros(
        x.shape,
        device=device,
        dtype=dtype[0],
        layout=layout[0],
        memory_config=output_mem_config,
    )

    return ttnn_tensor_to_torch(t1)


def zeros_like(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.zeros_like(
        t0,
        memory_config=output_mem_config,
    )

    return ttnn_tensor_to_torch(t1)


def preprocessing_model_conv_conv(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    torch.manual_seed(234)
    num_channels = x.shape[1]

    # create torch model
    torch_model = TorchConvConv(num_input_channels=num_channels, num_output_channels=num_channels)
    torch_model.eval()

    torch_input_tensor = x.to(torch.float32)

    # get model parameters
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    # create and run TTNN model
    ttnn_model = TTNNConvConv(parameters)
    output_tensor = run_conv_conv(ttnn_model, torch_input_tensor)

    return output_tensor


def preprocessing_model_conv_relu_conv(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    torch.manual_seed(234)
    torch_input_tensor = x.to(torch.float32)
    num_channels = x.shape[1]

    # create torch model
    torch_model = TorchConvReluConv(num_input_channels=num_channels, num_output_channels=num_channels)
    torch_model.eval()

    # get model parameters
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    # create and run TTNN model
    ttnn_model = TTNNConvReluConv(parameters)
    output_tensor = run_conv_relu_conv(ttnn_model, torch_input_tensor, device)

    return output_tensor


def preprocessing_model_bert_1(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    torch.manual_seed(234)

    model_name = "phiyodr/bert-large-finetuned-squad2"

    config = transformers.BertConfig.from_pretrained(model_name)
    model = BertFeedForward(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = x

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype[0], layout=layout[0], device=device)
    output = ttnn_bert.bert_feedforward(
        config,
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    return output


def preprocessing_model_bert_2(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    torch.manual_seed(234)

    model_name = "phiyodr/bert-large-finetuned-squad2"

    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    model = transformers.models.bert.modeling_bert.BertEncoder(config).eval()

    torch_hidden_states = x.to(torch.float32)
    torch_attention_mask = None

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype[0], layout=layout[0], device=device)
    if torch_attention_mask is not None:
        attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        attention_mask = None
    output = ttnn_bert.bert_encoder(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    return output


def preprocessing_model_bert_3(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    torch.manual_seed(234)

    model_name = "phiyodr/bert-large-finetuned-squad2"

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertAttention(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = x
    sequence_size = x.shape[1]
    torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype[0], layout=layout[0], device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask, dtype[0], layout=layout[0], device=device)
    output = ttnn_bert.bert_attention(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    return output


def preprocessing_model_bert_4(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    torch.manual_seed(0)
    model_name = "phiyodr/bert-large-finetuned-squad2"

    # set parameters
    batch_size = x.shape[0]
    sequence_size = x.shape[1]
    num_hidden_layers = 1

    # get torch model
    config = transformers.BertConfig.from_pretrained(model_name)
    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers
    else:
        pytest.skip("Test mismatches when the default number of hidden layers is used")
    model = transformers.BertForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    # set inputs
    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = None

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    ttnn_bert_inputs = ttnn_bert.preprocess_inputs(
        torch_input_ids,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        device=device,
    )
    output = ttnn_bert.bert_for_question_answering(
        config,
        *ttnn_bert_inputs,
        parameters=parameters,
    )

    output = ttnn.to_torch(output)
    start_logits = output[..., 0]
    end_logits = output[..., 1]

    return start_logits


def eltwise_mac(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttnn.mac(t0, t1, t2, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t3)


def mean(x, *args, dim, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.mean(t0, dim)

    return ttnn_tensor_to_torch(t1)


def std(x, *args, dim, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.std(t0, dim)

    return ttnn_tensor_to_torch(t1)


def var(x, *args, dim, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.var(t0, dim)

    return ttnn_tensor_to_torch(t1)


def max_pool2d_tt(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    batch_size = x.shape[0]
    input_height = x.shape[2]
    input_width = x.shape[3]

    m = ttnn.MaxPool2d(
        kernel_size=3,
        stride=2,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        reader_patterns_cache={},
    )

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = m(t0)

    return ttnn_tensor_to_torch(t1)


def repeat(
    x,
    *args,
    shape,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.repeat(t0, ttnn.Shape(shape))

    return ttnn_tensor_to_torch(t1)


def eltwise_subtract_and_apply_activation(
    x,
    y,
    *args,
    activation,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    if activation is not None:
        activations = [activation]
    else:
        activations = None

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.subtract(t0, t1, activations=activations, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_subtract_and_apply_activation_(
    x,
    y,
    *args,
    activation,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    if activation is not None:
        activations = [activation]
    else:
        activations = None

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.subtract_(t0, t1, activations=activations)

    return ttnn_tensor_to_torch(t2)


def eltwise_multiply_and_apply_activation(
    x,
    y,
    *args,
    activation,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    if activation is not None:
        activations = [activation]
    else:
        activations = None

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.multiply(t0, t1, activations=activations, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def eltwise_multiply_and_apply_activation_(
    x,
    y,
    *args,
    activation,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    if activation is not None:
        activations = [activation]
    else:
        activations = None

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.multiply_(t0, t1, activations=activations)

    return ttnn_tensor_to_torch(t2)


def pad(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.pad(
        t0,
        output_tensor_shape,
        input_tensor_start,
        pad_value,
        memory_config=output_mem_config,
    )

    return ttnn_tensor_to_torch(t1)


def eltwise_relu_min(
    x,
    *args,
    lower_limit,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.relu_min(t0, lower_limit, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_relu_max(
    x,
    *args,
    upper_limit,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.relu_max(t0, upper_limit, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def unpad(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.slice(t0, output_tensor_start, output_tensor_end, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def unary_add_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t3 = ttnn.add_bw(t0, t1, scalar, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t3[0])


def clamp_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    if scalar >= 0:
        max = scalar
        min = -scalar
    else:
        max = -scalar
        min = scalar

    t2 = ttnn.clamp_bw(t0, t1, min=min, max=max, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def eltwise_unary_fmod(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.fmod(t0, value, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_softmax_in_place(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.softmax_in_place(t0)

    return ttnn_tensor_to_torch(t1)


def eltwise_scale_mask_softmax_in_place(
    x,
    y,
    scale,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.scale_mask_softmax_in_place(t0, scale, t1)

    return ttnn_tensor_to_torch(t2)


def clamp_max_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.clamp_bw(t0, t1, max=scalar, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def addcmul_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor1
    w,  # other_tensor2
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_ttnn_tensor(w, device, layout[3], input_mem_config[3], dtype[3])

    t4 = ttnn.addcmul_bw(t0, t1, t2, t3, scalar, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t4[0]), ttnn_tensor_to_torch(t4[1]), ttnn_tensor_to_torch(t4[2])]


def addalpha_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor1
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.addalpha_bw(t0, t1, t2, alpha, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t3[0]), ttnn_tensor_to_torch(t3[1])]


def eltwise_add_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.add_bw(t0, t1, t2, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t3[0]), ttnn_tensor_to_torch(t3[1])]


def clamp_min_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.clamp_bw(t0, t1, min=scalar, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def eltwise_rdiv(
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
    queue_id = 0
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.rdiv(t0, factor, memory_config=output_mem_config, queue_id=queue_id)

    return ttnn_tensor_to_torch(t1)


def eltwise_rsub(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.rsub(t0, factor, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def neg_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.neg_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def log_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.log_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def rsub_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(y, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.rsub_bw(t0, t1, t2, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t3)


def where_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor1
    w,  # other_tensor2
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    y = y > 0

    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_ttnn_tensor(w, device, layout[3], input_mem_config[3], dtype[3])

    t4 = ttnn.where_bw(t0, t1, t2, t3, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t4[0]), ttnn_tensor_to_torch(t4[1])]


def addcdiv_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor1
    w,  # other_tensor2
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_ttnn_tensor(w, device, layout[3], input_mem_config[3], dtype[3])

    t4 = ttnn.addcdiv_bw(t0, t1, t2, t3, scalar, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t4[0]), ttnn_tensor_to_torch(t4[1]), ttnn_tensor_to_torch(t4[2])]


def sqrt_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.sqrt_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def rsqrt_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.rsqrt_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def relu_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.relu_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def log10_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.log10_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def fmod_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.fmod_bw(t0, t1, value, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.subalpha(t0, t1, alpha=alpha, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def frac_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.frac_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def log2_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.log2_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def gelu_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    approximate = "tanh" if fast_and_approx else "none"

    t2 = ttnn.gelu_bw(t0, t1, approximate=approximate, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def log1p_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.log1p_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def log_sigmoid_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.log_sigmoid_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def logaddexp_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.logaddexp_bw(t0, t1, t2, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t3[0]), ttnn_tensor_to_torch(t3[1])]


def ceil_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.ceil_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def logaddexp2_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.logaddexp2_bw(t0, t1, t2, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t3[0]), ttnn_tensor_to_torch(t3[1])]


def hardshrink_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    _lambda,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.hardshrink_bw(t0, t1, lambd=_lambda, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def celu_bw(
    x,  # grad_tensor
    y,  # input_tensor
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.celu_bw(t0, t1, alpha=alpha, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def hardtanh_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.hardtanh_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def hypot_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.hypot_bw(t0, t1, t2, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t3[0]), ttnn_tensor_to_torch(t3[1])]


def i0_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.i0_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def cosh_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.cosh_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def concat_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor
    *args,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[1], dtype[2])

    t3 = ttnn.concat_bw(t0, t1, t2, dim=dim, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t3[0]), ttnn_tensor_to_torch(t3[1])]


def cos_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.cos_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def normalize_global(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.normalize_global(t0)

    return ttnn_tensor_to_torch(t1)


def normalize_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.normalize_hw(t0)

    return ttnn_tensor_to_torch(t1)


def erf_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.erf_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def erfc_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.erfc_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def erfinv_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.erfinv_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def expm1_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.expm1_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def eltwise_bitwise_and(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.bitwise_and(t0, value, memory_config=output_mem_config, queue_id=0)

    return ttnn_tensor_to_torch(t1)


def eltwise_bitwise_or(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.bitwise_or(t0, value, memory_config=output_mem_config, queue_id=0)

    return ttnn_tensor_to_torch(t1)


def floor_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.floor_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def floor(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.floor(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_unary_floor_div(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.floor_div(t0, value, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def fill_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.fill_bw(t0, t1, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def eltwise_div_no_nan(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.div_no_nan(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def complex_polar_bw(
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
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.complex_tensor(
        setup_ttnn_tensor(y.real, device, layout[1], input_mem_config[0], dtype[1]),
        setup_ttnn_tensor(y.imag, device, layout[1], input_mem_config[0], dtype[1]),
    )

    t2 = ttnn.polar_bw(t0, t1, memory_config=output_mem_config)[0]

    return torch.complex(
        ttnn_tensor_to_torch(t2.real).to(torch.float32), ttnn_tensor_to_torch(t2.imag).to(torch.float32)
    )


def eltwise_unary_div_no_nan(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.div_no_nan(t0, value, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def complex_conj(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.conj(t0, memory_config=output_mem_config)

    return torch.complex(
        ttnn_tensor_to_torch(t1.real).to(torch.float32), ttnn_tensor_to_torch(t1.imag).to(torch.float32)
    )


def complex_abs(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.abs(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def complex_recip_bw(
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
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )
    t1 = ttnn.complex_tensor(
        setup_ttnn_tensor(y.real, device, layout[1], input_mem_config[1], dtype[1]),
        setup_ttnn_tensor(y.imag, device, layout[1], input_mem_config[1], dtype[1]),
    )

    t2 = ttnn.reciprocal_bw(t0, t1, memory_config=output_mem_config)[0]

    return torch.complex(
        ttnn_tensor_to_torch(t2.real).to(torch.float32), ttnn_tensor_to_torch(t2.imag).to(torch.float32)
    )


def complex_polar(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.polar(t0, memory_config=output_mem_config)

    return torch.complex(
        ttnn_tensor_to_torch(t1.real).to(torch.float32), ttnn_tensor_to_torch(t1.imag).to(torch.float32)
    )


def complex_mul_bw(
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
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )
    t1 = ttnn.complex_tensor(
        setup_ttnn_tensor(y.real, device, layout[1], input_mem_config[1], dtype[1]),
        setup_ttnn_tensor(y.imag, device, layout[1], input_mem_config[1], dtype[1]),
    )

    t2 = ttnn.complex_tensor(
        setup_ttnn_tensor(z.real, device, layout[2], input_mem_config[2], dtype[2]),
        setup_ttnn_tensor(z.imag, device, layout[2], input_mem_config[2], dtype[2]),
    )

    t3 = ttnn.mul_bw(t0, t1, t2, memory_config=output_mem_config)

    return [
        torch.complex(
            ttnn_tensor_to_torch(t3[0].real).to(torch.float32), ttnn_tensor_to_torch(t3[0].imag).to(torch.float32)
        ),
        torch.complex(
            ttnn_tensor_to_torch(t3[1].real).to(torch.float32), ttnn_tensor_to_torch(t3[1].imag).to(torch.float32)
        ),
    ]


def complex_recip(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.reciprocal(t0, memory_config=output_mem_config)

    return torch.complex(
        ttnn_tensor_to_torch(t1.real).to(torch.float32), ttnn_tensor_to_torch(t1.imag).to(torch.float32)
    )


def complex_add_bw(
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
    alpha = torch.tensor(1.0, dtype=torch.bfloat16)

    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.complex_tensor(
        setup_ttnn_tensor(y.real, device, layout[1], input_mem_config[1], dtype[1]),
        setup_ttnn_tensor(y.imag, device, layout[1], input_mem_config[1], dtype[1]),
    )

    t2 = ttnn.complex_tensor(
        setup_ttnn_tensor(z.real, device, layout[2], input_mem_config[2], dtype[2]),
        setup_ttnn_tensor(z.imag, device, layout[2], input_mem_config[2], dtype[2]),
    )

    t3 = ttnn.add_bw(t0, t1, t2, alpha=alpha, memory_config=output_mem_config)

    return [
        torch.complex(
            ttnn_tensor_to_torch(t3[0].real).to(torch.float32), ttnn_tensor_to_torch(t3[0].imag).to(torch.float32)
        ),
        torch.complex(
            ttnn_tensor_to_torch(t3[1].real).to(torch.float32), ttnn_tensor_to_torch(t3[1].imag).to(torch.float32)
        ),
    ]


def complex_angle(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.angle(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def complex_conj_bw(
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
    if dtype[0] == ttnn.bfloat8_b:
        dtype[0] = ttnn.bfloat16

    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )
    t1 = ttnn.complex_tensor(
        setup_ttnn_tensor(y.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(y.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t2 = ttnn.conj_bw(t0, t1, memory_config=output_mem_config)[0]

    return torch.complex(
        ttnn_tensor_to_torch(t2.real).to(torch.float32), ttnn_tensor_to_torch(t2.imag).to(torch.float32)
    )


def eltwise_bias_gelu_unary(x, *args, bias, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.bias_gelu(t0, bias, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_identity(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.identity(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def subalpha_bw(
    x,  # grad_tensor
    y,  # input_tensor
    z,  # other_tensor1
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.subalpha_bw(t0, t1, t2, alpha, memory_config=output_mem_config)

    return [ttnn_tensor_to_torch(t3[0]), ttnn_tensor_to_torch(t3[1])]


def unary_remainder(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.remainder(t0, scalar, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_remainder(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.remainder(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2)


def argmax(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    dim = kwargs.pop("dim")

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.argmax(t0, dim=dim, memory_config=output_mem_config)

    if dim:
        return ttnn_tensor_to_torch(t1).squeeze(dim=dim)
    else:
        return ttnn_tensor_to_torch(t1).squeeze()


def unary_remainder_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.remainder_bw(t0, t1, scalar, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)


def topk(
    x,
    *args,
    dim,
    k,
    largest,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1_values, t1_indices = ttnn.topk(t0, k=k, dim=dim, largest=largest, sorted=True)
    t1_values = ttnn_tensor_to_torch(t1_values)
    t1_indices = ttnn_tensor_to_torch(t1_indices).to(torch.int64)
    t1_gather_values = torch.gather(x, dim, t1_indices)

    return [t1_values, t1_gather_values]


def eltwise_trunc(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.trunc(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_frac(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.frac(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def eltwise_ceil(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.ceil(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def complex_imag(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.imag(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def complex_imag_bw(
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
    if dtype[1] == ttnn.bfloat8_b:
        dtype[1] = ttnn.bfloat16

    t0 = setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.complex_tensor(
        setup_ttnn_tensor(y.real, device, layout[1], input_mem_config[1], dtype[1]),
        setup_ttnn_tensor(y.imag, device, layout[1], input_mem_config[1], dtype[1]),
    )

    t2 = ttnn.imag_bw(t0, t1, memory_config=output_mem_config)[0]

    return torch.complex(
        ttnn_tensor_to_torch(t2.real).to(torch.float32), ttnn_tensor_to_torch(t2.imag).to(torch.float32)
    )


def complex_real(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.real(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def complex_real_bw(
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
    if dtype[1] == ttnn.bfloat8_b:
        dtype[1] = ttnn.bfloat16

    t0 = setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.complex_tensor(
        setup_ttnn_tensor(y.real, device, layout[1], input_mem_config[1], dtype[1]),
        setup_ttnn_tensor(y.imag, device, layout[1], input_mem_config[1], dtype[1]),
    )

    t2 = ttnn.real_bw(t0, t1, memory_config=output_mem_config)[0]

    return torch.complex(
        ttnn_tensor_to_torch(t2.real).to(torch.float32), ttnn_tensor_to_torch(t2.imag).to(torch.float32)
    )


def complex_angle_bw(
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
    if dtype[1] == ttnn.bfloat8_b:
        dtype[1] = ttnn.bfloat16

    t0 = setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.complex_tensor(
        setup_ttnn_tensor(y.real, device, layout[1], input_mem_config[1], dtype[1]),
        setup_ttnn_tensor(y.imag, device, layout[1], input_mem_config[1], dtype[1]),
    )

    t2 = ttnn.angle_bw(t0, t1, memory_config=output_mem_config)[0]

    return torch.complex(
        ttnn_tensor_to_torch(t2.real).to(torch.float32), ttnn_tensor_to_torch(t2.imag).to(torch.float32)
    )


def complex_is_real(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.is_real(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def complex_is_imag(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = ttnn.complex_tensor(
        setup_ttnn_tensor(x.real, device, layout[0], input_mem_config[0], dtype[0]),
        setup_ttnn_tensor(x.imag, device, layout[0], input_mem_config[0], dtype[0]),
    )

    t1 = ttnn.is_imag(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1)


def logiteps_bw(
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
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.logiteps_bw(t0, t1, eps=0.0001, memory_config=output_mem_config)[0]

    return ttnn_tensor_to_torch(t2)
